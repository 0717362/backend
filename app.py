"""Local music API: persisted jobs, bounded worker, audio and expression cards."""
from __future__ import annotations

import html
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import ValidationError

from .audio import _binary, generate_symbolic, mix_background, normalize_audio
from .cards import render_card
from .evaluation import evaluate_candidate
from .models import CardSelection, GenerateRequest, MusicBlueprint, ReplyContext
from .planner import canonical_instruments, load_catalog, resolve_mode, validate_plan
from .curation import facets
from .providers import generate_ace, plan_expression
from .poster import artwork_for, cover_asset, public_card_url as validated_public_url

HERE = Path(__file__).resolve().parent
DATA = Path(os.getenv("MUSIC_DATA_DIR", str(HERE / "runtime"))).resolve()
DB = DATA / "jobs.sqlite3"
ACTIVE = ("queued", "planning", "generating", "evaluating")
SUBMIT_LOCK = threading.Lock()
MAX_PENDING = 8
log = logging.getLogger(__name__)


@contextmanager
def connection():
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize():
    DATA.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
            progress REAL NOT NULL, request TEXT NOT NULL, result TEXT, error TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS blueprint_confirmations (
            id TEXT PRIMARY KEY, request TEXT NOT NULL, blueprint TEXT NOT NULL)""")
        conn.execute("UPDATE jobs SET status='failed', error=?, progress=1 WHERE status IN (?,?,?,?)",
                     ("上次服务中断，请重新提交生成。", *ACTIVE))


def update(job_id: str, status: str, progress: float, *, result=None, error=None):
    with connection() as conn:
        conn.execute("UPDATE jobs SET status=?,progress=?,result=?,error=? WHERE id=?",
                     (status, progress, json.dumps(result, ensure_ascii=False) if result else None,
                      error, job_id))


def fetch_job(job_id: str):
    with connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "找不到生成任务")
    result = json.loads(row["result"]) if row["result"] else None
    if result:
        result["card"].setdefault("scene", result["blueprint"].get("scene", ""))
        result["card"] = present_card(result["card"])
    return {"id": row["id"], "created_at": row["created_at"], "status": row["status"],
            "progress": row["progress"], "error": row["error"],
            "result": result}


def media_url(path: Path) -> str:
    return "/api/audio/" + Path(path).resolve().relative_to(DATA).as_posix()


def public_card_url(card_id: str) -> str:
    base = os.getenv("MUSIC_PUBLIC_BASE_URL", "").rstrip("/")
    try:
        parsed = urlsplit(base)
        if parsed.query or parsed.fragment: return ""
    except ValueError:
        return ""
    # The deployed React card and its QR destination must use the same scene.
    # Keep the classic server-rendered card available for API-only deployments.
    route = "/cards/" if os.getenv("MUSIC_PUBLIC_CARD_ROUTE") == "cinema" else "/share/"
    return validated_public_url(base + route + card_id)


def reply_context(req: GenerateRequest) -> ReplyContext | None:
    if not req.parent_card_id:
        return None
    parent = fetch_job(req.parent_card_id)
    if parent["status"] != "completed":
        raise ValueError("音乐回应的原作品尚未完成")
    blueprint = parent["result"]["blueprint"]
    return ReplyContext(card_id=req.parent_card_id, **{field: blueprint[field] for field in (
        "original_expression", "scene", "core_emotions", "emotion_curve", "bpm", "key", "scale", "instruments")})


def present_card(data: dict) -> dict:
    card_id = data["id"]
    artwork = artwork_for(data)
    return {**data, "english_expression": data.get("english_expression", ""),
            "template_id": "sound-radio-reference-v1", "public_url": public_card_url(card_id),
            "poster_url": f"/api/cards/{card_id}/poster.png", "artwork": artwork,
            "cover_url": "/api/card-assets/" + artwork["asset_name"]}


def confirmed_blueprint(req: GenerateRequest) -> MusicBlueprint:
    """Accept only revisions to a blueprint actually issued by this server."""
    supplied = req.blueprint
    if supplied is None or not supplied.confirmation_id:
        raise ValueError("请先请求音乐蓝图，再确认生成；蓝图缺少服务器确认记录")
    with connection() as conn:
        row = conn.execute("SELECT request,blueprint FROM blueprint_confirmations WHERE id=?",
                           (supplied.confirmation_id,)).fetchone()
    if row is None:
        raise ValueError("音乐蓝图确认记录不存在，请重新理解这段表达")
    original_req = GenerateRequest.model_validate_json(row["request"])
    original = MusicBlueprint.model_validate_json(row["blueprint"])
    for field in ("expression", "instruments", "duration_seconds", "mode", "allow_approximation", "parent_card_id"):
        if getattr(req, field) != getattr(original_req, field):
            raise ValueError("表达、乐器、时长或回应对象已变化，请重新生成音乐蓝图")
    editable = {"scene", "core_emotions", "bpm", "dynamics"}
    incoming, baseline = supplied.model_dump(), original.model_dump()
    if any(incoming[key] != baseline[key] for key in baseline if key not in editable):
        raise ValueError("蓝图来源、曲式、乐器或时间结构不可由客户端替换，请重新请求蓝图")
    revision = {key: incoming[key] for key in editable if incoming[key] != baseline[key]}
    if not revision:
        return validate_plan(original, req)
    catalog = load_catalog()
    duration = original.duration_seconds
    performance = "; ".join(f"{catalog[p.id]['name_en']}: {p.role}, from {p.entry_seconds:g}s to {p.exit_seconds:g}s"
                            for p in original.instruments)
    prompt = (
        f"Instrumental only, no singing or spoken words. {duration:g} seconds, {supplied.bpm:g} BPM, "
        f"{original.key}, {original.scale} mode. Only these instruments: "
        + ", ".join(catalog[p.id]["name_en"] for p in original.instruments)
        + f". User-confirmed scene: {supplied.scene}. User-confirmed emotions: "
        + ", ".join(e.label for e in supplied.core_emotions)
        + f". Dynamics and direction: {supplied.dynamics}. {original.density} note density; space: {original.space}. "
        + f"Arrangement targets: {performance}. A short memorable motif, breathing rests, natural ending. "
        + "Do not add instruments beyond the specified set."
    )
    revised = MusicBlueprint.model_validate({**baseline, **revision, "generation_prompt": prompt,
        "planner_source": original.planner_source + "+user_revision",
        "warnings": [*original.warnings, "用户已修订表达蓝图；情绪曲线仍为原理解阶段的创作规划，不是情绪测量。"]})
    return validate_plan(revised, req)


def candidate_warnings(blueprint: dict, candidate: dict) -> list[str]:
    warnings = [*blueprint.get("warnings", []), *candidate["evaluation"].get("warnings", [])]
    if candidate["mode"] == "ace_step":
        warnings.append("ACE-Step乐器提示为软约束，未核验实际乐器出现情况。")
    elif candidate["mode"] == "hybrid":
        warnings.append("指定乐器独立音轨已渲染；生成背景的乐器、调性与节拍仍需试听。")
    return warnings


def run_job(job_id: str, req: GenerateRequest):
    try:
        update(job_id, "planning", 0.05)
        blueprint = req.blueprint or plan_expression(req, reply_context=reply_context(req))
        catalog = load_catalog()
        # Mode is frozen on submission so later config changes cannot switch the route.
        mode = resolve_mode(req, bool(os.getenv("ACE_STEP_BASE_URL")))
        candidates = []
        for i in range(req.candidates):
            update(job_id, "generating", 0.1 + 0.7 * i / req.candidates)
            out_dir = DATA / job_id / f"candidate-{i + 1}"
            seed = (req.seed + i * 9973) % 2147483648
            if mode == "symbolic":
                audio = generate_symbolic(blueprint, catalog, out_dir, seed)
            else:
                source = generate_ace(blueprint, out_dir, seed, background=mode == "hybrid")
                if mode == "hybrid":
                    controlled = generate_symbolic(blueprint, catalog, out_dir / "controlled", seed)
                    audio = mix_background(source, controlled["wav_path"], out_dir,
                                           blueprint.duration_seconds)
                    audio.update({k: controlled[k] for k in (
                        "midi_path", "stems", "rendered_instruments", "instrument_evidence"
                    )})
                else:
                    audio = normalize_audio(source, out_dir, blueprint.duration_seconds)
                    audio.update(rendered_instruments=[], instrument_evidence=[], stems=[])
            update(job_id, "evaluating", 0.1 + 0.7 * (i + 1) / req.candidates)
            scores = evaluate_candidate(blueprint, audio, catalog)
            candidate = {"id": f"candidate-{i + 1}", "seed": seed, "mode": mode,
                         "audio_url": media_url(audio["mp3_path"]),
                         "wav_url": media_url(audio["wav_path"]), "evaluation": scores,
                         "instrument_evidence": audio.get("instrument_evidence", []),
                         "stems": []}
            if audio.get("midi_path"):
                candidate["midi_url"] = media_url(audio["midi_path"])
            for stem in audio.get("stems", []):
                candidate["stems"].append({k: v for k, v in stem.items()
                                           if not isinstance(v, Path) and not k.endswith("path")})
                path = stem.get("wav_path") or stem.get("path")
                if path:
                    candidate["stems"][-1]["audio_url"] = media_url(path)
            candidates.append(candidate)
        best = max(candidates, key=lambda c: c["evaluation"]["score"])
        instruments = []
        for assigned in blueprint.instruments:
            item = catalog[assigned.id]
            instruments.append({"id": assigned.id, "name_zh": item["name_zh"],
                                "name_en": item["name_en"], "role": assigned.role,
                                "reason": assigned.reason, "story": item.get("story", ""),
                                "culture": item.get("culture", ""),
                                "sound_family": item["sound_family"],
                                "cultural_region": item["cultural_region"],
                                "heritage": item["heritage"],
                                "render": item["render"]})
        warning = candidate_warnings(blueprint.model_dump(), best)
        job = fetch_job(job_id)
        card = {"id": job_id, "title": " · ".join(e.label for e in blueprint.core_emotions[:2]),
                "english_expression": blueprint.english_expression,
                "original_expression": req.expression, "created_at": job["created_at"],
                "scene": blueprint.scene, "artwork_theme": req.artwork_theme, "artwork_motion": req.artwork_motion,
                "duration_seconds": blueprint.duration_seconds,
                "audio_url": best["audio_url"], "wav_url": best["wav_url"],
                "waveform": best["evaluation"].get("waveform", []),
                "selected_candidate_id": best["id"],
                "instrument_evidence": best.get("instrument_evidence", []),
                "instruments": instruments, "emotion_tags": [e.label for e in blueprint.core_emotions],
                "emotion_curve": [e.model_dump() for e in blueprint.emotion_curve],
                "share_url": f"/share/{job_id}", "parent_card_id": req.parent_card_id,
                "planner_source": blueprint.planner_source, "generation_mode": mode,
                "warnings": warning}
        if blueprint.reply_context is not None:
            card["reply_context"] = blueprint.reply_context.model_dump()
        result = {"blueprint": blueprint.model_dump(), "candidates": candidates,
                  "best_candidate_id": best["id"], "selected_candidate_id": best["id"], "card": card}
        (DATA / job_id / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        update(job_id, "completed", 1, result=result)
    except Exception as exc:
        log.exception("Music job failed: %s", job_id)
        message = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else "音乐生成失败，请检查本地服务日志。"
        update(job_id, "failed", 1, error=message[:500])


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    # ponytail: one CPU/model worker, move to a durable queue when multiple processes are needed.
    app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music")
    yield
    app.state.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="情绪音乐表达 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv(
    "MUSIC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
).split(",") if x.strip()], allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])


@app.get("/api/health")
def health():
    def tool_present(name):
        try:
            _binary(name)
            return True
        except RuntimeError:
            return False
    return {"status": "ok", "planner": "llm_configured" if os.getenv("MUSIC_LLM_BASE_URL") or os.getenv("MUSIC_LLM_USE_WORKSTATION") == "1" else "rules_fallback",
            "ace_step": "configured_unverified" if os.getenv("ACE_STEP_BASE_URL") else "not_configured",
            "fluidsynth": tool_present("fluidsynth"),
            "ffmpeg": tool_present("ffmpeg"), "clap": "configured_unverified" if os.getenv("MUSIC_CLAP_BASE_URL") or os.getenv("MUSIC_CLAP_MODEL_DIR") else "not_configured",
            "workers": 1, "max_pending": MAX_PENDING}


@app.get("/api/instruments")
def instruments(q: str = Query(default="", max_length=100),
                capability: Literal["any", "symbolic", "ace_step"] = "any",
                sound_family: str = Query(default="", max_length=40),
                cultural_region: str = Query(default="", max_length=40),
                heritage: Literal["any", "verified", "source_reference"] = "any"):
    items = list(load_catalog().values())
    selected = [item for item in items if q.casefold() in " ".join(
        [item["id"], item["label"], item["name_en"], item["culture"],
         item["sound_family"], item["cultural_region"], item['heritage']['project'],
         *item["aliases"]]).casefold()]
    selected = [item for item in selected if (not sound_family or item['sound_family'] == sound_family)
                and (not cultural_region or item['cultural_region'] == cultural_region)
                and (heritage == 'any' or item['heritage']['status'] == heritage)]
    if capability != "any":
        selected = [item for item in selected if item.get("generation_capabilities", {}).get(capability, False)]
    return {"instruments": selected, "total_count": len(items), "returned_count": len(selected),
            "families": sorted({item["family"] for item in items}),
            "facets": facets(items),
            "capability_note": "symbolic表示已有音源；ace_step表示可作为生成提示，实际音色未核验。"}


@app.get("/api/instrument-graph")
def instrument_graph():
    nodes, edges = {}, []
    for item in load_catalog().values():
        source = "instrument:" + item["id"]
        nodes[source] = {**item, "id": source, "type": "instrument"}
        for field, kind, relation in (("suitable_emotions", "emotion", "can_express"),
                                      ("suitable_scenes", "scene", "suits_scene"),
                                      ("typical_roles", "role", "can_play_role"),
                                      ("culture", "culture", "cultural_context")):
            values = item[field] if isinstance(item[field], list) else [item[field]]
            for value in values:
                target = kind + ":" + value
                nodes[target] = {"id": target, "type": kind, "label": value}
                edges.append({"source": source, "target": target, "relation": relation})
        for field, relation in (('sound_family', 'sound_family'), ('cultural_region', 'exhibition_region')):
            target = field + ':' + item[field]
            nodes[target] = {'id': target, 'type': field, 'label': item[field]}
            edges.append({'source': source, 'target': target, 'relation': relation})
        if item['heritage']['status'] == 'verified':
            target = 'heritage:' + item['heritage']['project']
            nodes[target] = {**item['heritage'], 'id': target, 'type': 'heritage'}
            edges.append({'source': source, 'target': target, 'relation': 'associated_living_tradition'})
    return {"nodes": list(nodes.values()), "edges": edges,
            "interpretation": "Curated creative associations; not empirical probabilities or fixed emotion rules."}


def canonical_request(req: GenerateRequest):
    try:
        req = req.model_copy(update={"instruments": canonical_instruments(req)})
        req = req.model_copy(update={"mode": resolve_mode(req, bool(os.getenv("ACE_STEP_BASE_URL")))})
        from .planner import plan
        if req.blueprint is None:
            validate_plan(plan(req), req)
        else:
            req = req.model_copy(update={"blueprint": confirmed_blueprint(req)})
        if req.parent_card_id:
            parent = fetch_job(req.parent_card_id)
            if parent["status"] != "completed":
                raise ValueError("音乐回应的原作品尚未完成")
        if req.mode in {"ace_step", "hybrid"} and not os.getenv("ACE_STEP_BASE_URL"):
            raise ValueError("ACE-Step尚未配置；本机可选择symbolic模式生成真实音频。")
        return req
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/blueprints", response_model=MusicBlueprint)
def blueprint(req: GenerateRequest):
    try:
        if req.blueprint is not None:
            raise HTTPException(422, "理解接口不接受已确认蓝图；请提交原始表达")
        canonical = canonical_request(req)
        planned = plan_expression(canonical, reply_context=reply_context(canonical))
        planned = planned.model_copy(update={"confirmation_id": uuid.uuid4().hex})
        with connection() as conn:
            conn.execute("INSERT INTO blueprint_confirmations VALUES (?,?,?)",
                         (planned.confirmation_id, canonical.model_dump_json(), planned.model_dump_json()))
        return planned
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/generations", status_code=202)
def generate(req: GenerateRequest):
    req = canonical_request(req)
    with SUBMIT_LOCK:
        with connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN (?,?,?,?)", ACTIVE).fetchone()[0]
            if count >= MAX_PENDING:
                raise HTTPException(429, "生成队列已满，请稍后重试")
            job_id = uuid.uuid4().hex
            created = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
                         (job_id, created, "queued", 0, req.model_dump_json(), None, None))
        app.state.executor.submit(run_job, job_id, req)
    return {"id": job_id, "status": "queued", "status_url": f"/api/generations/{job_id}"}


@app.get("/api/generations/{job_id}")
def generation(job_id: str):
    return fetch_job(job_id)


@app.get("/api/cards/{card_id}")
def card(card_id: str):
    job = fetch_job(card_id)
    if job["status"] != "completed":
        raise HTTPException(409, "音乐卡片尚未生成")
    return job["result"]["card"]


async def bounded_selection(request: Request) -> CardSelection:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 256:
            raise HTTPException(413, "候选选择请求过大")
        body.extend(chunk)
    try:
        return CardSelection.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(422, "候选选择需包含有效 candidate_id，且不能添加其他字段") from exc


@app.post("/api/cards/{card_id}/selection", openapi_extra={"requestBody": {
    "required": True, "content": {"application/json": {"schema": CardSelection.model_json_schema()}}}})
def select_candidate(card_id: str, selection: CardSelection = Depends(bounded_selection)):
    with SUBMIT_LOCK:
        job = fetch_job(card_id)
        if job["status"] != "completed":
            raise HTTPException(409, "音乐生成尚未完成，无法选择候选")
        result = job["result"]
        chosen = next((c for c in result["candidates"] if c["id"] == selection.candidate_id), None)
        if chosen is None:
            raise HTTPException(422, "该候选不属于这次音乐生成")
        result["selected_candidate_id"] = chosen["id"]
        result["card"].update(audio_url=chosen["audio_url"], wav_url=chosen["wav_url"],
            waveform=chosen["evaluation"].get("waveform", []), selected_candidate_id=chosen["id"],
            instrument_evidence=chosen.get("instrument_evidence", []),
            warnings=candidate_warnings(result["blueprint"], chosen))
        result_path = DATA / job["id"] / "result.json"
        temporary = result_path.with_suffix(".selection.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(result_path)
        update(job["id"], "completed", 1, result=result)
        return present_card(result["card"])


@app.get("/api/cards/{card_id}/replies")
def replies(card_id: str, limit: int = Query(default=20, ge=1, le=50)):
    card(card_id)
    with connection() as conn:
        where = "status='completed' AND json_extract(request,'$.parent_card_id')=?"
        total = conn.execute("SELECT COUNT(*) FROM jobs WHERE " + where, (card_id,)).fetchone()[0]
        rows = conn.execute("SELECT result FROM jobs WHERE " + where + " ORDER BY created_at,id LIMIT ?",
                            (card_id, limit)).fetchall()
    items = [present_card(json.loads(row["result"])["card"]) for row in rows]
    return {"parent_card_id": card_id, "total_count": total, "returned_count": len(items), "items": items}


@app.get("/api/card-assets/{name}")
def card_asset(name: str):
    directory = HERE / "assets" / "card"
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or path.suffix.lower() not in {".png", ".jpg", ".webp"} or not path.is_file():
        raise HTTPException(404, "找不到卡片图片")
    return FileResponse(path)


@app.get("/api/cards/{card_id}/poster.png")
def poster(card_id: str):
    from .poster import render_poster
    data = card(card_id)
    stamp = (HERE / "poster.py").stat().st_mtime_ns, cover_asset(data).stat().st_mtime_ns
    key = hashlib.sha256(json.dumps([data, stamp], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    path = DATA / card_id / f"poster-{key}.png"
    if not path.is_file():
        render_poster(data, path, public_url=data["public_url"])
    return FileResponse(path, media_type="image/png", filename=f"music-card-{card_id}.png")


@app.get("/api/cards/{card_id}/share")
def share_payload(card_id: str):
    data = card(card_id)
    return {"template_id": data["template_id"], "title": data["title"],
            "text": data["original_expression"], "public_url": data["public_url"],
            "poster_url": data["poster_url"], "offline_html_url": f"/api/cards/{card_id}/download",
            "artwork": data["artwork"],
            "audio_url": data["audio_url"], "qr_status": "configured" if data["public_url"] else "awaiting_public_url",
            "note": "PNG通过二维码连接音乐；离线HTML内嵌音乐。系统分享由前端调用，不会自动向他人发送。"}


@app.get("/api/audio/{relative_path:path}")
def audio_file(relative_path: str):
    path = (DATA / relative_path).resolve()
    if not path.is_relative_to(DATA) or path.suffix not in {".wav", ".mp3", ".mid"} or not path.is_file():
        raise HTTPException(404, "找不到音频文件")
    return FileResponse(path)


@app.get("/share/{card_id}", response_class=HTMLResponse)
def share(card_id: str):
    return render_card(card(card_id))


@app.get("/api/cards/{card_id}/download")
def download_card(card_id: str):
    data = card(card_id)
    source = (DATA / data["audio_url"].removeprefix("/api/audio/")).read_bytes()
    return Response(render_card(data, embedded_audio=source), media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="music-card-{card_id}.html"'})


@app.get("/api/cards/{card_id}/bundle")
def download_bundle(card_id: str):
    import hashlib
    job = fetch_job(card_id)
    data = card(card_id)
    result = job["result"]
    files = {}
    for label, url in (("music.mp3", data["audio_url"]), ("music.wav", data["wav_url"])):
        files[label] = (DATA / url.removeprefix("/api/audio/")).read_bytes()
    files["card.html"] = render_card(data, embedded_audio=files["music.mp3"]).encode("utf-8")
    from .poster import render_poster
    poster_path = DATA / card_id / "review-poster.png"
    render_poster(data, poster_path, public_url=data["public_url"])
    files["poster.png"] = poster_path.read_bytes()
    files["share.json"] = json.dumps(share_payload(card_id), ensure_ascii=False, indent=2).encode("utf-8")
    files["blueprint.json"] = json.dumps(result["blueprint"], ensure_ascii=False, indent=2).encode("utf-8")
    with connection() as conn:
        request = conn.execute("SELECT request FROM jobs WHERE id=?", (card_id,)).fetchone()[0]
    files["request.json"] = request.encode("utf-8")
    files["candidates.json"] = json.dumps(result["candidates"], ensure_ascii=False, indent=2).encode("utf-8")
    files["reference-model-versions.json"] = (HERE / "assets" / "model-versions.json").read_bytes()
    for candidate in result["candidates"]:
        prefix = f"candidates/{candidate['id']}"
        for label, url in (("music.mp3", candidate["audio_url"]), ("music.wav", candidate["wav_url"])):
            files[f"{prefix}/{label}"] = (DATA / url.removeprefix("/api/audio/")).read_bytes()
        source = (DATA / candidate["wav_url"].removeprefix("/api/audio/")).parent / "ace-source.wav"
        if source.is_file():
            files[f"{prefix}/ace-source.wav"] = source.read_bytes()
        if candidate.get("midi_url"):
            files[f"{prefix}/music.mid"] = (DATA / candidate["midi_url"].removeprefix("/api/audio/")).read_bytes()
        for stem in candidate["stems"]:
            files[f"{prefix}/stems/{stem['id']}.wav"] = (DATA / stem["audio_url"].removeprefix("/api/audio/")).read_bytes()
    best = next(c for c in result["candidates"] if c["id"] == result["best_candidate_id"])
    selected_id = result.get("selected_candidate_id", result["best_candidate_id"])
    selected = next(c for c in result["candidates"] if c["id"] == selected_id)
    if selected.get("midi_url"):
        files["music.mid"] = (DATA / selected["midi_url"].removeprefix("/api/audio/")).read_bytes()
    for i, stem in enumerate(selected["stems"]):
        files[f"stems/{i + 1}-{stem['id']}.wav"] = (DATA / stem["audio_url"].removeprefix("/api/audio/")).read_bytes()
    files["manifest.json"] = json.dumps({"generation_mode": data["generation_mode"],
        "best_seed": best["seed"], "best_candidate_id": best["id"],
        "selected_seed": selected["seed"], "selected_candidate_id": selected["id"],
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        "reproduction": f"Run python -m backend.reproduce --blueprint blueprint.json --seed {selected['seed']} --mode {data['generation_mode']} --out reproduced. Neural mode requires ACE_STEP_BASE_URL; cross-device bit identity is not promised."},
        indent=2).encode("utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for name, contents in files.items():
            output.writestr(name, contents)
    return Response(archive.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="music-review-{card_id}.zip"'})


@app.get("/demo")
@app.get("/")
def demo():
    return FileResponse(HERE / "demo.html")
