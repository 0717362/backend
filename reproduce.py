"""Re-render an archived blueprint without an LLM; neural modes need an ACE endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audio import generate_symbolic, mix_background, normalize_audio
from .models import GenerateRequest, MusicBlueprint
from .planner import load_catalog, validate_plan
from .providers import generate_ace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("reproduced"))
    parser.add_argument("--mode", choices=["symbolic", "ace_step", "hybrid"], default="symbolic")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be a non-negative 32-bit integer")
    blueprint = MusicBlueprint.model_validate_json(args.blueprint.read_text(encoding="utf-8"))
    request = GenerateRequest(expression=blueprint.original_expression,
                              instruments=[i.id for i in blueprint.instruments],
                              duration_seconds=blueprint.duration_seconds, allow_approximation=True,
                              mode=args.mode)
    blueprint = validate_plan(blueprint, request)
    if args.mode == "symbolic":
        result = generate_symbolic(blueprint, load_catalog(), args.out, args.seed)
    else:
        source = generate_ace(blueprint, args.out, args.seed, background=args.mode == "hybrid")
        if args.mode == "hybrid":
            stems = generate_symbolic(blueprint, load_catalog(), args.out / "controlled", args.seed)
            result = mix_background(source, stems["wav_path"], args.out, blueprint.duration_seconds)
            result.update(midi_path=stems["midi_path"], instrument_evidence=stems["instrument_evidence"])
        else:
            result = normalize_audio(source, args.out, blueprint.duration_seconds)
    hashes = {name: hashlib.sha256(result[name].read_bytes()).hexdigest()
              for name in ("midi_path", "wav_path", "mp3_path") if name in result}
    report = {"mode": args.mode, "seed": args.seed, "metrics": result["metrics"], "sha256": hashes,
              "instrument_evidence": result.get("instrument_evidence", []),
              "note": "Replays the archived blueprint without an LLM. Neural audio requires the pinned model/device; cross-device bit identity is not promised."}
    (args.out / "reproduction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "duration": result["duration_seconds"], "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
