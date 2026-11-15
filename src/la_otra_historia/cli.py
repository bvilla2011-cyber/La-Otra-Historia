"""Entrada de comandos del estudio."""
import argparse
import json
from pathlib import Path
from .schema import read_script, stats

def main():
    parser = argparse.ArgumentParser(prog="loh")
    subs = parser.add_subparsers(dest="command", required=True)
    val = subs.add_parser("validate")
    val.add_argument("script", type=Path)
    voices = subs.add_parser("design-voices")
    voices.add_argument("--dest", type=Path, default=Path("voices"))
    voices.add_argument("--candidates", type=int, default=3)
    prod = subs.add_parser("render")
    prod.add_argument("script", type=Path)
    prod.add_argument("--provider", choices=["mock", "espeak-preview", "piper-preview", "qwen"], default="mock")
    prod.add_argument("--voices", type=Path, default=Path("voices"))
    prod.add_argument("--assets", type=Path, default=Path("assets"))
    prod.add_argument("--output", type=Path, default=Path("output"))
    prod.add_argument("--allow-short", action="store_true")
    approve = subs.add_parser("approve-voice", help="Fijar una toma sintética escuchada")
    approve.add_argument("speaker", choices=["juan", "carla"])
    approve.add_argument("take", type=int)
    approve.add_argument("--voices", type=Path, default=Path("voices"))
    approve.add_argument("--preview", action="store_true", help="Selección provisional, no aprobada")
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(stats(read_script(args.script)), ensure_ascii=False, indent=2))
    elif args.command == "design-voices":
        from .voices import audition_voices
        print(json.dumps(audition_voices(args.dest, candidates=args.candidates), ensure_ascii=False, indent=2))
    elif args.command == "approve-voice":
        from .voices import approve_voice
        print(approve_voice(args.voices, args.speaker, args.take, preview=args.preview))
    else:
        from .render import render
        print(json.dumps(render(args.script, args.output, args.voices,
                                args.assets, args.provider, args.allow_short),
                         ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
