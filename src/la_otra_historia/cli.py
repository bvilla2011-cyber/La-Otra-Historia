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
    prod = subs.add_parser("render")
    prod.add_argument("script", type=Path)
    prod.add_argument("--provider", choices=["mock", "espeak-preview", "qwen"], default="mock")
    prod.add_argument("--voices", type=Path, default=Path("voices"))
    prod.add_argument("--assets", type=Path, default=Path("assets"))
    prod.add_argument("--output", type=Path, default=Path("output"))
    prod.add_argument("--allow-short", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(stats(read_script(args.script)), ensure_ascii=False, indent=2))
    elif args.command == "design-voices":
        from .voices import design_voices
        design_voices(args.dest)
    else:
        from .render import render
        print(json.dumps(render(args.script, args.output, args.voices,
                                args.assets, args.provider, args.allow_short),
                         ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
