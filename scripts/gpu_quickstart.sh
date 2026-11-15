#!/usr/bin/env bash
# Ejecutar en Linux con CUDA ya instalada. No crea gastos ni alquila GPUs por sí mismo.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice,dev]'
loh doctor --strict
loh validate stories/voice_test_papa_noel.json
loh design-voices --dest voices --candidates "${CANDIDATES:-1}"
echo "Audiciones generadas en voices/auditions/{juan,carla}/. Escúchalas antes de aprobar."
echo "Para una prueba PROVISIONAL: loh approve-voice juan 1 --preview; loh approve-voice carla 1 --preview"
echo "Después: loh render stories/voice_test_papa_noel.json --provider qwen --voices voices --takes 1 --allow-short --output output/neural"
