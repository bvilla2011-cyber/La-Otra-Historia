"""Esquema estricto y estadísticas de guiones."""
import json
from pathlib import Path

SPEAKERS = frozenset(("juan", "carla"))

def validate_script(script):
    if not isinstance(script, dict):
        raise ValueError("El guion debe ser un objeto")
    for name in ("id", "title", "segments"):
        if name not in script:
            raise ValueError("Falta " + name)
    if not isinstance(script["id"], str) or not script["id"].strip():
        raise ValueError("id vacío")
    if not isinstance(script["title"], str) or not script["title"].strip():
        raise ValueError("title vacío")
    segments = script["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments vacío")
    for i, s in enumerate(segments, 1):
        if not isinstance(s, dict) or s.get("speaker") not in SPEAKERS:
            raise ValueError(f"Bloque {i}: speaker debe ser juan o carla")
        if not isinstance(s.get("text"), str) or not s["text"].strip():
            raise ValueError(f"Bloque {i}: texto vacío")
        if len(s["text"]) > 1500:
            raise ValueError(f"Bloque {i}: texto supera 1500 caracteres")
        pause = s.get("pause_ms", 500)
        if isinstance(pause, bool) or not isinstance(pause, (int, float)) or not 0 <= pause <= 10000:
            raise ValueError(f"Bloque {i}: pausa no válida")

def read_script(path):
    script = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_script(script)
    return script

def stats(script, wpm=145):
    words = sum(len(s["text"].split()) for s in script["segments"])
    pauses = sum(s.get("pause_ms", 500) for s in script["segments"]) / 60000
    return {"words": words, "blocks": len(script["segments"]),
            "estimated_minutes": round(words / wpm + pauses, 2),
            "target_min_words": 5000, "target_max_words": 6000}
