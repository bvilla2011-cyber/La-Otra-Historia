"""Proveedor de previsualización neuronal español en CPU.

Utiliza el modelo externo es_ES-sharvard-medium (Piper >=1.5).
Modelo de dos hablantes: M=0, F=1, validado al abrir.
El modelo no se distribuye con el repositorio.
"""
from __future__ import annotations
import hashlib
import json
import wave
from pathlib import Path

MODEL_NAME = "es_ES-sharvard-medium"

class PiperPreviewTTS:
    """Voz neuronal básica, no es equivalente a una narración profesional."""
    def __init__(self, voices_dir):
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError as exc:
            raise RuntimeError("Instala pip install -e '.[piper]'") from exc
        models = Path(voices_dir)
        model_path = models / (MODEL_NAME + ".onnx")
        config_path = models / (MODEL_NAME + ".onnx.json")
        if not model_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                "Falta el modelo Piper. Ejecuta: python -m piper.download_voices "
                "--data-dir voices es_ES-sharvard-medium"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        speakers = config.get("speaker_id_map", {})
        if config.get("num_speakers", 0) < 2 or speakers.get("M") != 0 or speakers.get("F") != 1:
            raise ValueError("Piper sharvard no identifica M=0 y F=1; revisa la descarga.")
        self._SynthesisConfig = SynthesisConfig
        self.voice = PiperVoice.load(str(model_path))
        digest = hashlib.sha256()
        digest.update(model_path.read_bytes())
        digest.update(config_path.read_bytes())
        self.identity = f"piper-neural-preview:{MODEL_NAME}:{digest.hexdigest()[:20]}"
        self.speakers = {"juan": 0, "carla": 1}

    def synthesize(self, speaker, text, path):
        if speaker not in self.speakers:
            raise ValueError(f"Locutor desconocido: {speaker}")
        cfg = self._SynthesisConfig(
            speaker_id=self.speakers[speaker],
            length_scale=1.09 if speaker == "juan" else 1.05,
            noise_scale=0.55,
            noise_w_scale=0.65,
        )
        with wave.open(str(path), "wb") as wav:
            self.voice.synthesize_wav(text, wav, syn_config=cfg)
