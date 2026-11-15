"""Proveedores de voz: simulación, preview robótica y Qwen3 neuronal."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

class MockTTS:
    identity = "mock-v1"
    def __init__(self, **kwargs): pass
    def synthesize(self, speaker, text, path):
        sr = 24000
        duration = min(3, max(.4, len(text.split()) / 30))
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        f = 260 if speaker == "juan" else 390
        fade = np.minimum(1, np.minimum(t / .05, (duration - t) / .05)).clip(0, 1)
        sf.write(path, (.06 * np.sin(2 * np.pi * f * t) * fade).astype(np.float32), sr)

class EspeakPreviewTTS:
    """Voces humanas audibles, pero ROBÓTICAS; solo prueba del montaje."""
    identity = "espeak-preview-es-2026-v1"
    VOICES = {"juan": ("es+m3", "143", "37"), "carla": ("es+f3", "152", "60")}
    def __init__(self, **kwargs):
        if not shutil.which("espeak"):
            raise RuntimeError("Instala espeak para la prueba CPU")
    def synthesize(self, speaker, text, path):
        if speaker not in self.VOICES:
            raise ValueError("speaker desconocido")
        voice, speed, pitch = self.VOICES[speaker]
        subprocess.run(["espeak", "-v", voice, "-s", speed, "-p", pitch,
                        "-a", "130", "--stdin", "-w", str(path)],
                       input=text, text=True, check=True, capture_output=True)

class QwenTTS:
    identity = MODEL_BASE
    def __init__(self, voices_dir):
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise RuntimeError("Instala pip install -e '.[voice]'") from e
        if not torch.cuda.is_available():
            raise RuntimeError("Para Qwen3-TTS se necesita una GPU CUDA")
        self.voices_dir = Path(voices_dir)
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        h = hashlib.sha256()
        for speaker in ("juan", "carla"):
            h.update((self.voices_dir / f"{speaker}.wav").read_bytes())
        self.identity += ":" + h.hexdigest()[:16]
        self.prompts = {}
    def _prompt(self, speaker):
        if speaker not in self.prompts:
            config = json.loads((self.voices_dir / f"{speaker}.json").read_text(encoding="utf-8"))
            audio = (self.voices_dir / config["ref_audio"]).resolve()
            if not audio.is_relative_to(self.voices_dir.resolve()):
                raise ValueError("Ruta de referencia no autorizada")
            self.prompts[speaker] = self.model.create_voice_clone_prompt(
                ref_audio=str(audio), ref_text=config["ref_text"], x_vector_only_mode=False)
        return self.prompts[speaker]
    def synthesize(self, speaker, text, path):
        wavs, sr = self.model.generate_voice_clone(
            text=text, language="Spanish", voice_clone_prompt=self._prompt(speaker))
        sf.write(path, wavs[0], sr)
