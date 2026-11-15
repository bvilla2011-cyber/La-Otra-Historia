"""Proveedores de voz: simulación, preview robótica y Qwen3 neuronal."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from .naturalness import describe_audio

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
    """Síntesis neuronal con referencias aprobables, múltiples tomas y control técnico."""
    identity = MODEL_BASE

    def __init__(self, voices_dir):
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("Instala pip install -e '.[voice]'") from exc
        if not torch.cuda.is_available():
            raise RuntimeError('Para Qwen3-TTS hace falta GPU CUDA')
        self._torch = torch
        self.voices_dir = Path(voices_dir).resolve()
        self.configs = {}
        digest = hashlib.sha256()
        for speaker in ('juan', 'carla'):
            config = json.loads((self.voices_dir / f'{speaker}.json').read_text(encoding='utf-8'))
            audio = (self.voices_dir / config['ref_audio']).resolve()
            if not audio.is_relative_to(self.voices_dir) or not audio.is_file():
                raise ValueError(f'Referencia no autorizada o inexistente: {speaker}')
            info = sf.info(audio)
            if not 3 <= info.duration <= 35 or info.channels not in (1, 2):
                raise ValueError(f'Referencia de {speaker} debe durar entre 3 y 35 segundos')
            if len(config.get('ref_text', '').split()) < 5:
                raise ValueError(f'La referencia de {speaker} necesita transcripción correcta')
            self.configs[speaker] = (config, audio)
            digest.update(audio.read_bytes())
            digest.update(config['ref_text'].encode())
        self.identity += ':' + digest.hexdigest()[:20] + ':multi-take-v1'
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_BASE, device_map='cuda:0', dtype=torch.bfloat16,
            attn_implementation='sdpa')
        self.prompts = {}
        self.last_takes = []

    def _prompt(self, speaker):
        if speaker not in self.prompts:
            config, audio = self.configs[speaker]
            self.prompts[speaker] = self.model.create_voice_clone_prompt(
                ref_audio=str(audio), ref_text=config['ref_text'],
                x_vector_only_mode=False)
        return self.prompts[speaker]

    def synthesize(self, speaker, text, path, takes=2):
        if speaker not in self.configs:
            raise ValueError('Locutor desconocido')
        if not 1 <= takes <= 5:
            raise ValueError('takes debe ser entre 1 y 5')
        key = f'{self.identity}/{speaker}/{text}'
        base = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        results = []
        for i in range(takes):
            seed = (base + i * 7919) % (2**31 - 1)
            self._torch.manual_seed(seed)
            try:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language='Spanish', voice_clone_prompt=self._prompt(speaker),
                    do_sample=True, temperature=(.84, .92, .88, .79, .97)[i],
                    top_p=.95, repetition_penalty=1.10)
                data = np.asarray(wavs[0], dtype=np.float32)
                metric = describe_audio(data, sr, len(text.split()))
                metric.update({'take': i + 1, 'seed': seed})
                results.append((metric, data, sr))
            except Exception as exc:
                results.append(({'valid': False, 'score': -1000, 'take': i + 1,
                                 'reason': f'{type(exc).__name__}: {str(exc)[:120]}'}, None, None))
        self.last_takes = [r[0] for r in results]
        valid = [r for r in results if r[0]['valid']]
        if not valid:
            raise RuntimeError(f'Ninguna toma superó QA técnica: {self.last_takes}')
        selected = max(valid, key=lambda x: x[0]['score'])
        sf.write(path, selected[1], selected[2])
        return {'selected_take': selected[0]['take'], 'attempts': self.last_takes}
