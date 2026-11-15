"""Síntesis por bloques, caché y exportación podcast."""
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .schema import read_script
from .tts import MockTTS, QwenTTS, EspeakPreviewTTS
from .piper_preview import PiperPreviewTTS

SR = 48000

def resample(audio, sr):
    if sr == SR: return audio
    gcd = math.gcd(SR, sr)
    return resample_poly(audio, SR // gcd, sr // gcd, axis=0)

def stereo(audio):
    if audio.ndim == 1: return np.repeat(audio[:, None], 2, axis=1)
    if audio.shape[1] == 1: return np.repeat(audio, 2, axis=1)
    return audio[:, :2]

def render(script_path, output_dir, voices_dir, assets_dir, provider_name, allow_short=False):
    script = read_script(script_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(exist_ok=True)
    factories = {
        "mock": lambda: MockTTS(),
        "espeak-preview": lambda: EspeakPreviewTTS(),
        "qwen": lambda: QwenTTS(voices_dir),
        "piper-preview": lambda: PiperPreviewTTS(voices_dir)
    }
    if provider_name not in factories: raise ValueError("Proveedor desconocido")
    tts = factories[provider_name]()
    out_wav = output_dir / (script["id"] + ".wav")
    report = {"id": script["id"], "title": script["title"], "provider": provider_name,
              "quality": {"espeak-preview": "robotic-technical-preview", "piper-preview": "neural-technical-preview", "mock": "synthetic-tone-test"}.get(provider_name, "model-dependent"),
              "cached_segments": 0, "new_segments": 0, "segments": len(script["segments"])}
    with sf.SoundFile(out_wav, "w", samplerate=SR, channels=2, subtype="PCM_16") as master:
        for i, segment in enumerate(script["segments"], 1):
            key = json.dumps({"model": tts.identity, "speaker": segment["speaker"],
                              "text": segment["text"]}, sort_keys=True, ensure_ascii=False)
            path = cache / (hashlib.sha256(key.encode()).hexdigest() + ".wav")
            if path.exists(): report["cached_segments"] += 1
            else:
                tts.synthesize(segment["speaker"], segment["text"], path)
                report["new_segments"] += 1
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
            master.write(stereo(resample(samples, sr)))
            pause = int(SR * segment.get("pause_ms", 500) / 1000)
            if pause: master.write(np.zeros((pause, 2), dtype=np.float32))
            print(f"[{i}/{len(script['segments'])}] {segment['speaker']}", flush=True)
    report["duration_minutes"] = round(sf.info(out_wav).duration / 60, 3)
    report["meets_duration"] = 35 <= report["duration_minutes"] <= 45
    if provider_name == "qwen" and not allow_short and not report["meets_duration"]:
        raise ValueError("El render neuronal no cumple 35–45 minutos; revisa antes de publicar")
    if shutil.which("ffmpeg"):
        out_mp3 = output_dir / (script["id"] + ".mp3")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(out_wav), "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(out_mp3)], check=True)
        report["mp3"] = str(out_mp3)
    report["wav"] = str(out_wav)
    (output_dir / (script["id"] + ".report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
