"""Preparación de textos y métricas objetivas de tomas. No evalúa humanización perceptual."""
from __future__ import annotations
import re
import unicodedata
import numpy as np

_SENTENCE = re.compile(r'(?<=[.!?…])\s+(?=[¿¡«“"A-ZÁÉÍÓÚÜÑ0-9])')
_CLAUSE = re.compile(r'(?<=[,;:])\s+')

def _pack(parts: list[str], limit: int) -> list[str]:
    out, current = [], ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = (current + ' ' + part).strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                out.append(current)
            current = part
    if current:
        out.append(current)
    return out

def split_narration(text: str, max_chars: int = 250) -> list[str]:
    """Separa en frases de longitud estable, conservando signos y palabras."""
    if max_chars < 45:
        raise ValueError('max_chars debe ser >=45')
    text = unicodedata.normalize('NFC', text).strip()
    if not text:
        raise ValueError('Texto vacío')
    if len(text) <= max_chars:
        return [text]
    parts = []
    for sentence in _SENTENCE.split(text):
        if len(sentence) <= max_chars:
            parts.append(sentence)
            continue
        for clause in _CLAUSE.split(sentence):
            if len(clause) <= max_chars:
                parts.append(clause)
            else:
                parts.extend(_pack(clause.split(), max_chars))
    out = _pack(parts, max_chars)
    if any(len(part) > max_chars for part in out):
        raise ValueError('Una palabra supera el máximo de caracteres')
    return out

def describe_audio(audio: np.ndarray, sample_rate: int, words: int) -> dict:
    """Control técnico de volumen, recortes y duración. NO mide naturalidad."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sample_rate <= 0 or x.size == 0 or not np.isfinite(x).all():
        return {'valid': False, 'reason': 'audio vacío, no finito o frecuencia incorrecta', 'score': -1000}
    duration = x.size / sample_rate
    absx = np.abs(x)
    peak = float(np.max(absx))
    rms = float(np.sqrt(np.mean(x ** 2)))
    clipped_fraction = float(np.mean(absx >= .985))
    frame = max(1, int(sample_rate * .025))
    frame_rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2)) for i in range(0, x.size, frame)])
    quiet_fraction = float(np.mean(frame_rms < max(.001, rms * .075)))
    wpm = (60 * words / duration) if duration else 0
    valid = (peak > .008 and rms > .002 and clipped_fraction < .01
             and 65 <= wpm <= 260 and quiet_fraction < .72)
    score = 100 - 600 * clipped_fraction - 65 * quiet_fraction
    score -= max(0, 105 - wpm) * .22 + max(0, wpm - 205) * .22
    score -= max(0, .045 - rms) * 180
    if not valid:
        score -= 200
    return {'valid': bool(valid), 'score': round(float(score), 3),
            'duration_s': round(float(duration), 3), 'wpm': round(float(wpm), 1),
            'rms': round(rms, 5), 'peak': round(peak, 5),
            'clipped_fraction': round(clipped_fraction, 5),
            'quiet_fraction': round(quiet_fraction, 4)}

def soften_edges(audio: np.ndarray, sample_rate: int, fade_ms: int = 12) -> np.ndarray:
    """Suaviza posibles clics sin modificar tono o ritmo."""
    x = np.array(audio, dtype=np.float32, copy=True)
    n = min(int(sample_rate * fade_ms / 1000), len(x) // 3)
    if n > 1:
        env = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= env[:, None] if x.ndim == 2 else env
        x[-n:] *= env[::-1, None] if x.ndim == 2 else env[::-1]
    return x
