"""Producción por fragmentos: caché, tomas, micropausas y mezcla final."""
from __future__ import annotations
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .naturalness import describe_audio, soften_edges, split_narration
from .schema import read_script
from .tts import MockTTS, QwenTTS, EspeakPreviewTTS
from .piper_preview import PiperPreviewTTS

SR = 48000

def resample(audio, sr):
    if sr == SR:
        return audio
    factor = math.gcd(SR, sr)
    return resample_poly(audio, SR // factor, sr // factor, axis=0)

def stereo(audio):
    if audio.ndim == 1:
        return np.repeat(audio[:, None], 2, axis=1)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]

def render(script_path, output_dir, voices_dir, assets_dir, provider_name,
           allow_short=False, takes=2, max_chars=260):
    script = read_script(script_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / 'cache'
    cache.mkdir(exist_ok=True)
    factories = {
        'mock': lambda: MockTTS(),
        'espeak-preview': lambda: EspeakPreviewTTS(),
        'piper-preview': lambda: PiperPreviewTTS(voices_dir),
        'qwen': lambda: QwenTTS(voices_dir),
    }
    if provider_name not in factories:
        raise ValueError('Proveedor desconocido')
    if not 1 <= takes <= 5:
        raise ValueError('takes debe estar entre 1 y 5')
    tts = factories[provider_name]()
    if provider_name == 'qwen' and not allow_short:
        if not all(c.get('approved', False) for c, _ in tts.configs.values()):
            raise ValueError('Antes de producir la temporada aprueba Juan y Carla por escucha')
    out_wav = output_dir / (script['id'] + '.wav')
    report = {'id': script['id'], 'title': script['title'], 'provider': provider_name,
              'quality': {'espeak-preview': 'robotic-technical-preview',
                          'piper-preview': 'neural-technical-preview',
                          'mock': 'synthetic-tone-test'}.get(provider_name, 'model-dependent'),
              'cached_segments': 0, 'new_segments': 0,
              'segments': len(script['segments']),
              'takes_per_chunk': takes if provider_name == 'qwen' else 1, 'chunks': []}
    with sf.SoundFile(out_wav, 'w', samplerate=SR, channels=2, subtype='PCM_16') as master:
        for i, segment in enumerate(script['segments'], 1):
            pieces = (split_narration(segment['text'], max_chars=max_chars)
                      if provider_name == 'qwen' else [segment['text']])
            for j, piece in enumerate(pieces, 1):
                key = json.dumps({'model': tts.identity, 'speaker': segment['speaker'],
                                  'text': piece,
                                  'takes': takes if provider_name == 'qwen' else 1},
                                 sort_keys=True, ensure_ascii=False)
                path = cache / (hashlib.sha256(key.encode()).hexdigest() + '.wav')
                selection = None
                if path.exists():
                    report['cached_segments'] += 1
                else:
                    if provider_name == 'qwen':
                        selection = tts.synthesize(segment['speaker'], piece, path, takes=takes)
                    else:
                        tts.synthesize(segment['speaker'], piece, path)
                    report['new_segments'] += 1
                samples, sr = sf.read(path, dtype='float32', always_2d=True)
                check = describe_audio(samples, sr, len(piece.split())) if provider_name == 'qwen' else None
                if check is not None and not check['valid']:
                    path.unlink(missing_ok=True)
                    raise RuntimeError(f'Fallo de calidad en bloque {i}.{j}: {check}')
                master.write(soften_edges(stereo(resample(samples, sr)), SR))
                pause_ms = segment.get('pause_ms', 500) if j == len(pieces) else 160
                if pause_ms:
                    master.write(np.zeros((int(SR * pause_ms / 1000), 2), dtype=np.float32))
                report['chunks'].append({'block': i, 'part': j, 'speaker': segment['speaker'],
                                         'words': len(piece.split()), 'technical_qa': check,
                                         'selection': selection})
                print(f'[{i}/{len(script["segments"])} | {j}/{len(pieces)}] {segment["speaker"]}', flush=True)
    report['duration_minutes'] = round(sf.info(out_wav).duration / 60, 3)
    report['meets_duration'] = 35 <= report['duration_minutes'] <= 45
    if provider_name == 'qwen' and not allow_short and not report['meets_duration']:
        raise ValueError('El episodio debe durar de 35 a 45 minutos; revisa el guion')
    if shutil.which('ffmpeg'):
        out_mp3 = output_dir / (script['id'] + '.mp3')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(out_wav), '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5',
                        '-codec:a', 'libmp3lame', '-b:a', '192k', str(out_mp3)], check=True)
        report['mp3'] = str(out_mp3)
    report['wav'] = str(out_wav)
    (output_dir / (script['id'] + '.report.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
