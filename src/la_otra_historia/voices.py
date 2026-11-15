"""Audiciones de voces sintéticas originales y selección manual para Juan y Carla."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from .gpu import qwen_load_kwargs

VOICE_PROFILES = {
    'juan': {
        'ref_text': 'Aquella noche, Tomás encontró una llave sobre la mesa. Nadie sabía cómo había llegado allí, pero todos parecían reconocerla.',
        'description': 'Natural adult male from Spain speaking native European Spanish, warm baritone, intimate storytelling, easy conversational breath, precise articulation, varied melodic intonation, gentle endings. A subtle cinema documentary narrator, human not announcer, never exaggerated or robotic.',
    },
    'carla': {
        'ref_text': 'Clara abrió el sobre sin hacer ruido. Dentro había una fotografía que llevaba treinta años esperando a que alguien la encontrase.',
        'description': 'Natural adult female from Spain speaking native European Spanish, warm medium register, intimate and emotionally subtle narration, lively natural cadence, gentle pauses and varied intonation. Storyteller, not a radio announcer, never exaggerated or robotic.',
    },
}

def audition_voices(dest, candidates=3, model_id='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'):
    """Produce A/B/C, sin aprobar voces automáticamente."""
    if not 1 <= candidates <= 5:
        raise ValueError('candidates debe ser 1 a 5')
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError("Instala pip install -e '.[voice]'") from exc
    if not torch.cuda.is_available():
        raise RuntimeError('Para diseñar voces hace falta GPU CUDA')
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(model_id, **qwen_load_kwargs(torch))
    inventory = []
    for speaker, profile in VOICE_PROFILES.items():
        folder = root / 'auditions' / speaker
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, candidates + 1):
            out = folder / f'take_{i}.wav'
            if out.exists():
                raise FileExistsError(f'Ya existe {out}')
            torch.manual_seed((1191 if speaker == 'juan' else 2391) + i * 37)
            wavs, sr = model.generate_voice_design(
                text=profile['ref_text'], language='Spanish',
                instruct=profile['description'], do_sample=True,
                temperature=(.82, .90, .86, .94, .78)[i - 1])
            sf.write(out, wavs[0], sr)
            inventory.append({'speaker': speaker, 'take': i, 'audio': str(out),
                              'reference_text': profile['ref_text'], 'approved': False})
    # Evitar mantener VoiceDesign en VRAM al cargar Base.
    del model
    torch.cuda.empty_cache()
    (root / 'auditions' / 'manifest.json').write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    return inventory

def approve_voice(voices_dir, speaker, take, preview=False):
    """Fija una voz tras escucharla; con preview=True no es final."""
    if speaker not in VOICE_PROFILES:
        raise ValueError('Locutor debe ser juan o carla')
    if not 1 <= take <= 5:
        raise ValueError('Toma inválida')
    root = Path(voices_dir).resolve()
    candidate = root / 'auditions' / speaker / f'take_{take}.wav'
    if not candidate.is_file():
        raise FileNotFoundError(f'No existe la toma {candidate}')
    import soundfile as sf
    info = sf.info(candidate)
    if not 3 <= info.duration <= 35:
        raise ValueError('La referencia debe tener entre 3 y 35 segundos')
    selected = root / f'{speaker}.wav'
    shutil.copyfile(candidate, selected)
    config = {
        'ref_audio': f'{speaker}.wav',
        'ref_text': VOICE_PROFILES[speaker]['ref_text'],
        'language': 'Spanish', 'source': 'Qwen3-TTS VoiceDesign synthetic original',
        'approved': not preview, 'selected_take': take,
        'approval_note': 'Aprobación requiere escucha humana previa',
    }
    (root / f'{speaker}.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return selected

def design_voices(dest, model_id='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'):
    """Compatibilidad: genera audiciones, no voces automáticamente aprobadas."""
    return audition_voices(dest, candidates=3, model_id=model_id)
