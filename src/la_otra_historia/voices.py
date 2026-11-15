"""Diseñar identidades vocales sintéticas originales con Qwen3-TTS."""
import json
from pathlib import Path

VOICE_PROFILES = {
    "juan": {
        "ref_text": "En aquella pequeña ciudad nadie recordaba cuándo se cerró la estación. Hasta que una noche, un tren volvió a detenerse en el andén.",
        "description": "Voz masculina adulta de España, español peninsular nativo, barítono cálido, articulación clara, natural, relajada, narrador literario íntimo y sobrio. Sin teatralidad excesiva."
    },
    "carla": {
        "ref_text": "Aquella fotografía llevaba treinta años guardada. Cuando Julia la encontró, descubrió que había alguien al fondo que no debía estar allí.",
        "description": "Voz femenina adulta de España, español peninsular nativo, tono medio cálido, respiración natural, articulación clara, narrativa sensible, cercanía y sutileza."
    }
}

def design_voices(dest, model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
    try:
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise RuntimeError("Instala pip install -e '.[voice]'") from e
    if not torch.cuda.is_available():
        raise RuntimeError("Para el diseño de voces se necesita GPU CUDA")
    output = Path(dest)
    output.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    for speaker, profile in VOICE_PROFILES.items():
        path = output / f"{speaker}.wav"
        if path.exists():
            raise FileExistsError(f"Esta referencia ya existe: {path}")
        wavs, sr = model.generate_voice_design(
            text=profile["ref_text"], language="Spanish", instruct=profile["description"])
        sf.write(path, wavs[0], sr)
        (output / f"{speaker}.json").write_text(
            json.dumps({"ref_audio": f"{speaker}.wav", "ref_text": profile["ref_text"],
                        "license": "voz sintética original", "language": "Spanish"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
