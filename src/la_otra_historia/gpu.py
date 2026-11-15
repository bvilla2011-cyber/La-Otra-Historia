"""Diagnóstico CUDA y carga segura de modelos Qwen3-TTS: no descarga pesos."""
from __future__ import annotations
import importlib.util
import platform
import shutil

RECOMMENDED_VRAM_GB = 16

def qwen_load_kwargs(torch_module):
    """Usar BF16 solo cuando CUDA lo soporta; FP16 si no."""
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError('No hay GPU CUDA: necesitas GPU local o un servidor externo para Qwen3-TTS')
    supports = getattr(cuda, 'is_bf16_supported', None)
    bf16 = callable(supports) and supports()
    return {
        'device_map': 'cuda:0',
        'dtype': torch_module.bfloat16 if bf16 else torch_module.float16,
        'attn_implementation': 'sdpa',
    }

def doctor():
    """Verificación OFFLINE; no presupone que los checkpoints estén descargados."""
    report = {
        'python': platform.python_version(),
        'ffmpeg': bool(shutil.which('ffmpeg')),
        'qwen_tts_installed': importlib.util.find_spec('qwen_tts') is not None,
        'torch_installed': importlib.util.find_spec('torch') is not None,
        'cuda_available': False,
        'gpu': None,
        'model_ready': False,
        'recommended_vram_gb': RECOMMENDED_VRAM_GB,
        'issues': [],
    }
    if not report['ffmpeg']:
        report['issues'].append('Falta FFmpeg: exportación MP3 no disponible')
    if not report['qwen_tts_installed']:
        report['issues'].append("Falta qwen-tts: pip install -e '.[voice]'")
    if not report['torch_installed']:
        report['issues'].append('Falta PyTorch')
        return report
    try:
        import torch
        report['torch_version'] = torch.__version__
        if not torch.cuda.is_available():
            report['issues'].append('CUDA no disponible: no se puede sintetizar audio Qwen3-TTS aquí')
            return report
        report['cuda_available'] = True
        free, total = torch.cuda.mem_get_info(0)
        bf16 = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        report['gpu'] = {
            'name': torch.cuda.get_device_name(0),
            'total_vram_gb': round(total / 2**30, 2),
            'available_vram_gb': round(free / 2**30, 2),
            'supports_bf16': bf16,
            'precision': 'bf16' if bf16 else 'fp16',
        }
        if free < 10 * 2**30:
            report['issues'].append('Memoria GPU libre inferior a 10 GiB; usar al menos 16 GiB como margen prudente, no garantía')
        # Comprobar dependencias y disponibilidad de CUDA NO garantiza poder generar voz.
        report['model_ready'] = report['ffmpeg'] and report['qwen_tts_installed'] and free >= 10 * 2**30
    except Exception as exc:
        report['issues'].append(f'Error inspeccionando CUDA: {type(exc).__name__}: {exc}')
    return report
