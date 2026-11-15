"""Pruebas sin GPU para diagnóstico y selección de precisión."""
from types import SimpleNamespace
import pytest
from la_otra_historia.gpu import doctor, qwen_load_kwargs

def test_precision_selection():
    fake = SimpleNamespace(bfloat16='bf16', float16='fp16',
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True))
    assert qwen_load_kwargs(fake)['dtype'] == 'bf16'
    fake.cuda.is_bf16_supported = lambda: False
    assert qwen_load_kwargs(fake)['dtype'] == 'fp16'

def test_no_cuda_fails():
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match='CUDA'):
        qwen_load_kwargs(fake)

def test_doctor_is_offline_and_structured():
    data = doctor()
    assert isinstance(data['issues'], list)
    assert isinstance(data['cuda_available'], bool)
    assert 'recommended_vram_gb' in data
