"""Pruebas CPU para selección de múltiples tomas Qwen y renders neuronales simulados."""
import json
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import pytest
from la_otra_historia.tts import QwenTTS

def test_qwen_multitake_rejects_clipping(tmp_path):
    fake = QwenTTS.__new__(QwenTTS)
    fake.identity = 'test-qwen'
    fake.configs = {'juan': ({'ref_text': 'Texto de referencia'}, tmp_path / 'juan.wav')}
    fake.prompts = {'juan': ['fake-prompt']}
    fake._torch = SimpleNamespace(manual_seed=lambda _: None)
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        sr = 24000
        if len(calls) == 1:
            audio = np.ones(sr * 2, dtype=np.float32)
        else:
            t = np.arange(sr * 2) / sr
            audio = (.12 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        return [audio], sr
    fake.model = SimpleNamespace(generate_voice_clone=generate)
    path = tmp_path / 'out.wav'
    report = fake.synthesize('juan', 'Hola, soy Papá Noel.', path, takes=2)
    assert path.exists()
    assert report['selected_take'] == 2
    assert report['attempts'][0]['valid'] is False
    assert report['attempts'][1]['valid'] is True
    assert all(x['language'] == 'Spanish' for x in calls)

def test_render_qwen_chunking_and_preview_approval(tmp_path, monkeypatch):
    from la_otra_historia import render as mod
    class FakeQwen:
        identity = 'test-qwen-qa'
        def __init__(self, voices_dir):
            self.configs = {'juan': ({'approved': False}, None), 'carla': ({'approved': False}, None)}
        def synthesize(self, speaker, text, path, takes=2):
            sr = 24000
            dur = len(text.split()) * 60 / 145
            t = np.arange(int(sr * dur), dtype=np.float32) / sr
            sf.write(path, (.12 * np.sin(2 * np.pi * 210 * t)).astype(np.float32), sr)
            return {'selected_take': 1, 'attempts': []}
    monkeypatch.setattr(mod, 'QwenTTS', FakeQwen)
    story = {'id': 'test', 'title': 'Narración', 'segments': [
        {'speaker': 'juan', 'text': 'Clara abrió la puerta. ' * 14},
        {'speaker': 'carla', 'text': 'Tomás nunca volvió a mirar atrás. ' * 14},
    ]}
    src = tmp_path / 'story.json'
    src.write_text(json.dumps(story))
    with pytest.raises(ValueError, match='aprueba'):
        mod.render(src, tmp_path / 'no-approval', tmp_path, tmp_path, 'qwen', allow_short=False)
    out = mod.render(src, tmp_path / 'approved-preview', tmp_path, tmp_path,
                     'qwen', allow_short=True, takes=2, max_chars=100)
    assert out['new_segments'] > 2
    assert all(x['technical_qa']['valid'] for x in out['chunks'])
    assert sf.info(out['wav']).samplerate == 48000
