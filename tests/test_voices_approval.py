from pathlib import Path
import json
import numpy as np
import soundfile as sf
import pytest
from la_otra_historia.voices import approve_voice

def test_approve_requires_take(tmp_path):
    with pytest.raises(FileNotFoundError):
        approve_voice(tmp_path, 'juan', 2)

def test_approve_writes_reference_and_config(tmp_path):
    folder = tmp_path / 'auditions' / 'carla'
    folder.mkdir(parents=True)
    signal = .03 * np.sin(np.arange(24000 * 4) / 30)
    sf.write(folder / 'take_1.wav', signal, 24000)
    selected = approve_voice(tmp_path, 'carla', 1)
    assert selected.is_file()
    conf = json.loads((tmp_path / 'carla.json').read_text())
    assert conf['approved'] is True and conf['ref_audio'] == 'carla.wav'
    assert 'Clara abrió' in conf['ref_text']

def test_preview_not_approved(tmp_path):
    folder = tmp_path / 'auditions' / 'juan'
    folder.mkdir(parents=True)
    sf.write(folder / 'take_1.wav', .03 * np.sin(np.arange(24000 * 5) / 30), 24000)
    approve_voice(tmp_path, 'juan', 1, preview=True)
    conf = json.loads((tmp_path / 'juan.json').read_text())
    assert conf['approved'] is False

def test_invalid_take(tmp_path):
    with pytest.raises(ValueError):
        approve_voice(tmp_path, 'juan', 0)
