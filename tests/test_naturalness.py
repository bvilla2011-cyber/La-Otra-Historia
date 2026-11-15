import numpy as np
import pytest
from la_otra_historia.naturalness import describe_audio, soften_edges, split_narration

def test_split_short_untouched():
    assert split_narration('Hola, soy Papá Noel.') == ['Hola, soy Papá Noel.']

def test_split_long_unchanged_words():
    text = 'Clara encontró la primera pista. ' * 35
    parts = split_narration(text, 150)
    assert len(parts) > 2
    assert all(len(p) <= 150 for p in parts)
    assert ' '.join(parts) == text.strip()

def test_clipping_and_silence():
    sr = 24000
    x = .12 * np.sin(2 * np.pi * 160 * np.arange(sr * 2) / sr)
    assert describe_audio(x, sr, 6)['valid']
    assert not describe_audio(np.ones(sr, dtype=np.float32), sr, 5)['valid']

def test_soft_edges():
    x = np.ones(1000, dtype=np.float32) * .2
    y = soften_edges(x, 24000)
    assert y[0] == 0 and y[-1] == 0
    assert y[500] == pytest.approx(.2)
