"""Pruebas para la configuración del proveedor de preview neuronal."""
import json
import sys
import types
import wave
from pathlib import Path
from unittest.mock import Mock
import pytest

def test_missing_piper_voices_gives_instructions(tmp_path, monkeypatch):
    fake = types.ModuleType("piper")
    fake.PiperVoice = Mock()
    fake_config = types.ModuleType("piper.config")
    fake_config.SynthesisConfig = Mock
    monkeypatch.setitem(sys.modules, "piper", fake)
    monkeypatch.setitem(sys.modules, "piper.config", fake_config)
    from la_otra_historia.piper_preview import PiperPreviewTTS
    with pytest.raises(FileNotFoundError, match="download_voices"):
        PiperPreviewTTS(tmp_path)

def test_piper_speaker_ids_and_wav(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    import hashlib
    fake = types.ModuleType("piper")
    voice = Mock()
    def speak(text, writer, syn_config=None):
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes((np.zeros(2205,dtype=np.int16)).tobytes())
    voice.synthesize_wav.side_effect = speak
    fake.PiperVoice = Mock()
    fake.PiperVoice.load.return_value = voice
    fake_config = types.ModuleType("piper.config")
    class SynthesisConfig:
        def __init__(self,**kw): self.kw=kw
    fake_config.SynthesisConfig=SynthesisConfig
    monkeypatch.setitem(sys.modules,"piper",fake)
    monkeypatch.setitem(sys.modules,"piper.config",fake_config)
    d=tmp_path
    (d/"es_ES-sharvard-medium.onnx").write_bytes(b"test")
    (d/"es_ES-sharvard-medium.onnx.json").write_text(json.dumps({"num_speakers":2,"speaker_id_map":{"M":0,"F":1}}))
    from la_otra_historia.piper_preview import PiperPreviewTTS
    tts=PiperPreviewTTS(d)
    tts.synthesize("juan","Hola, soy Papá Noel.",d/"juan.wav")
    tts.synthesize("carla","Hola, soy Papá Noel.",d/"carla.wav")
    assert sf.info(d/"juan.wav").samplerate==22050
    assert sf.info(d/"carla.wav").duration>.05
    assert voice.synthesize_wav.call_args.kwargs["syn_config"].kw["speaker_id"]==1

def test_piper_refuses_wrong_mapping(tmp_path, monkeypatch):
    fake=types.ModuleType("piper")
    fake.PiperVoice=Mock()
    fc=types.ModuleType("piper.config")
    fc.SynthesisConfig=Mock
    monkeypatch.setitem(sys.modules,"piper",fake)
    monkeypatch.setitem(sys.modules,"piper.config",fc)
    (tmp_path/"es_ES-sharvard-medium.onnx").write_bytes(b"test")
    (tmp_path/"es_ES-sharvard-medium.onnx.json").write_text(json.dumps({"num_speakers":2,"speaker_id_map":{"M":1,"F":0}}))
    from la_otra_historia.piper_preview import PiperPreviewTTS
    with pytest.raises(ValueError,match="M=0"):
        PiperPreviewTTS(tmp_path)
