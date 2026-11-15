from pathlib import Path
import json
import shutil
import pytest
import soundfile as sf
from la_otra_historia.schema import validate_script, stats
from la_otra_historia.render import render

def test_valid_script():
    script={"id":"test","title":"Prueba","segments":[{"speaker":"juan","text":"Una historia que comienza."},{"speaker":"carla","text":"Y que termina."}]}
    validate_script(script)
    assert stats(script)["words"] == 7

def test_invalid_speaker():
    with pytest.raises(ValueError, match="speaker"):
        validate_script({"id":"test","title":"Prueba","segments":[{"speaker":"otro","text":"Hola."}]})

def test_mock_render_and_cache(tmp_path):
    script={"id":"test","title":"Prueba","segments":[{"speaker":"juan","text":"Hola, esta es una prueba."},{"speaker":"carla","text":"Y esta es otra voz."}]}
    path=tmp_path/"test.json"
    path.write_text(json.dumps(script),encoding="utf-8")
    a=render(path,tmp_path/"out",tmp_path/"voices",tmp_path/"assets","mock")
    assert a["new_segments"]==2 and Path(a["wav"]).exists()
    assert sf.info(a["wav"]).samplerate==48000
    b=render(path,tmp_path/"out",tmp_path/"voices",tmp_path/"assets","mock")
    assert b["cached_segments"]==2

def test_espeak_produces_speech(tmp_path):
    if not shutil.which("espeak"): pytest.skip("espeak no instalado")
    story=Path(__file__).resolve().parents[1]/"stories"/"voice_test_papa_noel.json"
    r=render(story,tmp_path,tmp_path/"voices",tmp_path/"assets","espeak-preview",True)
    assert r["new_segments"]==2
    assert sf.info(r["wav"]).channels==2
    assert r["quality"]=="robotic-technical-preview"
