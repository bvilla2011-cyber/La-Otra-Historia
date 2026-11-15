# LA OTRA HISTORIA — motor IA de podcast

Antología de ficción original, narrada en tercera persona por Juan (voz masculina) y Carla (voz femenina).

**Estado: prototipo 0.2.** La prueba «Hola, soy Papá Noel» está disponible con dos voces distintas mediante `espeak-preview`. Ese proveedor es ROBÓTICO: sirve para probar la tubería, **no** tiene calidad ElevenLabs. El destino de producción es Qwen3-TTS, pendiente de GPU, voces de referencia y revisión de tomas.

## Instalar en Linux (Python 3.11+)
```bash
sudo apt update && sudo apt install -y ffmpeg espeak
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
loh validate stories/voice_test_papa_noel.json
loh render stories/voice_test_papa_noel.json --provider espeak-preview --allow-short
pytest -q
```
El MP3 y WAV quedarán en `output/`. También se pueden obtener desde los artefactos de GitHub Actions.

## Modelo neuronal con GPU
```bash
pip install -e '.[voice]'
loh design-voices --dest voices
# Escuchar, ajustar y aprobar ambas referencias sintéticas antes de publicar.
loh render stories/voice_test_papa_noel.json --provider qwen --voices voices --allow-short
```
Los pesos se descargan de la fuente original y no deben subirse a GitHub. Los guiones de los doce episodios de 35–45 minutos están en fase de tratamiento, no de grabación.

## Calidad y licencia
No clonar voces de terceros sin consentimiento. Usar música y efectos propios o licenciados. QA: pronunciación, consistencia de identidad, expresividad, artefactos, mezcla y escucha humana. GitHub no ofrece GPU permanente; la inferencia neuronal necesitará GPU remota o local compatible.
