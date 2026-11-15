# Auditoría y guía GPU — 15 de noviembre de 2026

## Comprobaciones realizadas
- API oficial Qwen3-TTS: el modelo `VoiceDesign` expone `generate_voice_design(text, language, instruct)`; el modelo `Base` expone `create_voice_clone_prompt(ref_audio, ref_text)` y `generate_voice_clone(text, language, voice_clone_prompt)`.
- Fuentes: https://github.com/QwenLM/Qwen3-TTS y https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
- El entorno de auditoría disponible usa PyTorch solo CPU y no tiene `qwen-tts` ni CUDA. **No se sintetizaron audiciones neuronales de Juan o Carla aquí.**
- Se ejecutaron pruebas locales CPU y simulaciones; estas pruebas **no validan la calidad perceptiva** de audio neuronal real.

## Cambios v0.5
1. `loh doctor` inspecciona PyTorch, CUDA, nombre y VRAM GPU, soporte BF16 y herramientas sin descargar grandes modelos. `--strict` falla antes de la inferencia si el entorno no está listo.
2. Carga BF16 si compatible, si no FP16, con atención SDPA; evita asumir GPUs nuevas.
3. El trabajo GPU comprueba dependencias antes de diseñar voces y empieza por **una** candidata por narrador, para minimizar coste inicial.
4. Al terminar VoiceDesign, liberar memoria GPU antes de cargar el modelo Base.
5. No subir checkpoints ni tokens a GitHub; mantener audios de referencia fuera del árbol versionado. Las voces deben ser originales, sin suplantar terceros.

## Ejecutar en un servidor con GPU
GPU NVIDIA/CUDA, Linux, Python 3.11+ (el proveedor recomienda 3.12), FFmpeg, conexión a Internet y espacio para dos checkpoints de Qwen (VoiceDesign + Base).
Se recomienda partir de **16 GiB de VRAM** como margen de planificación; no es una especificación garantizada de memoria.
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[voice,dev]'
loh doctor --strict
loh validate stories/voice_test_papa_noel.json
loh design-voices --dest voices --candidates 1
# Escuchar de verdad voices/auditions/juan/take_1.wav y carla/take_1.wav
loh approve-voice juan 1 --voices voices --preview
loh approve-voice carla 1 --voices voices --preview
loh render stories/voice_test_papa_noel.json --provider qwen --voices voices --takes 1 --allow-short --output output/neural
loh render stories/voice_test_narracion.json --provider qwen --voices voices --takes 1 --allow-short --output output/neural
```
**IMPORTANTE:** el paso `--preview` no aprueba voces para la temporada. Para fijarlas, escuchar, seleccionar manualmente y volver a ejecutar `loh approve-voice` SIN `--preview`.

Se puede ejecutar desde una GPU alquilada con una terminal; GitHub Actions con etiquetas `[self-hosted,linux,x64,gpu]` solo funciona si has conectado un runner propio con esas etiquetas. Un repo público puede exponer las audiciones subidas como artefactos; considera hacer el repositorio privado cuando haya voces definitivas.

## Qué aún no está verificado
Carga REAL de los modelos con CUDA, tiempos de generación, artefactos de audio, acento y pronunciación, continuidad de locutores y equivalencia de calidad con ElevenLabs. Todo eso requiere una sesión efectiva de GPU y escucha humana. No ejecutar una temporada de 35–45 minutos hasta aprobar un piloto real.
