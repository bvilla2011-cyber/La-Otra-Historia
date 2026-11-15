# LA OTRA HISTORIA — motor IA de podcast

Antología de ficción original, narrada en tercera persona por Juan (voz masculina) y Carla (voz femenina).

**Estado: prototipo 0.5; aún NO se han probado audiciones Qwen reales en GPU.** La prueba «Hola, soy Papá Noel» está disponible con dos voces distintas mediante `espeak-preview`. Ese proveedor es ROBÓTICO: sirve para probar la tubería, **no** tiene calidad ElevenLabs. El destino de producción es Qwen3-TTS, pendiente de GPU, voces de referencia y revisión de tomas.

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

## v0.3 · Prueba neuronal adicional en CPU

Hay un nuevo proveedor `piper-preview` de dos voces neuronales españolas (modelo Sharvard: M=Juan; F=Carla), que permite comparar un salto respecto al preview robótico. No equivale a ElevenLabs y las muestras todavía deben evaluarse por oído. El modelo se descarga en GitHub Actions, no se incorpora al repositorio.

- [Generar y descargar las audiciones en GitHub Actions](https://github.com/bvilla2011-cyber/La-Otra-Historia/actions/workflows/voice-neural-cpu.yml).
- [Guía y licencias](docs/objetivo_voces_neuronales.md).

## v0.4 — Motor de interpretación y audiciones

Se han añadido audiciones A/B/C de **voces originales** de Juan y Carla, selección humana de referencia vocal, tomas múltiples de cada fragmento, descartes de errores técnicos, separación de frases largas sin cortar palabras, fundidos muy cortos, informe por frase y validación de que las voces estén aprobadas antes de producir una temporada. La validación técnica NO demuestra realismo humano.

- Demo CPU: seguir usando el proveedor `piper-preview` como prueba técnica.
- Voz avanzada: ejecutar `loh design-voices --dest voices --candidates 3` con GPU CUDA, escuchar todas las muestras y elegir cada voz con `loh approve-voice juan 2` / `loh approve-voice carla 1` (ejemplo).
- Grabar: `loh render stories/voice_test_narracion.json --provider qwen --voices voices --takes 3 --allow-short`.
- En Actions se ha preparado un [workflow experimental con GPU propia](https://github.com/bvilla2011-cyber/La-Otra-Historia/actions/workflows/voice-studio-gpu.yml). No puede ejecutarse en el runner gratuito normal de GitHub y no se ha verificado aún en una GPU.

Lee [Calidad y aprobaciones](docs/calidad_voces.md) antes de publicar.

## v0.5 — Auditoría de inferencia Qwen y GPU
Se verificó la API upstream VoiceDesign/VoiceClone, se añadió carga FP16/BF16 compatible y diagnóstico previo a descargas. La calidad vocal **no** está demostrada por tests unitarios.
```bash
loh doctor
loh doctor --strict  # devuelve error si no hay CUDA + dependencias
```
Para una máquina GPU, revisa [Auditoría y guía de ejecución](docs/auditoria_gpu.md) o `bash scripts/gpu_quickstart.sh`; empieza por **una** muestra por narrador. Evita ejecutar voces Qwen en Actions salvo que hayas conectado un runner GPU propio.
