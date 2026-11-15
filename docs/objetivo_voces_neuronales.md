# Próximo objetivo: voces neuronales Juan y Carla

## Previsualización CPU (Piper)
El modelo [es_ES-sharvard-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/sharvard/medium) es **español de España** y contiene dos voces: M (speaker 0) y F (speaker 1). El modelo exige Piper 1.5 o superior. La fuente declara dataset CC-BY-3.0; atribuye autores y comprueba licencias de modelos y bases antes de publicación comercial. El paquete Piper es GPL-3.0; revisar compatibilidad con distribución de software.

El objetivo de Piper es una **prueba neuronal en CPU**, no sustituir la calidad del motor definitivo.

```bash
sudo apt install -y ffmpeg espeak-ng
pip install -e '.[piper,dev]'
python -m piper.download_voices --data-dir voices es_ES-sharvard-medium
loh render stories/voice_test_papa_noel.json --provider piper-preview --voices voices --allow-short
loh render stories/voice_test_narracion.json --provider piper-preview --voices voices --allow-short
```

En GitHub, abrir **Actions → Demo neuronal CPU Juan y Carla (Piper)** para lanzar una ejecución manual, o revisar la ejecución del push. Descargar el artefacto `LA-OTRA-HISTORIA-PIPER-v0-3`. No se versionan modelos ni WAV en el repositorio.

## Destino final Qwen3-TTS con GPU
El flujo `loh design-voices --dest voices` crea voces sintéticas originales. Se deben revisar y aprobar las referencias antes de `loh render ... --provider qwen`, que fija los timbres mediante prompts de clonación. No usamos voces de personas reales sin permiso. La calidad comparada con ElevenLabs sigue **sin comprobar**; requiere evaluación auditiva de al menos 3–5 minutos por voz y pruebas de continuidad.
