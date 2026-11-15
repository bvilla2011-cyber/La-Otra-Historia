# Calidad de voces: aprobación editorial y técnica

## Por qué cambiar de una voz robótica a una voz creíble exige más que filtros
Ni ecualización, ni cambiar el tono de eSpeak, ni agregar respiraciones falsas produce prosodia humana.
Para Juan y Carla la opción principal es **Qwen3-TTS VoiceDesign + Base**. El método Voice Design luego Clone está documentado oficialmente en https://github.com/QwenLM/Qwen3-TTS .

## Cambios v0.4
1. **Audiciones A/B/C originales** por locutor; se generan en una GPU y no se almacenan en Git.
2. **Aprobación separada**: elegir un timbre después de oírlo y registrar la elección en `voices/juan.json` / `voices/carla.json`. El estado provisional no autoriza temporadas.
3. **Múltiples tomas por fragmento** (por defecto dos, configurable 1–5). Variar semillas y parámetros de sampling puede ayudar; la mejora no está garantizada.
4. **Descartar audio roto**: clips vacíos, recortados, con silencio excesivo o duración aberrante. Estas mediciones NO miden dicción, emociones ni naturalidad percibida.
5. **Toma elegida, caché por modelo y referencia**, frases máximas de 260 caracteres, pausas de 160ms entre fragmentos y fade de 12ms para prevenir chasquidos.
6. **Informes JSON** con cada toma, duración y medidas; master estéreo 48 kHz y MP3 a 192 kb/s con normalización de volumen.

## Cómo probar audiciones neuronales reales
Necesitas Linux, Python 3.11, FFmpeg y **una GPU CUDA** capaz de alojar el modelo 1.7B. También pueden usarse servicios externos de GPU; NO existe tal GPU en los runners habituales gratuitos de GitHub.
```bash
python -m pip install -e '.[voice,dev]'
loh design-voices --dest voices --candidates 3
# Escuchar voices/auditions/juan/take_1.wav ... take_3.wav y Carla igual
loh approve-voice juan 2 --voices voices
loh approve-voice carla 1 --voices voices
loh render stories/voice_test_papa_noel.json --provider qwen --voices voices --takes 3 --allow-short
loh render stories/voice_test_narracion.json --provider qwen --voices voices --takes 3 --allow-short
```

En GitHub Actions, el flujo manual `Estudio de voces Qwen3 (GPU propia)` requiere un runner con etiquetas `self-hosted, linux, x64, gpu`; sus selecciones son de **preview**, no aprobación. Los archivos WAV y MP3 se entregan como artefacto de la ejecución. No lanzar este workflow en máquinas no confiables ni en PRs ajenas.

## Umbral para empezar episodios de 35–45 min
Hacer escuchar 3–5 minutos por narrador a varias personas, corregir pronunciación de números y nombres, comprobar que ambas voces se reconocen durante 20+ min, confirmar consistencia y expresividad, y solo entonces aprobar las referencias. Nada del código garantiza equivalencia con ElevenLabs; la comparación debe ser por escucha ciega.
