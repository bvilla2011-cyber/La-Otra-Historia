# Arquitectura y ruta de calidad

## Pipeline
Guion JSON validado → bloques por speaker → TTS intercambiable → WAV con caché por modelo/voz/texto → silencios → master estéreo 48 kHz → loudness -16 LUFS, techo -1.5 dBTP con ffmpeg → MP3 192 kbps y WAV.

## Tres proveedores
- `mock`: tonos de prueba, sin habla.
- `espeak-preview`: habla inteligible pero **robótica** en CPU. Para comprobar alternancia de Juan y Carla, no para escuchar 40 minutos.
- `qwen`: síntesis neuronal mediante Qwen3-TTS Base y muestras sintéticas fijadas; requiere CUDA, modelo descargado y pruebas auditivas.

## Criterios antes de grabar la temporada
1. Pronunciación natural de español de España.
2. Consistencia de ambos timbres en bloques largos.
3. Expresividad sin sobreactuación y pausas orgánicas.
4. Ausencia de artefactos, repeticiones y cortes.
5. Flujo editorial, pronunciación de nombres y números.
6. Master accesible, sin música que tape narración.
7. Control de derechos y revisión humana.
8. Escucha ciega comparada con voz de referencia profesional.

## Infraestructura
GitHub = fuente, versionado y pruebas CPU. GPU para voz neuronal = máquina propia o servicio externo; nunca poner tokens en el repositorio. El audio final se distribuye a través del alojamiento elegido, no mediante GitHub Pages como backend TTS. No hay garantía de equivalencia ElevenLabs sin evaluación real.
