# vision-addons

Canal **publico** de add-ons del companion de Vision IA. Esta vacio a proposito: solo
guarda los **assets de los Releases** (los add-ons ya empaquetados), que el companion
descarga bajo demanda con verificacion SHA-256.

- Canal de descarga gratuito (CDN de GitHub) para no cargar el servidor del plugin.
- Los pesos de los modelos (p. ej. 2,3 GB de Chatterbox) NO se sirven aqui: se descargan
  del modelo publico de HuggingFace la primera vez y quedan en el PC del usuario.

## Releases

- oice-clon-chatterbox-1.0 - add-on "Clonar tu voz" (Chatterbox, MIT) para el companion.

## Licencias

Los binarios redistribuidos conservan sus licencias originales: Chatterbox TTS (MIT),
PyTorch (BSD-3), y el resto de dependencias tal como se indican en el propio add-on.