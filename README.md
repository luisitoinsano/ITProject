# ITProject

Este repositorio contiene una demo de detección de gestos y sobreposición de "memes" en tiempo real usando MediaPipe y OpenCV.

Instrucciones de uso (Windows / PowerShell):

1. Clona el repo:

   git clone https://github.com/luisitoinsano/ITProject.git
   cd ITProject

2. Crea un entorno virtual (recomendado):

   python -m venv .venv

3. Activa el entorno (PowerShell):

   & .\\.venv\\Scripts\\Activate.ps1

   o si prefieres usar el venv que ya está en `detector_emociones/venv`, activa esa ruta:

   & .\\detector_emociones\\venv\\Scripts\\Activate.ps1

4. Instala dependencias desde `requirements.txt`:

   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

5. Ejecuta la aplicación (abre la cámara):

   python realtime_fer.py

Notas importantes:
- No se incluyen entornos virtuales en el repositorio (se han eliminado del seguimiento). Usa `requirements.txt` para reproducir el entorno.
- Si tienes problemas con `mediapipe` por la versión de Python, crea un venv con Python 3.10/3.11.
- La carpeta `memes/` contiene imágenes que se cargan dinámicamente; añade tus memes ahí.