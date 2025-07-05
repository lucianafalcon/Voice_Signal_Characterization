#-----------------------------------------------------
#        Algoritmo para grabación de voz 
#-----------------------------------------------------

import sounddevice as sd
import numpy as np
import keyboard
from scipy.io.wavfile import write

frecuencia_muestreo = 44100  # Frecuencia de muestreo (Hz)
canales = 2  # Estéreo
dtype = np.int16  # Tipo de dato

print("Presiona 'Enter' para iniciar la grabación...")
keyboard.wait("enter")

print("Grabando... Presiona 'Enter' para detener.")

audio = []  # Lista para almacenar los fragmentos de audio grabados

def callback(indata, frames, time, status):
    """Función de callback que se ejecuta en cada bloque de audio grabado."""
    if status:
        print(status)
    audio.append(indata.copy())

# Iniciar grabación en modo streaming
with sd.InputStream(samplerate=frecuencia_muestreo, channels=canales, dtype=dtype, callback=callback):
    keyboard.wait("enter")  # Espera hasta que presionemos Enter para detener
    print("Grabación detenida.")

# Convertir lista en un array de NumPy
audio_np = np.concatenate(audio, axis=0)

# Guardar el audio en un archivo WAV
nombre_archivo = "grabacion_manual.wav"
write(nombre_archivo, frecuencia_muestreo, audio_np)
print(f"Audio guardado como {nombre_archivo}")






























