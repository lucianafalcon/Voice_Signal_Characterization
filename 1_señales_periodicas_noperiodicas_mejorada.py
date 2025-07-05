#-----------------------------------------------------
#        Script para graficar las señales 
#                  en el tiempo
#-----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# Cargar archivo de audio
nombre_archivo = "grabacion_lenta.wav"  
frecuencia_muestreo, audio = wavfile.read(nombre_archivo)

# Convertir a mono si es estéreo (xqe alcanza, menos datos y menos laburo)
if len(audio.shape) > 1:
    audio = audio[:, 0]

# Crear el eje de tiempo
duracion = len(audio) / frecuencia_muestreo
t = np.linspace(0, duracion, len(audio))

# Graficar 
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, audio, color="b")
plt.xlabel("Tiempo (s)")

plt.ylabel("Amplitud")
plt.title("Señal /o/ Lenta en el Tiempo")
plt.grid()

plt.tight_layout()
plt.show()
