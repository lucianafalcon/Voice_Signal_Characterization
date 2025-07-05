#-----------------------------------------------------
#      Script para graficar la FFT de la señal 
#    completa y chequear con la de 'segun periodo'  
#-----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# Cargar archivo de audio
nombre_archivo = "grabacion_rapida.wav"  
frecuencia_muestreo, audio = wavfile.read(nombre_archivo)

# Convertir a mono 
if len(audio.shape) > 1:
    audio = audio[:, 0]

# Crear el eje de tiempo
duracion = len(audio) / frecuencia_muestreo
t = np.linspace(0, duracion, len(audio))

#----------------------------------------------------------------------
# Transformada de Fourier (FFT) para ver el espectro
N = len(audio)
fft_audio = np.abs(fft(audio))[:N//2]  # sólo la mitad positiva x ser real -> simetría :)
frecuencias = np.linspace(0, frecuencia_muestreo/2, N//2)
#----------------------------------------------------------------------

# Graficar
plt.subplot(2, 1, 2)
plt.plot(frecuencias, fft_audio, color="r")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia /i/ Rapida")
plt.grid()

plt.tight_layout()
plt.show()
