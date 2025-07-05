#-----------------------------------------------------
#      Script para graficar la FFT de partes 
#                  de la señal 
#-----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cargar el audio
fs, data = wavfile.read("grabacion_rapida.wav")
if len(data.shape) > 1:
    data = data[:, 0]

# Estimar la frecuencia fundamental
f0 = 200  # Hz                                                   <----- AJUSTAR 
T = 1 / f0
periodo_muestras = int(fs * T)

# Seleccionar una parte central de la señal para evitar ruido inicial
tiempo_inicio = 0.42  # en segundos                              <----- AJUSTAR   
inicio_muestra = int(tiempo_inicio * fs)
vocal = 'i'     # vocal q estoy analizando                       <----- AJUSTAR 

# FFT de 1 período
segmento_1_periodo = data[inicio_muestra:inicio_muestra + periodo_muestras]

# FFT de varios períodos 
num_periodos = 5
segmento_varios = data[inicio_muestra:inicio_muestra + num_periodos * periodo_muestras]

#  FFT funcion ------------ creo la funcion 'calcular_fft'
def calcular_fft(segmento, fs):  
    N = len(segmento)
    fft_vals = np.fft.fft(segmento)
    fft_freqs = np.fft.fftfreq(N, d=1/fs)
    return fft_freqs[:N//2], np.abs(fft_vals[:N//2])
#--------------------------------------------------

# Calcular FFTs ------ llamo a la funcion y le paso los parametros 
frecs_1, mag_1 = calcular_fft(segmento_1_periodo, fs)     # al fin! devuelve los ejes x e y para el grafico de 1 periodoo
frecs_varios, mag_varios = calcular_fft(segmento_varios, fs)   #idem pero para el grafico de varios periodos

# Graficar 
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(frecs_1, mag_1, color='orange')
plt.title(f"FFT de 1 período de /{vocal}/")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frecs_varios, mag_varios, color='green')
plt.title(f"FFT de {num_periodos} períodos de /{vocal}/")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)

plt.tight_layout()
plt.show()
