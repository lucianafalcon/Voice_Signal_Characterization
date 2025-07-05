#-----------------------------------------------------------
#    Algoritmo para acelerar la velocdad de la señal lenta  
#                     por Decimación
# Filtro antialiasing -> decimación
#-----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, spectrogram
import soundfile as sf

# Cargar la señal
fs, picasso_lenta = wavfile.read('grabacion_lenta.wav')

# Si es estéreo, convertir a mono
if picasso_lenta.ndim > 1:
    picasso_lenta = picasso_lenta.mean(axis=1)

print("Frecuencia de muestreo original:", fs)
print("Longitud original:", len(picasso_lenta))
print("Duración original (s):", len(picasso_lenta) / fs)

#-------------------------------------------------------
# Filtrado antialiasing - Ventana de Hamming
cutoff = fs / 4  # porque vamos a tomar 1 de cada 2 muestras → Nyquist se reduce a la mitad
numtaps = 101    #número de coeficientes del filtro (orden del filtro FIR).   [ mayor numyaps: mas definicion, mas lento, mas retardo ]
nyquist = fs / 2  
filtro = firwin(numtaps, cutoff / nyquist, window='hamming')   #firwin = funcion que necesita: (#coeficientes, donde poner frec corte, tipo de ventana)
                                                               # cutoff / nyquist = (fs / 4) / (fs / 2) = 0.5 --> osea fs / 4.
picasso_filtrada = lfilter(filtro, 1.0, picasso_lenta)   # <--- lfilter(num, denom, señal de entrada) -> denom=1 -> filtro FIR, que no tiene parte recursiva, te dice que no usa salidas pasadas

# Compensar delay del filtro (porque los filtros FIR introducen un retardo lineal de delay muestras)
delay = (numtaps - 1) // 2
picasso_filtrada = picasso_filtrada[delay:]  # recorta el inicio
picasso_filtrada = picasso_filtrada[:len(picasso_lenta) - delay]  # igualar longitud
#-------------------------------------------------------

# Decimado para acelerar (sin cambiar fs)
picasso_acelerada = picasso_filtrada[::2]    # <------------ DECIMADO (ME QUEDO CON 1 MUESTRA CADA 2 DE LA SEÑAL ORIGINAL, eje 'x') 
fs_acelerada = fs  # Mantenemos fs para que se reproduzca el doble de rápido

#print("Longitud acelerada:", len(picasso_acelerada))
#print("Duración acelerada (s):", len(picasso_acelerada) / fs_acelerada)

#------------------------------------------------------
# Espectrograma
f, t, Sxx = spectrogram(picasso_acelerada, fs_acelerada)
Sxx = np.squeeze(Sxx)
print("Tiempo final en espectrograma:", t[-1])

plt.figure(figsize=(10, 4))  

plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='gray')

plt.title('Espectrograma - Picasso lenta decimada (rapida x2 velocidad)')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.colorbar(label='dB')
plt.tight_layout()

# Guardar audio decimado
picasso_acelerada_norm = picasso_acelerada / np.max(np.abs(picasso_acelerada)) # Normalizar a [-1, 1]
sf.write('picasso_lenta_decimada.wav', picasso_acelerada_norm.astype(np.float32), fs) # Guardar como float32 (explícitamente)

plt.show()
