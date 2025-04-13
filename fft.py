import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# GRAFICO DE FORMANTES ###########################################
# === CONFIGURACIÓN ===
file_path = "grabacion_rapida.wav"  # Ruta del archivo
low_freq = 150   # Frecuencia mínima a analizar (Hz)
high_freq = 2200  # Frecuencia máxima a analizar (Hz)

# === CARGAR AUDIO ===
sample_rate, data = wavfile.read(file_path)

# Si es estéreo, tomar un solo canal
if len(data.shape) > 1:
    data = data[:, 0]

# Normalizar
data = data.astype(np.float32)
data = data / np.max(np.abs(data))

# === DETECTAR REGIONES VOCALES ===
window_size = int(0.05 * sample_rate)  # Ventana de 50 ms
energy = np.array([
    np.sum(data[i:i+window_size]**2) 
    for i in range(0, len(data)-window_size, window_size)
])
threshold = 0.1 * np.max(energy)
vocal_indices = np.where(energy > threshold)[0]
vocal_samples = np.concatenate([
    data[i*window_size:(i+1)*window_size] 
    for i in vocal_indices
])

# === FFT ===
n = len(vocal_samples)
yf = fft(vocal_samples)
xf = fftfreq(n, 1 / sample_rate)

# Filtrar frecuencias positivas
yf = np.abs(yf[xf > 0])
xf = xf[xf > 0]

# === FILTRAR RANGO ELEGIDO ===
range_mask = (xf >= low_freq) & (xf <= high_freq)
xf_filtered = xf[range_mask]
yf_filtered = yf[range_mask]

# === GRAFICAR ===
plt.figure(figsize=(10, 5))
plt.plot(xf_filtered, yf_filtered, color='#FF5733')
plt.title(f"FFT entre {low_freq} Hz y {high_freq} Hz - /i/ rápida")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()

# 1 PERIODO #########################################

fs, signal = wavfile.read("grabacion_rapida.wav")  # fs es la frecuencia de muestreo

if len(signal.shape) > 1:
    signal = signal[:, 0]

# Definir F0 y calcular duración de un período
f0 = 200  # frecuencia fundamental (en Hz)
periodo_muestras = int(fs / f0)  # Número de muestras que corresponde a un período de 200 Hz

# Elegir inicio en tiempo de la vocal, del grafico anterior
inicio_segundos = 0.425  # El tiempo en segundos donde empieza el primer ciclo de la vocal
inicio_muestras = int(inicio_segundos * fs)

segmento = signal[inicio_muestras:inicio_muestras + periodo_muestras] 

fft_resultado = np.fft.fft(segmento)

frecuencias = np.fft.fftfreq(len(segmento), 1/fs)

plt.plot(frecuencias[:len(frecuencias)//2], np.abs(fft_resultado)[:len(fft_resultado)//2], color='#FF5733') 
plt.title("FFT de un solo período de /i/ rapida ")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.show()
