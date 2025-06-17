import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# === 1. Cargar el archivo de audio ===
fs, signal = wavfile.read('grabacion_lenta.wav')  # Cambiá por el nombre correcto

# Si es estéreo, tomar un solo canal
if signal.ndim > 1:
    signal = signal[:, 0]

# Normalizar señal
signal = signal / np.max(np.abs(signal))

# === 2. Definir intervalo de la vocal "a" ===
start_time = 1.49  # en segundos   
end_time = 1.53    # en segundos

#start_time = 2.16  # en segundos  <---- para vocal o 
#end_time = 2.2    # en segundos   
#start_time = 0.42  # en segundos  <---- para vocal i 
#end_time = 0.47    # en segundos


start_sample = int(start_time * fs)
end_sample = int(end_time * fs)

vocal = signal[start_sample:end_sample]

# === 3. Espectrograma de banda angosta ===
ventana_angosta =  1024  # int(0.025 * fs)  # 25 ms  # <----- si la ventana es muy grande no veo nada en el grafico! 
solapamiento_angosta = ventana_angosta // 2

f1, t1, Sxx1 = spectrogram(vocal, fs, window='hamming', nperseg=ventana_angosta, noverlap=solapamiento_angosta)

# === 4. Espectrograma de banda ancha ===
ventana_ancha = 256  # int(0.005 * fs)  # 5 ms
solapamiento_ancha = ventana_ancha // 2

f2, t2, Sxx2 = spectrogram(vocal, fs, window='hamming', nperseg=ventana_ancha, noverlap=solapamiento_ancha)

# === 5. Graficar ambos espectrogramas en escala de grises ===
plt.figure(figsize=(12, 6))

# Banda angosta
plt.subplot(2, 1, 1)
plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud', cmap='gray')
plt.title('Espectrograma de Banda Angosta – Vocal "a"')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(label='Intensidad (dB)')

# Banda ancha
plt.subplot(2, 1, 2)
plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud', cmap='gray')
plt.title('Espectrograma de Banda Ancha – Vocal "a"')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(label='Intensidad (dB)')

plt.tight_layout()
plt.show()
