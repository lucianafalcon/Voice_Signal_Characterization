import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Leer el archivo de audio
fs, data = wavfile.read('grabacion_rapida.wav')  # fs: frecuencia de muestreo

# Si el audio es estéreo, tomar solo un canal
if len(data.shape) > 1:
    data = data[:, 0]

# Crear vector de tiempo
t = np.linspace(0, len(data)/fs, num=len(data))

# Seleccionar el segmento entre segundos
start_time = 0
end_time = 3000
segment_mask = (t >= start_time) & (t <= end_time)

segment = data[segment_mask]
segment_time = t[segment_mask]

# Calcular la FFT
N = len(segment)
fft_values = np.fft.fft(segment)
fft_freqs = np.fft.fftfreq(N, d=1/fs)

# Usar solo la mitad positiva del espectro
positive_freqs = fft_freqs[:N // 2]
positive_magnitude = np.abs(fft_values[:N // 2])

# Graficar la FFT
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude, color='red')
#plt.title("FFT del Segmento 0.895s - 0.935s (Señal /o/ rápida)")
plt.title("FFT de 1-Período (Señal /o/ rápida)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.show()

