import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Cargar señal
x, sr = librosa.load("grabacion_lenta.wav", sr=None)
#x, sr = librosa.load("grabacion_rapida.wav", sr=None)

# Parámetros STFT
n_fft = 1024
hop_length = 256

# 1. TFCT: Calcular la STFT
D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

# Separar magnitud y fase
magnitude, phase = np.abs(D), np.angle(D)

# Factor de cambio de velocidad (por ejemplo x2)
rate = 2.0
#rate = 0.5 # ------- Factor de escala (ralentizar = aumentar duración)


# Vector de nuevos tiempos (indices fraccionales de columnas)
time_steps = np.arange(0, D.shape[1], rate)

# Inicializar matrices para la magnitud y fase re-muestreadas
magnitude_stretched = np.zeros((magnitude.shape[0], len(time_steps)), dtype=magnitude.dtype)
phase_stretched = np.zeros_like(magnitude_stretched)

# ---- Phase vocoder (ajuste de fase) ----
phase_acc = phase[:, 0]  # fase acumulada inicial (primer frame)

# Calculamos la diferencia de fase entre frames consecutivos
phase_advances = np.diff(phase, axis=1)
# Repetimos la última columna para que tenga el mismo tamaño
phase_advances = np.hstack((phase_advances, phase_advances[:, -1][:, None]))

# Interpolamos magnitud y calculamos la fase acumulada para cada frame nuevo
for i, t in enumerate(time_steps):
    t_floor = int(np.floor(t))
    t_ceil = min(t_floor + 1, D.shape[1] - 1)
    alpha = t - t_floor

    # Interpolación lineal de magnitud entre frames originales
    magnitude_stretched[:, i] = (1 - alpha) * magnitude[:, t_floor] + alpha * magnitude[:, t_ceil]

    # Ajuste de fase acumulada
    if i == 0:
        phase_stretched[:, i] = phase[:, 0]
    else:
        # Acumulamos la diferencia de fase para mantener la coherencia temporal
        phase_acc += phase_advances[:, t_floor]
        phase_stretched[:, i] = phase_acc
# ----------------------------------------

# Reconstrucción STFT con la magnitud y fase corregida
D_stretched = magnitude_stretched * np.exp(1j * phase_stretched)

# 3. Síntesis temporal: iSTFT para obtener la señal en el dominio temporal
y = librosa.istft(D_stretched, hop_length=hop_length) #NO LA USO, ES PARA ESCUCHAR SI QUISIERA AGREGANDO: import IPython.display as ipd,  ipd.Audio(y_slow, rate=sr)

# Graficar espectrograma en escala gris
plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_stretched), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time', cmap='gray')
plt.title('Phase Vocoder - Señal lenta acelerada x2')  
#plt.title('Phase Vocoder - Señal rápida ralentizada x0.5')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()






