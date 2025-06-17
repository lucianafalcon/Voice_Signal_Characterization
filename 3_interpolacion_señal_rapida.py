import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.interpolate import interp1d

# 1. Cargar la señal original rápida
x_rapida, fs = librosa.load("grabacion_rapida.wav", sr=None)

# 2. Crear nuevo eje temporal (más largo)
duracion_original = len(x_rapida)
factor = 2  # queremos ralentizar x2
nuevo_largo = duracion_original * factor

# Eje temporal original y nuevo
t_original = np.linspace(0, duracion_original - 1, duracion_original)
t_nuevo = np.linspace(0, duracion_original - 1, nuevo_largo)

# 3. Interpolación lineal
interp_func = interp1d(t_original, x_rapida, kind='linear')
x_lenta_interpolada = interp_func(t_nuevo)

# 4. Guardar señal ralentizada
sf.write("picasso_rapida_interpolada.wav", x_lenta_interpolada, fs)

# 5. Graficar espectrograma en escala de grises
X = librosa.stft(x_lenta_interpolada, n_fft=1024, hop_length=256)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X), ref=np.max),
                         sr=fs, hop_length=256, y_axis='linear', x_axis='time', cmap='gray')
plt.title("Picasso rápida intepolada (lenta x0.5 velocidad)")
plt.colorbar(format='%+2.0f dB')
plt.ylim(0, 4000)  #<--- el espectograma daba hasta 20kz
plt.tight_layout()
plt.show()
