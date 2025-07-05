#---------------------------------------------------------------------
#    Algoritmo para disminuír velocdad de la señal rápida  
#                    por Interpolación
# Expansor (inserción de ceros) -> Filtro LP (eliminación de replicas)
#----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.interpolate import interp1d

# Cargar la señal
x_rapida, fs = librosa.load("grabacion_rapida.wav", sr=None)

#----------------------------------------------------------
# Crear nuevo eje temporal más largo osea EXPANSOR
duracion_original = len(x_rapida)
factor = 2  # osea ralentizar x2
nuevo_largo = duracion_original * factor

# Eje temporal original y nuevo
t_original = np.linspace(0, duracion_original - 1, duracion_original)
t_nuevo = np.linspace(0, duracion_original - 1, nuevo_largo)

#----------------------------------------------------------
# Interpolación lineal
interp_func = interp1d(t_original, x_rapida, kind='linear')
x_lenta_interpolada = interp_func(t_nuevo)

#----------------------------------------------------------

# Guardar señal ralentizada
sf.write("picasso_rapida_interpolada.wav", x_lenta_interpolada, fs)

# Graficar
plt.figure(figsize=(10, 4)) 

X = librosa.stft(x_lenta_interpolada, n_fft=1024, hop_length=256)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X), ref=np.max), sr=fs, hop_length=256, y_axis='linear', x_axis='time', cmap='gray') #grafico

plt.title("Picasso rápida interpolada (lenta x0.5 velocidad)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
