#-------------------------------------------------------------------------------------------
#                Algoritmo para disminuir velocidad de la señal rápida  
#                               por TFCT (matriz)
# TFCT -> Interpolacion (duplico frames) -> iTFCT (reconstuyo la señal en el dom del tiempo)
#-------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.interpolate import interp1d

# Cargar la señal
x_fast, fs = librosa.load("grabacion_rapida.wav", sr=None)

#----------------------------------------------------------
# TFCT 
X_fast = librosa.stft(x_fast, n_fft=1024, hop_length=256)  # Transforma. al dominio tiempo-frecuencia / n_fft = ventana / hop_length = solapam. entre ventanas (separacion temporal)
                                                                            
# Interpolación para ralentizar (duplicar columnas)
n_bins, n_frames = X_fast.shape    # X_fast.shape → matriz de cómo varía el cont. frecuencial del audio en el tiempo / n_bins = (1+ n_fft)/ 2 = componentes de frec.
n_frames_slow = n_frames * 2     # Se duplican las columnas (frames), porque queremos el doble de duración
new_time = np.linspace(0, n_frames - 1, n_frames_slow)  # empieza en o termina en n_frames-1, con n_frames-slow canitdad de muestras
                                                        # me da como resutado el doble de muestras en cantidad, osea un array con doble de muestras q el original 


# Interpolar parte real e imaginaria
real_interp = interp1d(np.arange(n_frames), X_fast.real, kind='linear', axis=1)
imag_interp = interp1d(np.arange(n_frames), X_fast.imag, kind='linear', axis=1)

X_slow = real_interp(new_time) + 1j * imag_interp(new_time)  # se reconstruye X_slow, que es la versión ralentizada en frecuencia

# iTFCT
x_slow = librosa.istft(X_slow, hop_length=256)
#-------------------------------------------------------------

#  Guardar audio resultante
sf.write("picasso_rapida_lenta_TFCT.wav", x_slow, fs)

# Graficar
plt.figure(figsize=(10, 4)) 
 
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_slow), ref=np.max), sr=fs, hop_length=256, y_axis='linear', x_axis='time', cmap='gray') 

plt.title("Espectrograma - Picasso rápida ralentizada x0.5 (TFCT)")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
