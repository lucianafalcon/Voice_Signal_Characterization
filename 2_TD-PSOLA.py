#--------------------------------------------------------------------------------------
#       Algoritmo TD-PSOLA para cambiar el pich (dominio temporal)
# 1- Detectar los sonidos sonoros (periodicos)
# 2- Localizar los picos de cada ciclo de los sonoros -> osea corto las ventanas temporales centradas en cada ciclo
# 3- Aplicar la ventana en cada pico, desplazarla temporalmente y sumarla -> pego ventanas cada T0/alfa, alfa=0.7 para hombre→mujer y 1.4 para mujer→hombre
#-------------------------------------------------------------------------------------

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import get_window, find_peaks
import matplotlib.pyplot as plt
import librosa.display

# Cargar audio
y, sr = librosa.load("grabacion_lenta.wav", sr=None)
y = y / np.max(np.abs(y))  # Normalizar amplitud
y_trim, _ = librosa.effects.trim(y)  # Recortar silencios al inicio y final

# Estimar pitch
f0, voiced_flag, _ = librosa.pyin(y_trim, fmin=50, fmax=300, sr=sr)
if f0 is None or np.sum(~np.isnan(f0)) < 2:
    raise ValueError("No se detectó pitch claramente.")

# Parámetros de pitch
f0_mean = np.nanmean(f0[voiced_flag])
factor = 0.7  #                                                    <--------
new_f0 = f0_mean * factor   # la nueva distancia = T0/alfa

frame_len = len(y_trim) // len(f0)
voiced_mask = np.repeat(voiced_flag, frame_len)

# Ajustar longitud de voiced_mask a y_trim
if len(voiced_mask) > len(y_trim):
    voiced_mask = voiced_mask[:len(y_trim)]
else:
    voiced_mask = np.pad(voiced_mask, (0, len(y_trim) - len(voiced_mask)), mode='constant')

y_voiced = y_trim * voiced_mask

frame_period = int(sr / f0_mean)
new_period = int(sr / new_f0)

# Detectar picos pitch-sincrónicos en la señal sonora
peaks, _ = find_peaks(y_voiced, distance=frame_period)
window_len = 2 * frame_period
window = get_window('hann', window_len)

# Extraer segmentos centrados en picos
segments = []
for p in peaks:
    start = p - frame_period
    end = p + frame_period
    if start < 0 or end > len(y_trim):
        continue
    segment = y_trim[start:end] * window
    segments.append(segment)

# Reconstruir señal manteniendo duración original
output = np.zeros(len(y_trim))

# Posiciones para insertar segmentos usando nuevo periodo
positions = np.arange(peaks[0], peaks[0] + new_period * len(segments), new_period)

# Para que la salida tenga la misma longitud que la original, recortamos posiciones
positions = positions[positions + window_len//2 < len(output)]

# Añadir segmentos sin solapamiento excesivo ni recorte final
for i, pos in enumerate(positions):
    start = int(pos - window_len // 2)
    end = start + window_len
    if start < 0 or end > len(output):
        continue
    output[start:end] += segments[i]

# Normalizar para evitar clipping
output = output / np.max(np.abs(output))

# Guardar resultado
sf.write("voz_modificada_PSOLA.wav", output, sr)
print("Archivo 'voz_modificada_PSOLA.wav' guardado correctamente.")

# Graficar original vs modificado
plt.figure(figsize=(12, 4))
plt.title("Señal original vs modificada (TD-PSOLA)")
librosa.display.waveshow(y_trim, sr=sr, alpha=0.5, label='Original')
librosa.display.waveshow(output, sr=sr, color='r', alpha=0.5, label='Modificada')
plt.legend()
plt.tight_layout()
plt.show()

