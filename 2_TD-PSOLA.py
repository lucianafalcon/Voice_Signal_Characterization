import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import get_window, find_peaks

# --- Cargar señal de voz desde archivo ---
file_path = 'grabacion_lenta.wav'  # Cambiá este path si tu archivo se llama distinto
y, sr = librosa.load(file_path)

# --- Estimar frecuencia fundamental con pyin ---
f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)

# --- Determinar cuántas muestras hay por cuadro ---
frame_length = len(y) // len(f0)  # Cada valor de f0 se estima sobre un marco de N muestras

# --- Repetimos el flag de voz sonora para que tenga el mismo largo que la señal ---
voiced_mask = np.repeat(voiced_flag, frame_length)

# --- Aseguramos que ambas señales tengan exactamente la misma longitud ---
min_len = min(len(y), len(voiced_mask))
y = y[:min_len]
voiced_mask = voiced_mask[:min_len]

# --- Aplicamos la máscara: nos quedamos solo con lo que es sonoro (voz con tono) ---
y_voiced = y * voiced_mask

# --- Calculamos la frecuencia fundamental promedio solo de los tramos sonoros ---
f0_mean = np.nanmean(f0[voiced_flag])  # Solo se usa lo que realmente es voz
new_f0_mean = f0_mean * 0.7  # Cambiá este valor: 0.7 para hombre → mujer, 1.4 para mujer → hombre

# --- Calculamos el período de pitch en muestras ---
frame_period = int(sr / f0_mean)

# --- Detectamos picos dentro de la señal sonora (sincrónicos con el pitch) ---
peaks, _ = find_peaks(y_voiced, distance=frame_period)

# --- Definimos la longitud de la ventana y la generamos (ventana de Hann) ---
window_length = 2 * frame_period
window = get_window('hann', window_length)

# --- Extraemos segmentos centrados en los picos ---
segments = []
for p in peaks:
    start = p - frame_period
    end = p + frame_period
    if start < 0 or end > len(y):
        continue
    segment = y[start:end] * window
    segments.append(segment)

# --- Inicializamos la señal de salida ---
output = np.zeros(len(y))

# --- Calculamos la nueva separación entre picos (nuevo pitch) ---
new_period = int(sr / new_f0_mean)
positions = np.arange(peaks[0], peaks[0] + new_period * len(segments), new_period)

# --- Pegamos los segmentos en los nuevos tiempos, conservando duración ---
for i, pos in enumerate(positions):
    if i >= len(segments):
        break
    start = int(pos - window_length // 2)
    end = start + window_length
    if start < 0 or end > len(output):
        continue
    output[start:end] += segments[i]

# --- Normalizamos y guardamos el archivo final ---
output = output / np.max(np.abs(output))  # Evitamos clipping
sf.write('voz_modificada.wav', output, sr)
print("Archivo 'voz_modificada.wav' guardado correctamente.")

# --- Mostramos gráfico comparativo de las señales ---
plt.figure(figsize=(10, 4))
plt.title('Señal original vs señal modificada (TD-PSOLA)')
librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Original')
librosa.display.waveshow(output, sr=sr, color='r', alpha=0.5, label='Modificada')
plt.legend()
plt.tight_layout()
plt.show()

