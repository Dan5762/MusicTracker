import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
from scipy.ndimage import gaussian_filter1d

from Errors import ProcessingError

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(round(result.size/2)):]

def freq2note(freq):
    """
    Determining note using equal temperment tuning
    """
    low_E = 82.41
    dist = np.log(freq / low_E) / np.log(2**(1/12))
    notes = ['E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#'] 
    note_idx = int(round(dist) % 12)
    note_name = notes[note_idx]
    octave = int(dist // 12)
    note = note_name + ' ' + str(octave)

    return note

def NoteFinder(audio, fs):
    N = len(audio)
    freqs = np.linspace(0, fs/2, int(round(N/2)))
    freq_amp = np.fft.fft(audio)[:len(freqs)]
    freq_power = freq_amp * np.conjugate(freq_amp)
    peak_indices = np.argpartition(freq_power, -5)[-5:]
    fundamental_peak = min(peak_indices)
    found_freq = freqs[fundamental_peak]
     
    if found_freq != 0:
        note = freq2note(found_freq)
    else:
        note = 'hmmm'

    return note

def BpmFinder(audio, fs):
    # Resample signal to 8kHz
    new_fs = 8000
    f = signal.resample(audio, int(round((len(audio) / fs) * new_fs)))

    # Mel transform on short time fourier transform
    window_len = int(round(0.032 * new_fs))
    overlap_len = int(round(0.004 * new_fs))
    n_mels = 40

    S = librosa.feature.melspectrogram(f, sr=new_fs, n_fft=window_len, 
                                   hop_length=overlap_len, 
                                   n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # First order difference in band power against time
    S_DB_diff = np.diff(S_DB, axis=1)
    # Negative differences set to zero
    S_DB_diff_rect = S_DB_diff.copy()
    S_DB_diff_rect[S_DB_diff_rect < 0] = 0
    # Positive differences summed over bands
    band_sum = np.sum(S_DB_diff_rect, axis=0)
    
    # High pass filter applied to signal
    spect_fs = new_fs / overlap_len
    band_sum_smooth = butter_highpass_filter(band_sum, 0.4, spect_fs, order=3)

    # Smooth with gaussian convolution
    gauss_std = 0.02 * spect_fs
    band_sum_smooth = gaussian_filter1d(band_sum_smooth, gauss_std)
    onset_strengths = band_sum_smooth / np.std(band_sum_smooth)
    
    # suggested values
    period_bias = 0.4
    weighting_width = 1.4

    # Compute autocorrelation of signal
    autocorrelation = autocorr(onset_strengths)
    periods = np.linspace(0, len(autocorrelation) / spect_fs, len(autocorrelation))
    weights = np.exp(-0.5 * (np.log2(periods / period_bias) / weighting_width)**2)

    period_strengths = weights * autocorrelation
    bpms = [60 / period for period in periods]
    bpms = np.flip(bpms)
    period_strengths = np.flip(period_strengths)

    # Extract peak of tempo period strength function
    peaks = signal.find_peaks(period_strengths, prominence=400)
    if len(peaks[0]) == 0:
        raise ProcessingError("No peaks found")
    peak_tempos = [bpms[peak] for peak in peaks[0]]
    peak_tempo = bpms[peaks[0][int(np.argmax(peaks[1]['prominences']))]]

    return peak_tempo