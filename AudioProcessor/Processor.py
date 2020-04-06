import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

    return []