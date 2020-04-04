import pydub
import Processor
import numpy as np
import matplotlib.pyplot as plt

files = ['E', 'A', 'D', 'G', 'B', 'High_E', 'E_major', 'A_minor']

file = 'E_major'
    
signal = pydub.AudioSegment.from_file('./' + file + '.m4a', format='m4a').get_array_of_samples()
info = pydub.utils.mediainfo('./' + file + '.m4a')

fs = float(info['sample_rate'])

time_stamps = np.linspace(0, (1/fs) * len(signal), len(signal))

note_freq = Processor.NoteFinder(signal, fs)

print(file, note_freq, 'Hz')