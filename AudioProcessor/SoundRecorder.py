import pyaudio
import wave
import keyboard
import struct
import matplotlib.pyplot as plt
import Processor
import numpy as np
import librosa

CHUNK = 10000
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording")

frames = []

while True: 
    data = stream.read(CHUNK)
    frames.append(data)
    if keyboard.is_pressed(' '):
        print("Done recording") 
        break 
    sig = np.frombuffer(data, dtype='<i2').reshape(-1, CHANNELS)

    note = Processor.NoteFinder(sig[:, 0], RATE)
    print(note)

    bpm = Processor.BpmFinder(sig[:, 0], RATE)
    print(bpm)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()