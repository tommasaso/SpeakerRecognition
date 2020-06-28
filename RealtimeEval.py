import pyaudio
import wave
import pandas as pd
import librosa
import glob 
import librosa.display
import os
import numpy as np
import tensorflow as tf
from threading import Thread
import time
from collections import deque
import tensorflow as tf
import _thread
from sklearn.preprocessing import LabelEncoder
import pickle 
#import Microphone

def extract_features(files):

    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(files)

    # Loads the audio file as a floating point time series - default sample rate is set to 22050 by default
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    except:
        print("extract_features(): error loading file")

    try:
        X, index = librosa.effects.trim(y=X, top_db=40) 
    except:
        print("extract_features(): error trimming the lowest power")

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    except:
        print("extract_features(): error generating mel-frequency cepstral coefficients")

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    try:
        stft = np.abs(librosa.stft(X))
    except:
        print("extract_features(): error generating short-time fourier transform")
    
    # Computes a chromagram from a waveform or power spectrogram.
    try:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    except:
        print("extract_features(): error computing chromagram")

    # Computes a mel-scaled spectrogram.
    try: 
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    except:
        print("extract_features(): error computing chromagram")
    
    # Computes spectral contrast
    try: 
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    except:
        print("extract_features(): error computing spectral contrast")
    
    # Computes the tonal centroid features (tonnetz)
    try: 
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    except:
        print("extract_features(): error computing centroid features")
    
    return mfccs, chroma, mel, contrast, tonnetz #np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=0)


def eval(mic,model,ss,lb):
    features = mic.getAudioFile()
    prob_array = model.predict_proba(ss.transform([features]))
    prob = max(prob_array[0])
    if prob > 0.1:
        pred_class = model.predict_classes(ss.transform([features]))
        print(">>> "+str(lb.inverse_transform(pred_class))+" - prob:"+str(prob))
        return str(pred_class)
    else:
        print(">>> Unknown")
        return "Unknown"

class Microphone(Thread):
    def __init__(self, Format, Chunk, Channels, Rate, Sec):
        ''' Constructor. '''  
        Thread.__init__(self)   
        self.p = pyaudio.PyAudio()
        chuck_per_second = 20480
        self.Channels = Channels
        self.Format = Format
        self.Rate = Rate
        self.Running = True
        self.Chunk = Chunk
        self.stream = self.p.open(format=Format,
                        channels=Channels,
                        rate=Rate,
                        input=True,
                        frames_per_buffer=Chunk)
        self.queue = deque(maxlen=(round(chuck_per_second*Sec/Chunk)))
 
    def run(self):
        try:
            print("Start acquisition")
            while self.Running:
                data = self.stream.read(Chunk, exception_on_overflow = False)
                self.queue.append(data)
        except:
            print("run() Error")
        
            
    def getAudioFile(self):
        try:   
            file = str(time.time())[:10]
            wf = wave.open("TempFiles/"+file+".flac", 'wb')
            wf.setnchannels(self.Channels)
            wf.setsampwidth(self.p.get_sample_size(self.Format))
            wf.setframerate(self.Rate)
            wf.writeframes(b''.join(self.queue))
            wf.close()
            features = []
            features = extract_features("TempFiles/"+file+".flac")
            os.remove("TempFiles/"+file+".flac")
            result = np.concatenate((features[0], features[1], features[2], features[3], features[4]), axis=0)
            return result
        except:
            print("getAudioFile() Error")
            return ""
        
        
    def stop(self):
        self.Running = False

def loadLabelEncoder(fileName): 
    f_encoder = open(fileName, "rb") 
    le = pickle.load(f_encoder)
    f_encoder.close()
    return le

def loadScaler(fileName):
    f_scaler = open("StandardScaler.pkl", "rb") 
    ss = pickle.load(f_scaler)
    f_scaler.close()
    return ss

'''
def main():  
    print("main funciotn")
    
if __name__ == "__main__":
    main()
'''

Chunk = 1024
Format = pyaudio.paInt16
Channels = 1
Rate = 22050
Sec = 3
Interval = 0.5

le = loadLabelEncoder("LabelEncoder.pkl")
ss = loadScaler("StandardScaler.pkl")
model = tf.keras.models.load_model('saved_model/my_model')

mic = Microphone(Format, Chunk, Channels, Rate, Sec)
mic.start()

for i in range(0,100):
    time.sleep(Interval)
    try:
        _thread.start_new_thread( eval, (mic, model, ss, le, ) )
    except:
        print("Error: unable to start thread")
        mic.stop()
mic.stop()