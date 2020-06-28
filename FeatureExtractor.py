import os
import librosa
import numpy as np


class FeatureExtractor:

    def __init__(self, path2files, sample):
        self.sample = sample
        self.path2files = path2files  # 'DataSet/train-other-500'
        self.features = None
        self.mfccs = None
        self.stft = None
        self.chroma = None
        self.mel = None
        self.contrast = None
        self.tonnetz = None

    def extract_features(self, files, fullPath=""):
        self.sample = self.sample + 1
        # Sets the name to be the path to where the file is in my computer
        if files != None and fullPath != "":
            file_name = os.path.join(
                os.path.abspath(self.path2files) + '/' + str(files.speaker) + '/' + str(
                    files.subfolder) + '/' + str(
                    files.files))
        else:
            file_name = fullPath

        print('[Feature extraction] sample_number: '+ str(self.sample) +' - filename: ', file_name)
        # Loads the audio file as a floating point time series - default sample rate is set to 22050 by default
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        X, index = librosa.effects.trim(y=X, top_db=40)
        # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
        self.mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
        self.stft = np.abs(librosa.stft(X))
        # Computes a chromagram from a waveform or power spectrogram.
        self.chroma = np.mean(librosa.feature.chroma_stft(S=self.stft, sr=sample_rate).T, axis=0)
        # Computes a mel-scaled spectrogram.
        self.mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        # Computes spectral contrast
        self.contrast = np.mean(librosa.feature.spectral_contrast(S=self.stft, sr=sample_rate).T, axis=0)
        # Computes the tonal centroid features (tonnetz)
        self.tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        self.features = [self.mfccs, self.chroma, self.mel, self.contrast, self.tonnetz]
        return self.mfccs, self.chroma, self.mel, self.contrast, self.tonnetz

    def get_features(self):
        return self.features

    def get_mfccs(self):
        return self.mfccs

    def get_chroma(self):
        return self.chroma

    def get_mel(self):
        return self.mel

    def get_contrast(self):
        return self.contrast

    def get_tonnetz(self):
        return self.tonnetz
