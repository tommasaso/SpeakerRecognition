from FeatureExtractor import FeatureExtractor

import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os


class DataFormat:

    def __init__(self, name, path2files, path2spklabel):
        self.name = name
        self.df = None
        self.df_reduced = None
        self.data = None
        self.sample = None
        self.features_label = None
        self.path2files = path2files  # '/Users/tommasaso/Applications/AI/SpeakerRecognition/DataSet/train-other-500'
        self.path2spklabel = path2spklabel  # '/Users/tommasaso/Applications/AI/SpeakerRecognition/DataSet/SPEAKERS.TXT'
        self.speakers = None
        self.traindDataset = 'train-other-500'

    # Load Files from dataset
    def load_files(self):
        print('[FormatData] loading files...')
        files = [file for r, d, f in os.walk(self.path2files) for file in f]
        files = list(filter(lambda s: s.endswith('.flac'), files))
        self.df = pd.DataFrame(files, columns=["files"])
        speaker = []
        subfolder = []
        for i in range(0, len(files)):
            speaker.append(self.df['files'][i].split('-')[0])
            subfolder.append(self.df['files'][i].split('-')[1])
        self.df['speaker'] = speaker
        self.df['subfolder'] = subfolder
        print(self.df.head())

    def add_speaker_label(self):
        print('[FormatData] add speaker label...')
        # add speaker name column
        self.data = pd.read_fwf(self.path2spklabel, delimiter="|",
                                header=None, dtype=None, encoding='utf-8', index=False, comment=';',
                                names=['speaker', 'gender', 'dataset', 'time', 'speaker_name'])
        self.data[['speaker']] = self.data[['speaker']].applymap(np.int64)

    def fn(self, row):
        try:
            return self.data[(self.data['speaker'] == int(row['speaker'])) & (
                self.data.dataset.str.contains(self.traindDataset, case=False))].speaker_name.values[0]
        except:
            return None

    def add_speaker_column(self):
        print('[FormatData] add speaker column...')
        self.df['speaker_name'] = self.df.apply(self.fn, axis=1)
        self.df.dropna(subset = ['speaker_name'], inplace=True)

    def reduce_dataset(self):
        print('[FormatData] reduce dataset...')
        # take a random sub sample of total speaker
        # df_reduced = df.sample(frac=0.05, replace=True, random_state=1)
        self.df = shuffle(self.df)
        self.speakers = self.df['speaker'].unique()
        self.speakers = self.speakers[:round(self.df['speaker'].nunique() / 20)]
        print(self.speakers)
        if not ('0' in self.speakers):
            self.speakers = np.append(self.speakers, '0')
        if not ('1' in self.speakers):
            self.speakers = np.append(self.speakers, '1')
        self.df_reduced = self.df[self.df['speaker'].isin(self.speakers)]

        # Checking the number of speakers or the number of different people in our voice data
        print("Number of sample: " + str(len(self.df)))
        print("Number of sample reduced: " + str(len(self.df_reduced)))
        print("Number of speaker: " + str(self.df['speaker'].nunique()))
        print("Number of speaker reduced: " + str(self.df_reduced['speaker'].nunique()))

        self.df_reduced = self.df_reduced.copy()
        self.df_reduced['speaker'] = self.df_reduced['speaker'].astype(int)
        self.sample = 0

    def feature_extraction(self):
        print('[FormatData] extract features...')
        fe = FeatureExtractor(self.path2files, self.sample)
        # apply feature extraction to each row of the dataframe
        self.features_label = self.df_reduced.apply(fe.extract_features, axis=1)

    def concat_features(self):
        print('[FormatData] concat features...')
        # We create an empty list where we will concatenate all the features into one long feature
        # for each file to feed into our neural network
        features = []
        for i in range(0, len(self.features_label)):
            features.append(np.concatenate((self.features_label.iloc[i][0], self.features_label.iloc[i][1],
                                            self.features_label.iloc[i][2], self.features_label.iloc[i][3],
                                            self.features_label.iloc[i][4]), axis=0))
        # print(features)

        features_data = np.array(features)
        print(features_data)
        for i in range(0, len(features[0])):
            self.df_reduced['feature_' + str(i)] = features_data[:, i]

    def get_reduced_dataframe(self):
        return self.df_reduced

    def get_num_speakers(self):
        return len(self.speakers)
