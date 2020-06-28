from FeatureExtractor import FeatureExtractor
from FormatData import DataFormat
from TrainModel import TrainModel

import pickle
import pandas as pd
import numpy as np
import tensorflow as tf


class SpeakerRecognition:
    def __init__(self, path2files, path2spklabel):
        self.label_encoder = None
        self.path2files = path2files
        self.path2spklabel = path2spklabel
        self.df = None
        self.loaded_df = None
        self.num_speakers = None
        self.model = None
        self.trainer = None
        self.loaded_lb = None
        self.loaded_ss = None

    def load_data_format(self):
        print('[SpeakerRecog] loading data...')
        data_format = DataFormat('my_format', self.path2files, self.path2spklabel)
        data_format.load_files()
        data_format.add_speaker_label()
        data_format.add_speaker_column()
        data_format.reduce_dataset()
        data_format.feature_extraction()
        data_format.concat_features()
        self.df = data_format.get_reduced_dataframe()
        self.num_speakers = data_format.get_num_speakers()

    def save_dataframe(self, path=''):
        f = open(path + 'dataframe.pkl', 'wb')
        pickle.dump(self.df, f)

    def load_label_encoder(self, path):
        f = open(path + 'LabelEncoder.pkl', 'rb')
        self.loaded_lb = pickle.load(f)

    def load_standard_scaler(self, path):
        f = open(path + 'StandardScaler.pkl', 'rb')
        self.loaded_ss = pickle.load(f)

    def load_features(self, path):
        f = open(path + 'features.pkl', 'rb')
        self.loaded_features = pickle.load(f)

    def load_features_label(self, path):
        f = open(path + 'features_label.pkl', 'rb')
        self.loaded_features_label = pickle.load(f)

    def copy_loaded_df_to_main_df(self):
        self.df = self.loaded_df

    def load_model(self):
        self.model = tf.keras.models.load_model('saved_model/my_model')

    def train_model(self):
        self.trainer = TrainModel(self.df, self.num_speakers)
        self.trainer.split_dataset()
        self.trainer.scaler()
        self.trainer.nn_layout()
        self.trainer.train()
        self.trainer.check_train_val_accuracy()
        self.trainer.compute_metrics()
        self.trainer.save_model()
        self.model = self.trainer.get_model()

    def train_model_df(self, df, num_speakers):
        self.trainer = TrainModel(df, num_speakers)
        self.trainer.split_dataset()
        self.trainer.scaler()
        self.trainer.nn_layout()
        self.trainer.train()
        self.trainer.check_train_val_accuracy()
        self.trainer.compute_metrics()
        self.trainer.save_model()
        self.model = self.trainer.get_model()

    def test_model_from_file(self, file_name, speaker, subfolder):
        # Test the model
        data = {"files": file_name, "speaker": speaker, 'subfolder': subfolder}
        test = pd.DataFrame(data=data, index=[0])
        ft = FeatureExtractor(self.path2files, 0)
        features_label_test = test.apply(ft.extract_features, axis=1)
        features_test = [np.concatenate((features_label_test.iloc[0][0], features_label_test.iloc[0][1],
                                         features_label_test.iloc[0][2], features_label_test.iloc[0][3],
                                         features_label_test.iloc[0][4]), axis=0)]
        test = np.array(features_test)
        if self.loaded_ss is None:
            self.load_standard_scaler('')
        if self.loaded_lb is None:
            self.load_label_encoder('')
        if self.model is None:
            self.load_model()
        test = self.loaded_ss.transform(test)
        test_pred = self.model.predict_classes(test)
        print('[SpeakerRecognition] test identification:'+self.loaded_lb.inverse_transform(test_pred)[0])
