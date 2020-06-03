from SpeakerRecognition import SpeakerRecognition
# from TrainModel import TrainModel

import pickle


def create_dataframe():
    sr = SpeakerRecognition()
    sr.load_data_format()
    sr.save_dataframe()
    # sr.train_model()


def training():
    df = load_dataframe()
    num_classes = get_num_speakers(df)
    # trainer = TrainModel(df, num_classes)
    # trainer.train_complete()
    sr = SpeakerRecognition()
    sr.train_model_df(df, num_classes)


def test():
    sr = SpeakerRecognition()
    sr.test_model_from_file()


def load_dataframe(path=''):
    f = open(path + 'dataframe.pkl', 'rb')
    return pickle.load(f)


def get_num_speakers(df):
    print('num speakers: ', len(df['speaker'].unique()))
    return len(df['speaker'].unique())


def main():
    #create_dataframe()
    #training()
    test()


if __name__ == "__main__":
    main()
