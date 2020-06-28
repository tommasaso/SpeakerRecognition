from SpeakerRecognition import SpeakerRecognition
import os
import pickle

path2files = '/Users/tommasaso/AI/SpeakerRecognition/DataSet/train-other-500'
path2spklabel = '/Users/tommasaso/AI/SpeakerRecognition/DataSet/SPEAKERS.TXT'

def create_dataframe():
    sr = SpeakerRecognition(path2files, path2spklabel)
    sr.load_data_format()
    sr.save_dataframe()
    # sr.train_model()


def training():
    df = load_dataframe()
    num_classes = get_num_speakers(df)
    # trainer = TrainModel(df, num_classes)
    # trainer.train_complete()
    sr = SpeakerRecognition(path2files, path2spklabel)
    sr.train_model_df(df, num_classes)


def test(file_name, speaker, subfolder):
    sr = SpeakerRecognition(path2files, path2spklabel)
    sr.test_model_from_file(file_name, speaker, subfolder)


def load_dataframe(path=''):
    f = open(path + 'dataframe.pkl', 'rb')
    return pickle.load(f)


def get_num_speakers(df):
    print('num speakers: ', len(df['speaker'].unique()))
    return len(df['speaker'].unique())

def modelExist():
    return ( os.path.exists('/Users/tommasaso/AI/SpeakerRecognition/saved_model/my_model/saved_model.pb') and 
        os.path.exists('/Users/tommasaso/AI/SpeakerRecognition/StandardScaler.pkl') and 
        os.path.exists('/Users/tommasaso/AI/SpeakerRecognition/LabelEncoder.pkl') )

def main():
    if not modelExist():
        create_dataframe()
        training()
    test("1-13-0077.flac", "1", "13")

if __name__ == "__main__":
    main()