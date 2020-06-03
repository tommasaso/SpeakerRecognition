import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import pickle


def save_label_encoder(lb):
    f_encoder = open('LabelEncoder.pkl', 'wb')
    pickle.dump(lb, f_encoder)


def save_scaler(ss):
    f_scaler = open('StandardScaler.pkl', 'wb')
    pickle.dump(ss, f_scaler)


class TrainModel:
    def __init__(self, data, num_classes):
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.features_cols = None
        self.numSpeakers = num_classes
        self.model = None
        self.history = None
        self.scaler_label = None
        self.label_encoder = None

    def split_dataset(self):
        # split dataset in train test val set
        self.features_cols = [col for col in self.data if col.startswith('feature')]
        x = self.data[self.features_cols].to_numpy()
        y = self.data['speaker_name'].to_numpy()
        lb = LabelEncoder()
        y_dec = lb.fit_transform(y)
        save_label_encoder(lb)
        self.label_encoder = lb

        x_train, x_test_val, y_train, y_test_val = train_test_split(x, y_dec, test_size=0.3, shuffle=False)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test_val[int(len(x_test_val) * 0.5):len(x_test_val)]
        self.x_val = x_test_val[0:int(len(x_test_val) * 0.5)]
        self.y_test = y_test_val[int(len(y_test_val) * 0.5):len(y_test_val)]
        self.y_val = y_test_val[0:int(len(y_test_val) * 0.5)]

    def scaler(self):
        # normalize data with a scaler
        ss = StandardScaler()
        self.x_train = ss.fit_transform(self.x_train)
        self.x_val = ss.transform(self.x_val)
        self.x_test = ss.transform(self.x_test)
        self.scaler_label = ss
        save_scaler(ss)

    def nn_layout(self):
        # define the structure of the neural network
        self.model = Sequential()

        self.model.add(Dense(len(self.features_cols), input_shape=(193,), activation='relu'))
        self.model.add(Dropout(0.1))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.numSpeakers, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    def train(self):
        # train the model
        # fitting the model with the train data and validation with the validation data
        # we used early stop with patience 100 because we did not want to use early stop
        # I leave the early stop regularization code in case anyone wants to use it
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
        self.history = self.model.fit(self.x_train, to_categorical(self.y_train), batch_size=256, epochs=100,
                                      validation_data=(self.x_val, to_categorical(self.y_val)),
                                      callbacks=[early_stop])

    def check_train_val_accuracy(self):
        # Check out our train accuracy and validation accuracy over epochs.
        train_accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']

        # Set figure size.
        plt.figure(figsize=(12, 8))

        # Generate line plot of training, testing loss over epochs.
        plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
        plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

        # Set title
        plt.title('Training and Validation Accuracy by Epoch', fontsize=25)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Categorical Crossentropy', fontsize=18)
        plt.xticks(range(0, 100, 5), range(0, 100, 5))

        plt.legend(fontsize=18)

    def compute_metrics(self):
        from sklearn import metrics
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import f1_score
        from sklearn.metrics import roc_curve, plot_roc_curve
        from sklearn.metrics import precision_score, recall_score
        import seaborn as sn
        print('***METRICS***')
        y_pred = self.model.predict_classes(self.x_test)
        # Accuracy
        print("Model Accuracy: " + str(accuracy_score(self.y_test, y_pred, normalize=True, sample_weight=None)))
        # MSE
        print("MSE: " + str(mean_squared_error(self.y_test, y_pred)))
        # Precision
        print("Precision: " + str(precision_score(self.y_test, y_pred, average=None)))
        # F1
        print("Recall: " + str(recall_score(self.y_test, y_pred, average=None)))
        # F1
        print("F1: " + str(f1_score(self.y_test, y_pred, average=None)))
        # Confusion matrix
        confusion_matrix = pd.crosstab(self.y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print('Confusion Matrix', confusion_matrix)
        plt.subplots(figsize=(20, 15))
        sn.heatmap(confusion_matrix, annot=True)
        # EER

    def save_model(self, path=''):
        self.model.save(path+'saved_model/my_model')

    def get_model(self):
        return self.model

    def train_complete(self):
        self.split_dataset()
        self.scaler()
        self.nn_layout()
        self.train()
        self.check_train_val_accuracy()
        self.compute_metrics()
        self.save_model()
        self.model = self.get_model()
