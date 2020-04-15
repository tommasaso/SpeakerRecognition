# SpeakerRecognition
This repository allows you to identify the interlocutor based on the analysis of his voice.

For reasons of space, I didn't load  the dataset has not been loaded on Github. Below is the tree of all the files:
```bash
├── DataSet
│   ├── BOOKS.TXT
│   ├── CHAPTERS.TXT
│   ├── LICENSE.TXT
│   ├── README.TXT
│   ├── SPEAKERS.TXT
│   └── train-other-500
│       ├── 1
│       │   └── 13
│       │       ├── 1-13-0000.flac
│       │       ├── ....
│       │       └── 1-13-0080.flac
│       ├── 1006
│       │   └── 135212
│       │       ├── 1006-135212-0000.flac
│       │       ├── ...
│       │       ├── 1006-135212-0138.flac
│       │       └── 1006-135212.trans.txt
│       ├── 102
│       │   └── 129232
│       │       ├── ...
│       │   ├── 132091
│       │   │   ├── ...
│       │   └── 132092
│       │   ...
│
├── LabelEncoder.pkl
├── README.md
├── RealTimeEval.ipynb
├── Record.ipynb
├── Speaker Recognition.ipynb
├── StandardScaler.pkl
├── dataframe.pkl
└── saved_model
    └── my_model
        ├── assets
        ├── saved_model.pb
        └── variables
```
The Speaker recognition file is the notebook contains the code to train the neural network. \
In Speaker Recognition.ipynb notebook we: 
1. load the DataSet folder reducing the speaker number from 8072 to 60 (for computational reason) 
2. extract features from any files in the reduced dataset
3. save the resulted dataframe (since the features' extraction is time consuming)
3. scale the measures (we save the scaler to use it in RealTimeEval.ipynb file)
4. train the dense neural network 
5. save the model

Without going into too much detail, the latest model obtained the following results:
- Model Accuracy: 0.9983484723369116
- MSE: 0.7241948802642444

RealTimeEval.ipynb file continuously evaluates the microphone audio signal to detect which person is actually speaking.
Without going into the details of the code, the important parameters are as follows:
- Sec: This variable defines the time window to be analyzed
- Interval: This variable defines how often to evaluate the current time window.

For example for Sec=3 and Interval=0.5 mean that we carry out an evaluation any 0.5 seconds of the time window of the previous 3 seconds at the time of evaluation.

The last notebook Record.ipynb, instead, it is used to register a new speaker to be included in our dataset. 


