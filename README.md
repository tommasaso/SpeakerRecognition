# SpeakerRecognition
Recognize the speaker based on speech analysis.

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
The Speaker recognition file is the notebook containing the code for training the neural network.
The file has the task of saving the model in the /saved_model/my_model folder, the encoder of the labels corresponding to the speakers and the dataframe which corresponds to the features extracted for each audio file contained in the dataset folder.
The StandardScaler.pkl, dataframe.pkl and LabelEncoder.pkl files are registered by the Speaker Recognition notebook for use within the RealTime Eval notebook.


