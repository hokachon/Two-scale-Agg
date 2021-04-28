import joblib
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score


def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting the features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        #print(type(fn))
        wav, rate = sf.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]

            X = mfcc(sample, rate, winlen=0.02, winstep=0.02, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            Xf = mfcc(sample, rate, winlen=0.02, winstep=0.02, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)


            X = (X - config.min) / (config.max - config.min)
            Xf = (Xf - config.min) / (config.max - config.min)
            #print(type(X))

            if config.mode == 'conv':
                X = X.reshape(1, X.shape[0], X.shape[1], 1)
                Xf = Xf.reshape(1, Xf.shape[0], Xf.shape[1], 1)

            #else config.mode == 'time':
                #X = np.expand_dims(X, axis=0)
                #Xf = np.expand_dims(Xf, axis=0)
            y_hat = model.predict([X, Xf])
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob


# load data
df = pd.read_csv('test17.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('joblib2017', 'conv.z')

with open(p_path, 'rb') as handle:
    config = joblib.load(handle)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('test17')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i,c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('dataset2017.csv', index=False)

