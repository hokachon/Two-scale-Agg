import os
import soundfile as sf
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from keras.callbacks import ModelCheckpoint, EarlyStopping
from cfg2019 import Config
from keras import backend as K
from keras import Model
from keras.optimizers import Adam
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def check_data():
    if os.path.isfile(config.p_path):
        print('loading existing data for {} model'. format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = joblib.load(handle)
            return tmp
    else:
        return None



def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]

    X = []
    Xf = []

    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        wav, rate = sf.read('audio17/' + file)

        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]

        X_sample = mfcc(sample, rate, winlen=0.02, winstep=0.02, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        #print(X_sample)
        Xf_sample = mfcc(sample, rate, winlen=0.02, winstep=0.02, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        #print(Xf_sample)



        _min = min(np.amin(X_sample), np.amin(Xf_sample),  _min)
        _max = max(np.amax(X_sample), np.amax(Xf_sample),  _max)


        X.append(X_sample)
        Xf.append(Xf_sample)

        y.append(classes.index(label))

    config.min = _min
    config.max = _max

    X, Xf,  y = np.array(X), np.array(Xf), np.array(y)
    print(X.shape)
    print(Xf.shape)

    print(y.shape)

    X = (X - _min) / (_max - _min)
    Xf = (Xf - _min) / (_max - _min)


    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        Xf = Xf.reshape(Xf.shape[0], Xf.shape[1], Xf.shape[2], 1)

        print(X.shape)
        print(Xf.shape)


    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    y = to_categorical(y, num_classes=10)
    print(y.shape)

    #config.data = (X, y)
    config.data = (X, Xf, y)
    with open(config.p_path, 'wb') as handle:
        joblib.dump(config, handle, compress=3)

    return X, Xf, y


# multiscale model

def get_conv_model():

    main_model = Sequential()
    main_model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))
    main_model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))
    main_model.add(Flatten())


    second_model = Sequential()
    second_model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape2))
    second_model.add(MaxPool2D((2, 2), padding='same'))
    second_model.add(Dropout(0.3))
    second_model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    second_model.add(MaxPool2D((2, 2), padding='same'))
    second_model.add(Dropout(0.3))
    second_model.add(Flatten())


    # first model upper
    main_model = Sequential()

    main_model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),padding='same', input_shape=input_shape))
    main_model.add(BatchNormalization())
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))

    main_model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    main_model.add(BatchNormalization())
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))

    main_model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    main_model.add(BatchNormalization())
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))

    main_model.add(Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    main_model.add(BatchNormalization())
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))

    main_model.add(Conv2D(1024, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    main_model.add(BatchNormalization())
    main_model.add(MaxPool2D((2, 2), padding='same'))
    main_model.add(Dropout(0.3))

    main_model.add(Flatten())


    # second model lower
    lower_model1 = Sequential()
    lower_model1.add(MaxPool2D((2, 2), padding='same', input_shape=input_shape2))
    lower_model1.add(BatchNormalization())

    lower_model1.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    lower_model1.add(BatchNormalization())
    lower_model1.add(MaxPool2D((2, 2), padding='same'))
    lower_model1.add(Dropout(0.3))

    lower_model1.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    lower_model1.add(BatchNormalization())
    lower_model1.add(MaxPool2D((2, 2), padding='same'))
    lower_model1.add(Dropout(0.3))

    lower_model1.add(Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    lower_model1.add(BatchNormalization())
    lower_model1.add(MaxPool2D((2, 2), padding='same'))
    lower_model1.add(Dropout(0.3))

    lower_model1.add(Conv2D(1024, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    lower_model1.add(BatchNormalization())
    lower_model1.add(MaxPool2D((2, 2), padding='same'))
    lower_model1.add(Dropout(0.3))

    lower_model1.add(Conv2D(2048, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    lower_model1.add(BatchNormalization())
    lower_model1.add(MaxPool2D((2, 2), padding='same'))
    lower_model1.add(Dropout(0.3))

    lower_model1.add(Flatten())



    # merged models

    merged_model = Concatenate()([main_model.output, lower_model1.output])
    x = Dense(2048, activation='relu')(merged_model)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    output = Dense(15, activation='softmax')(x)
    model = Model(inputs=[main_model.input, lower_model1.input], outputs=[output])
    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['acc'])
    print(K.eval(model.optimizer.lr))
    model.summary()

    return model




# read the data
df = pd.read_csv('train17.csv')
print(df)
df.set_index('fname', inplace=True)

for f in df.index:

    signal, rate = sf.read('audio17/' +f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

# number of samples
#n_samples = int(df['length'].sum())
n_samples = 1 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

config = Config(mode='conv')

if config.mode == 'conv':
    X, Xf, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    input_shape2 = (Xf.shape[1], Xf.shape[2], 1)

    model = get_conv_model()

'''
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y. axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
'''

#checkpoint
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

#earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience= 20, verbose= 1, restore_best_weights=False)

# weight
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

model.fit([X, Xf], y, epochs=100, batch_size=64, shuffle=True, validation_split=0.1, class_weight=class_weight, callbacks=[checkpoint])

model.save(config.model_path)
