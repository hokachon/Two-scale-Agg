import os

# config
class Config:
    def __init__(self, mode='conv', nfilt=104, nfeat=52, nfft=1103, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models2019', mode + '.model')
        self.p_path = os.path.join('joblib2019', mode + '.z')

