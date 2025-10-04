import torch
import torchaudio.transforms as transforms
import torch.nn as nn
import librosa
import numpy as np

SR = 16000
AUD_LENGTH = 30*SR

def audio_preprocessing(signal, sr, transform):


    # resample
    if sr != SR:
        resampler = transforms.Resample(sr, SR)
        signal = resampler(signal)

    # pad/crop
    if signal.shape[0] > AUD_LENGTH:
        signal = signal[:, :AUD_LENGTH]
    if signal.shape[0] < AUD_LENGTH:
        pad_len = AUD_LENGTH - signal.shape[0]
        signal = nn.functional.pad(signal, (0, pad_len))

    # mel spectrogram
    if transform is not None:
        signal = transform(signal)

    return signal


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.autopool = nn.AdaptiveAvgPool2d((7,7))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7*7*64,128)
        self.logit = nn.Linear(128,1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.autopool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.logit(x)
        return x



model = AudioCNN()
model.load_state_dict(torch.load('model_weights.pth',map_location=torch.device('cpu')),strict=False)

device = torch.device('cpu')

mel_spectrogram = transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft = 1024,
    hop_length = 512,
    n_mels = 64
)

def predict(audio_path):
    model.eval()
    with torch.no_grad():
        signal,sr = librosa.load(audio_path)
        signal = torch.tensor(signal, dtype=torch.float32)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        signal = audio_preprocessing(signal,sr,mel_spectrogram)
        
        signal = signal.unsqueeze(0)
        
        pred = model(signal)
        pred_label = torch.sigmoid(pred)
        pred_label = (pred_label > 0.5).int()
        if pred_label == 1:
            return 'female'
        else:
            return 'male'
        