import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import librosa
warnings.filterwarnings("ignore")

from glob import glob
from torch.nn import ConstantPad1d
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader

from tqdm import tqdm 
from torch import optim
from sklearn.metrics import f1_score,confusion_matrix

file = 'audio_mini.pt'

data = torch.load(file)
print(file+' load finish')
IDs = list(data.keys())

def sf_windows(input_,rate=22050,window_size=6,shift_size=1):
    input_ = torch.Tensor(input_)
    pad_len = (window_size/2)*rate
    pad = ConstantPad1d(int(pad_len),0)
    waveform = pad(input_)

    #make a sliding window 
    x = waveform.unfold(dimension = 0,
                             size = window_size*rate,
                             step =shift_size*rate).unsqueeze(1)
    return x

def process(i,rate=22050):
        x = data[IDs[i]]['x']
        y = data[IDs[i]]['y']

        x = x[:rate*299]
        y = y[:300]

        x_con = np.repeat(0,rate*299 - x.shape[0])
        y_con = np.repeat(0,300 - y.shape[0])

        y = np.append(y,y_con)    
        x = np.append(x,x_con)

        x = sf_windows(x)
        y = torch.Tensor(y)
        
        return x,y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu'

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.fft = MelSpectrogram(n_mels=128) #Fast Fourier Transform featrue = 128   
        
        cnn = nn.Sequential()
        cnn.add_module('conv{0}',   nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{0}',   nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{0}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{0}',nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{0}',   nn.Dropout(p=0.1))
                       
        cnn.add_module('conv{1}',   nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{1}',   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{1}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{1}',nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{1}',   nn.Dropout(p=0.1))
        self.cnn=cnn
        
        self.LSTM        = nn.LSTM(input_size = 320,hidden_size = 256,num_layers=7,bidirectional=True) #input_size change buy windows size
        self.Dropout     = nn.Dropout(p=0.1)
        
        linear = nn.Sequential()
        linear.add_module('Linear{1}',nn.Linear(in_features=27648 , out_features=2764, bias=True))
        linear.add_module('Linear{2}',nn.Linear(in_features=2764  , out_features=128, bias=True))
        linear.add_module('Linear{3}',nn.Linear(in_features=128   , out_features=1, bias=True))
        self.linear=linear
    
    def forward(self, x):
        x_FF                      = self.fft(x)
        #x_FF                      = torch.Tensor(librosa.power_to_db(x_FF, ref=np.max))
        #print(x_FF.shape)
        x_cnn                     = self.cnn(x_FF.flatten(start_dim=0,end_dim=1).unsqueeze(1))
        #print(x_cnn.shape)
        x_cnn_tr                  = x_cnn.transpose(1,3)
        #print(x_cnn_tr.shape)
        x_cnn_tr_fl               = x_cnn_tr.flatten(start_dim=2)
        #print(x_cnn_tr_fl.shape)
        x_cnn_tr_fl_ls,_          = self.LSTM(x_cnn_tr_fl)
        #print(x_cnn_tr_fl_ls.shape)
        x_cnn_tr_fl_ls_fl         = x_cnn_tr_fl_ls.flatten(start_dim=1)
        #print(x_cnn_tr_fl_ls_fl.shape)
        x_cnn_tr_fl_ls_fl_li      = self.linear(x_cnn_tr_fl_ls_fl)     
        #print(x_cnn_tr_fl_ls_fl_li.shape)
        
        return x_cnn_tr_fl_ls_fl_li
    
model = CRNN().float().to(device)
loss_fn = nn.MSELoss()
learning_rate = 1e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
score_trains,score_vals = [],[]
best_loss = 10000
er = 0
epoch = 0
h = 4

from sklearn.model_selection import train_test_split
train,test = train_test_split(list(range(len(IDs))),test_size=0.2)

while er <=50 :
    epoch += 1
    t_loss = 0
    t_loss_v = 0
    batch_size = 4
    i = 0

    xs = torch.Tensor([])
    ys = torch.Tensor([])

    for t in train[i*batch_size:(i+1)*batch_size]:
        x,y = process(t)
        xs = torch.cat((xs,x.unsqueeze(0)))
        ys = torch.cat((ys,y.unsqueeze(0)))


    x_ = DataLoader(xs,batch_size=batch_size,shuffle=False, num_workers=0)
    y_ = DataLoader(ys,batch_size=batch_size,shuffle=False, num_workers=0)
        
    for x,y in zip(x_,y_):

        x = torch.Tensor(x.float())
        y = torch.Tensor(y.float())

        model.train()
        output = model(x.to(device))
        loss = loss_fn(output, y.to(device))
        if torch.flatten(torch.isnan(loss)).any():
            continue
        optimizer.zero_grad()
        loss.float().backward()
        optimizer.step()
        t_loss += loss
    print('epoch',epoch,'train:',float(t_loss.cpu().detach().numpy()),end=' ')
    

    for i in test:
        x = data[IDs[i]]['x']
        y = data[IDs[i]]['y']

        x = x[:22050*299]
        y = y[:300]

        x_con = np.repeat(0,22050*299 - x.shape[0])
        y_con = np.repeat(0,300 - y.shape[0])

        y = np.append(y,y_con)    
        x = np.append(x,x_con)

        x = sf_windows(x)

        
        x = DataLoader(x,batch_size=1,shuffle=False, num_workers=0)
        y = DataLoader(y,batch_size=1,shuffle=False, num_workers=0)
        
        for x,y in zip(x_,y_):
            x = torch.Tensor(x.float())
            y = torch.Tensor(y.float())
            model.eval()
            output = model(x.to(device))
            loss = loss_fn(output, y.to(device))
            t_loss_v += loss
    print('epoch',epoch,'train:',float(t_loss_v.cpu().detach().numpy()))

    if t_loss_v < best_loss: 
        best_loss = t_loss_v
        torch.save(model.state_dict(), 'model/best_model'+str(epoch)+'.pt')
        er = 0
    else : 
        er += 1

loss_save  = {}
loss_save['score_trains'] = score_trains
loss_save['score_vals'] = score_vals
torch.save(loss_save, 'loss_save_'+str(epoch)+'.pt')
print('stop at epoch:',epoch)
