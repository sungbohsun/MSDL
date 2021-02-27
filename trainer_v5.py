import torch
import librosa
import numpy as np
import torchaudio
import torch.nn as nn

from tqdm import tqdm 
from torch import optim
from torch.nn import ConstantPad1d
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



file = 'audio_1000.pt'
data = torch.load(file)
print(file+' load finish')
IDs = list(data.keys())
train,test = train_test_split(list(range(len(IDs))),test_size=0.2)

def sf_windows(input_,rate=22050,window_size=12,shift_size=1):
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

        x_pad = np.repeat(0,rate*299 - x.shape[0])
        y_pad = np.repeat(0,300 - y.shape[0])
  
        x = np.append(x,x_pad)
        y = np.append(y,y_pad)  
        
        x = sf_windows(x)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        
        return x,y
    
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.Mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050,n_fft=2048,hop_length=512) 
        self.power_db  = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
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
        self.Dropout     = nn.Dropout(p=0.2)
        
        linear = nn.Sequential()
        linear.add_module('Linear{1}',nn.Linear(in_features=21504 , out_features=1024, bias=True))
        linear.add_module('Linear{2}',nn.Linear(in_features=1024  , out_features=128, bias=True))
        linear.add_module('Linear{3}',nn.Linear(in_features=128   , out_features=1, bias=True))
        self.linear=linear
    
    def forward(self, x):
        x_FF                      = self.Mel(x)
        #print(x_FF.shape)
        x_FF                      = self.power_db(x_FF)
        #print(x_FF.shape)
        x_cnn                     = self.cnn(x_FF.flatten(start_dim=0,end_dim=1).unsqueeze(1))
        x_cnn                     = self.Dropout(x_cnn)
        #print(x_cnn.shape)
        x_cnn_tr                  = x_cnn.transpose(1,3)
        #print(x_cnn_tr.shape)
        x_cnn_tr_fl               = x_cnn_tr.flatten(start_dim=2)
        #print(x_cnn_tr_fl.shape)
        x_cnn_tr_fl_ls,_          = self.LSTM(x_cnn_tr_fl)
        x_cnn_tr_fl_ls            = self.Dropout(x_cnn_tr_fl_ls)
        #print(x_cnn_tr_fl_ls.shape)
        x_cnn_tr_fl_ls_fl         = x_cnn_tr_fl_ls.flatten(start_dim=1)
        #print(x_cnn_tr_fl_ls_fl.shape)
        x_cnn_tr_fl_ls_fl_li      = self.linear(x_cnn_tr_fl_ls_fl)     
        #print(x_cnn_tr_fl_ls_fl_li.shape)
        
        return x_cnn_tr_fl_ls_fl_li
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

model = CRNN().float().to(device)
loss_fn = nn.MSELoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
score_trains,score_vals = [],[]
best_loss = 10000
er = 0
epoch = 0
batch_size = 1

#while er <=10 :   
for i in range(200):
    epoch += 1
    #train
    num = 1
    t_loss = 0

    for i in tqdm(range(int(len(train)/batch_size))):
        xs = torch.Tensor([])
        ys = torch.Tensor([])

        for t in train[i*batch_size:(i+1)*batch_size]:
            x,y = process(t)
            xs = torch.cat((xs,x.unsqueeze(0)))
            ys = torch.cat((ys,y.unsqueeze(0)))


        DataLoader_x = DataLoader(xs,batch_size=batch_size,shuffle=False, num_workers=0)
        DataLoader_y = DataLoader(ys,batch_size=batch_size,shuffle=False, num_workers=0)

        for x_,y_ in zip(DataLoader_x,DataLoader_y):

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
                t_loss += loss.cpu().detach().numpy()
                num += 1

    print('epoch',epoch,'train:',float(t_loss/num),end=' ')
    score_trains.append(float(t_loss/num))

    #eval
    v_t_loss = 0
    num = 1

    for i in range(int(len(test)/batch_size)):
        xs = torch.Tensor([])
        ys = torch.Tensor([])

        for t in train[i*batch_size:(i+1)*batch_size]:
            x,y = process(t)
            xs = torch.cat((xs,x.unsqueeze(0)))
            ys = torch.cat((ys,y.unsqueeze(0)))


        DataLoader_x = DataLoader(xs,batch_size=batch_size,shuffle=False, num_workers=0)
        DataLoader_y = DataLoader(ys,batch_size=batch_size,shuffle=False, num_workers=0)

        for x_,y_ in zip(DataLoader_x,DataLoader_y):

            for x,y in zip(x_,y_):

                x = torch.Tensor(x.float())
                y = torch.Tensor(y.float())

                model.eval()
                output = model(x.to(device))
                loss = loss_fn(output, y.to(device))
                v_t_loss += loss.cpu().detach().numpy()
                num += 1

    print('epoch',epoch,'val:',float(v_t_loss/num))
    score_vals.append(float(v_t_loss/num))
    
    torch.save(model.state_dict(), 'model/1000/best_model'+str(epoch)+'.pt')
    
    
# if v_t_loss < best_loss: 
#     best_loss = v_t_loss
#     torch.save(model.state_dict(), 'model/1000/best_model'+str(epoch)+'.pt')
#     er = 0
# else : 
#     er += 1
    

loss_save  = {}
loss_save['score_trains'] = score_trains
loss_save['score_vals'] = score_vals
torch.save(loss_save, 'loss_save_'+str(epoch)+'.pt')
print('stop at epoch:',epoch)