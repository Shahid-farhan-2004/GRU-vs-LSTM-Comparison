import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence

sequences = [[1, 2, 3], [4, 5], [1, 1, 1], [5, 4, 3], [2, 2], [0, 1, 0]]
labels = [1, 0, 1, 0, 1, 0] 

class SequenceDataset(Dataset):
  def __init__(self,X,Y):
    super().__init__()
    self.X=X
    self.Y=Y
  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    return torch.tensor(self.X[idx],dtype=torch.long),torch.tensor(self.Y[idx],dtype=torch.long)

def collate_fn(batch):
  xs,ys=zip(*batch)
  x=pad_sequence(xs,batch_first=True,padding_value=0)
  y=torch.tensor(ys)
  return x,y


loader=DataLoader(SequenceDataset(sequences,labels),batch_size=2,shuffle=True,collate_fn=collate_fn)

class BaseRNN(nn.Module):
  def __init__(self,mode="lstm"):
    super().__init__()
    self.embedding=nn.Embedding(6,16)
    if mode=="lstm":
      self.rnn=nn.LSTM(16,32,batch_first=True)
    elif mode=="gru":
      self.rnn=nn.GRU(16,32,batch_first=True)
    self.fc=nn.Linear(32,2)

  def forward(self,x):
    x=self.embedding(x)
    out,_=self.rnn(x)
    return self.fc(out[:,-1,:])
model_lstm=BaseRNN("lstm")
model_gru=BaseRNN("gru")

def train_model(model,name):
  criterion=nn.CrossEntropyLoss()
  optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
  print(f"training {name}")
  for epoch in range(5):
    losser=0
    for x,y in loader:
      outputs=model(x)
      loss=criterion(outputs,y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losser+=loss.item()
    print(f"the average loss is {(losser/len(loader)):.2f}")

train_model(model_lstm,"lstm")
train_model(model_gru,"gru")        
