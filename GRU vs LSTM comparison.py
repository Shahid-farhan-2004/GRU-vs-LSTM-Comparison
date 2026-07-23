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
  def __getitem(self,idx):
    return torch.tensor(self.X,dtype=torch.long),torch.tensor(self.Y,dtype=torch.long)

def collate_fn(batch):
  xs,ys=zip(*batch)
  x=pad_sequence(xs,batch_first=True,padding_value=0)
  y=torch.tensor(ys)
  return 
