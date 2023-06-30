import torch
#import torch.nn as nn
'''
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = BertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("such an adorable idiot", return_tensors="pt")
#outputs = model(**inputs)

class haha(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            self.encoder = model.encoder
            
    def forward():
        print("hahaha")
        
class BertEncoder(nn.Module): ### dealing with tensors, attentiontype='col'
    def __init__(self, *args, **kwargs):
        super().__init__()
        bert = AutoModel.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            self.encoder = bert.encoder
        
    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1) # dim=1: combine on rows
        with torch.no_grad():
            return self.encoder(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = BertEncoder()
        
    def forward():
        print("hahahaha")
bert = AutoModel.from_pretrained("bert-base-uncased")
encoder = bert.encoder
print("ha")
model = BertEncoder()
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")
print("haha")