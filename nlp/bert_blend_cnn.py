# -*- coding:utf-8 -*-
# bert融合textcnn思想的Bert+Blend-CNN
# model: Bert+Blend-CNN
# date: 2021.10.11 18:06:11

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 2
epoches = 100
model = "bert-base-chinese"
hidden_size = 768
n_class = 2
maxlen = 8

encode_layer=12
filter_sizes = [2, 2, 2]
num_filters = 3

# data，构造一些训练数据
sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
labels = [1, 1, 1, 0, 0, 0]  # 1积极, 0消极.

class MyDataset(Data.Dataset):
  def __init__(self, sentences, labels=None, with_labels=True,):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels
  def __len__(self):
    return len(sentences)

  def __getitem__(self, index):
    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=maxlen,  
                    return_tensors='pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = self.labels[index]
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids

train = Data.DataLoader(dataset=MyDataset(sentences, labels), batch_size=batch_size, shuffle=True, num_workers=1)

class TextCNN(nn.Module):
  def __init__(self):
    super(TextCNN, self).__init__()
    self.num_filter_total = num_filters * len(filter_sizes)
    self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False)
    self.bias = nn.Parameter(torch.ones([n_class]))
    self.filter_list = nn.ModuleList([
      nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
    ])

  def forward(self, x):
    # x: [bs, seq, hidden]
    x = x.unsqueeze(1) # [bs, channel=1, seq, hidden]
    
    pooled_outputs = []
    for i, conv in enumerate(self.filter_list):
      h = F.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
      mp = nn.MaxPool2d(
        kernel_size = (encode_layer-filter_sizes[i]+1, 1)
      )
      # mp: [bs, channel=3, w, h]
      pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
      pooled_outputs.append(pooled)
    
    h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [bs, h=1, w=1, channel=3 * 3]
    h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
    
    output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]

    return output

# model
class Bert_Blend_CNN(nn.Module):
  def __init__(self):
    super(Bert_Blend_CNN, self).__init__()
    self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
    self.linear = nn.Linear(hidden_size, n_class)
    self.textcnn = TextCNN()
    
  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    # 取每一层encode出来的向量
    # outputs.pooler_output: [bs, hidden_size]
    hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
    cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
    # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
    for i in range(2, 13):
      cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
    # cls_embeddings: [bs, encode_layer=12, hidden]
    logits = self.textcnn(cls_embeddings)
    return logits

bert_blend_cnn = Bert_Blend_CNN().to(device)

optimizer = optim.Adam(bert_blend_cnn.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()

# train
sum_loss = 0
total_step = len(train)
for epoch in range(epoches):
  for i, batch in enumerate(train):
    optimizer.zero_grad()
    batch = tuple(p.to(device) for p in batch)
    pred = bert_blend_cnn([batch[0], batch[1], batch[2]])
    loss = loss_fn(pred, batch[3])
    sum_loss += loss.item()

    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
      print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
  train_curve.append(sum_loss)
  sum_loss = 0

# test
bert_blend_cnn.eval()
with torch.no_grad():
  test_text = ['我不喜欢打篮球']
  test = MyDataset(test_text, labels=None, with_labels=False)
  x = test.__getitem__(0)
  x = tuple(p.unsqueeze(0).to(device) for p in x)
  pred = bert_blend_cnn([x[0], x[1], x[2]])
  pred = pred.data.max(dim=1, keepdim=True)[1]
  if pred[0][0] == 0:
    print('消极')
  else:
    print('积极')

pd.DataFrame(train_curve).plot() # loss曲线

