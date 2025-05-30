#!/usr/bin/env python3
"""
/**
 * This file is licensed under the European Union Public License (EUPL) v1.2.
 * You may only use this work in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed "as is",
 * without any warranty or conditions of any kind.
 *
 * Copyright (c) 2024- Andre Karlsson.. All rights reserved.
 *
 * Created on 5/30/25 :: 13:54 BY joyider <andre(-at-)sess.se>
 *
 * This file :: train_names.py is part of the cool_name_generator project.
 */
 """
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 1) Load data
with open('../data/base_names.json') as f:
    names = json.load(f)['base_names']

# 2) Build char vocab
chars = sorted({ch for name in names for ch in name.lower()})
stoi = {ch:i+1 for i,ch in enumerate(chars)}    # reserve 0 for padding

# inject an explicit EOS (end-of-sequence) token
eos_idx = len(stoi) + 1
stoi['<EOS>'] = eos_idx

itos = {i:ch for ch,i in stoi.items()}


# 3) Dataset
class NameDataset(Dataset):
    def __init__(self, names, seq_len=10):
        self.data = []
        for n in names:
            seq = [0]*seq_len + [stoi[ch] for ch in n.lower()] + [stoi['<EOS>']]
            for i in range(len(seq)-seq_len):
                self.data.append((seq[i:i+seq_len], seq[i+1:i+seq_len+1]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x,y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# 4) Model: simple one-layer RNN
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb=16, hidden=32):
        super().__init__()
        self.e = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.r = nn.GRU(emb, hidden, batch_first=True)
        self.o = nn.Linear(hidden, vocab_size)
    def forward(self, x, h=None):
        x = self.e(x)
        out, h = self.r(x, h)
        return self.o(out), h

# 5) Train
ds = NameDataset(names)
dl = DataLoader(ds, batch_size=64, shuffle=True)
model = CharRNN(len(stoi)+1)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(20):
    total = 0
    for x,y in dl:
        logits, _ = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {epoch} loss {total/len(dl):.3f}")

# 6) Export to ONNX
#   we export a single-step cell: it takes (prev_char, hidden_state) and returns (logits, new_hidden)
dummy_x = torch.zeros(1, 1, dtype=torch.long)
dummy_h = torch.zeros(1, 1, 32)
torch.save(model.state_dict(), '../data/model.pth')
torch.onnx.export(
    model, (dummy_x, dummy_h),
    "../data/model.onnx",
    input_names=["x","h"],
    output_names=["logits","h_out"],
    dynamic_axes={
      "x":   {0:"batch",1:"seq"},
      "h":   {1:"batch"},
      "logits": {0:"batch",1:"seq"}
    },
    opset_version=13)
with open('../data/vocab.json','w',encoding='utf-8') as f:
    json.dump({'stoi': stoi, 'itos': itos}, f, ensure_ascii=False, indent=2)
print("Exported model.onnx")

