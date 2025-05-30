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
 * Copyright (c) 2024- Andre Karlsson. All rights reserved.
 *
 * Created on 5/30/25 :: 09:59 BY joyider <andre(-at-)sess.se>
 *
 * This file :: get_base_names.py is part of the cool_name_generator project.
 */
 """
import torch
import json
import torch.nn.functional as F
import random

with open('../data/vocab.json', 'r') as f:
    vocab = json.load(f)
stoi = vocab['stoi']
itos = {int(i): ch for i, ch in vocab['itos'].items()}

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, emb=16, hidden=32):
        super().__init__()
        self.e = torch.nn.Embedding(vocab_size, emb, padding_idx=0)
        self.r = torch.nn.GRU(emb, hidden, batch_first=True)
        self.o = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.e(x)
        out, h = self.r(x, h)
        return self.o(out), h

model = CharRNN(len(stoi)+1)
model.load_state_dict(torch.load('../data/model.pth'))
model.eval()

def generate_name(
    model, stoi, itos,
    seq_len=10,
    min_len=6,
    max_len=12,
    temperature=0.9
):
    model.eval()
    eos_idx = stoi['<EOS>']
    with torch.no_grad():
        h = None
        input_seq = [0] * seq_len
        chars = []

        while True:
            if len(chars) >= max_len:
                break

            x = torch.tensor([input_seq[-seq_len:]], dtype=torch.long)
            logits, h = model(x, h)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1).squeeze()
            idx = torch.multinomial(probs, num_samples=1).item()

            if idx == eos_idx:
                if len(chars) >= min_len:
                    break
                else:
                    continue

            if idx == 0:
                continue

            chars.append(itos[idx])
            input_seq.append(idx)

        return ''.join(chars).capitalize()

for _ in range(10):
    name = generate_name(model, stoi, itos)
    print(f'"{name}",')
