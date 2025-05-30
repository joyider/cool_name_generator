#!/usr/bin/env node
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
 * Copyright (c) 2024- Tenforward AB. All rights reserved.
 *
 * Created on 5/30/25 :: 09:54 BY joyider <andre(-at-)sess.se>
 *
 * This file :: index.ts is part of the cool_name_generator project.
 */
// src/index.ts
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import type { InferenceSession, Tensor } from 'onnxruntime-node';

// ——————————————————————————————
// Setup ONNX runtime & vocab once per import
// ——————————————————————————————
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
const ortPkg = isNode ? 'onnxruntime-node' : 'onnxruntime-web';
const { InferenceSession: _Session, Tensor: _Tensor } = await import(ortPkg);
export type { InferenceSession, Tensor };

const vocabPath = path.resolve(__dirname, '../data/vocab.json');
const { stoi, itos }: {
  stoi: Record<string,number>,
  itos: Record<string,string>
} = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));

const modelPath = isNode
    ? path.resolve(__dirname, '../data/model.onnx')
    : '../data/model.onnx';
export const session = await _Session.create(modelPath!);

// ——————————————————————————————
// Library functions
// ——————————————————————————————

/** Returns the average log‐prob per char under your RNN. */
export async function nameScore(name: string): Promise<number> {
  let hidden = new _Tensor('float32', new Float32Array(32), [1,1,32]);
  let sumLog = 0, N = 0;

  for (const ch of name.toLowerCase()) {
    const idx = stoi[ch] || 0;
    const x = new _Tensor(
        isNode ? 'int64' : 'int32',
        isNode
            ? BigInt64Array.from([BigInt(idx)])
            : Int32Array.from([idx]),
        [1,1]
    );
    const out = await session.run({ x, h: hidden } as any);
    const logits = (out.logits.data as Float32Array);
    const freq = logits[idx] ?? 0;
    const total = logits.reduce((s,v)=>s+Math.exp(v), 0);
    const prob = Math.exp(freq) / total;
    sumLog += Math.log(prob + Number.EPSILON);
    N++;
    hidden = out.h_out as InstanceType<typeof _Tensor>;
  }

  return sumLog / N;
}

/** True if the RNN thinks `name` is “easy to say.” */
export async function isPronounceable(
    name: string,
    threshold = -4
): Promise<boolean> {
  const score = await nameScore(name);
  return score > threshold;
}

/** Generate one fantasy name, respects <EOS>, min/max, temperature… */
export async function generateName({
                                     seqLen      = 10,
                                     minLen      = 6,
                                     maxLen      = 12,
                                     temperature = 0.8
                                   } = {}): Promise<string> {
  const eosIdx = stoi['<EOS>'];
  let hidden = new _Tensor('float32', new Float32Array(32), [1,1,32]);
  const inputSeq: number[] = Array(seqLen).fill(0);
  const chars: string[]   = [];

  while (chars.length < maxLen) {
    const lastIdx = inputSeq[inputSeq.length - 1];
    const idxArr = isNode
        ? BigInt64Array.from([BigInt(lastIdx)])
        : Int32Array.from([lastIdx]);
    const x = new _Tensor(isNode ? 'int64' : 'int32', idxArr, [1,1]);
    const out = await session.run({ x, h: hidden } as any);
    hidden = out.h_out as InstanceType<typeof _Tensor>;

    const ld = (out.logits.data as Float32Array).map(v => v / temperature);
    const exps   = ld.map(Math.exp);
    const sumExp = exps.reduce((a,b) => a+b, 0);
    const probs  = exps.map(e => e / sumExp);

    // sample
    const r = Math.random(), acc = { sum: 0 };
    let idx = probs.findIndex(p => { acc.sum += p; return r < acc.sum; });
    if (idx < 0) idx = probs.length - 1;

    if (idx === eosIdx) {
      if (chars.length >= minLen) break;
      else continue;
    }
    if (idx === 0) continue;

    chars.push(itos[idx]);
    inputSeq.push(idx);
  }

  if (!chars.length) return '';
  const name = chars.join('');
  return name[0].toUpperCase() + name.slice(1);
}
