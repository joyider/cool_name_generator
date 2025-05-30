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
import fs from 'fs';
import path from 'path';
import { Command } from 'commander';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const isNode = typeof process !== 'undefined' && !!process.versions?.node;
const ortPkg = isNode ? 'onnxruntime-node' : 'onnxruntime-web';
const { InferenceSession, Tensor } = await import(ortPkg);

const vocabPath = path.resolve(__dirname, '../data/vocab.json');
const { stoi, itos }: { stoi: Record<string,number>, itos: Record<string,string> } =
  JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));

const modelPath = isNode
  ? path.resolve(__dirname, '../data/model.onnx')
  : '/model.onnx';
const session = await InferenceSession.create(modelPath);

async function nameScore(name: string): Promise<number> {
  let prev = 0;
  type OrtTensor = InstanceType<typeof Tensor>;
  let hidden: OrtTensor = new Tensor('float32', new Float32Array(32), [1,1,32]);
  let sumLog = 0, N = 0;

  for (const ch of name.toLowerCase()) {
    const idx = stoi[ch] || 0;
    const x = new Tensor(isNode ? 'int64' : 'int32',
                         isNode
                           ? BigInt64Array.from([BigInt(idx)])
                           : Int32Array.from([idx]),
                         [1,1]);
    const feeds = isNode
      ? { x, h: hidden }
      : { x, h: hidden };
    const out = await session.run(feeds as any);
    const logits = (out.logits.data as Float32Array);
    const freq = logits[idx] ?? 0;
    const total = logits.reduce((s,v)=>s+Math.exp(v), 0);
    const prob = Math.exp(freq) / total;
    sumLog += Math.log(prob + Number.EPSILON);
    N++;
    hidden = out.h_out as OrtTensor;
  }

  return sumLog / N;
}

async function isPronounceable(name: string, threshold = -3): Promise<boolean> {
  const score = await nameScore(name);
  return score > threshold;
}

import { generateNames as genMarkov } from './markov.js';

const names: string[] = JSON.parse(
    fs.readFileSync(path.resolve(__dirname, '../data/base_names.json'), 'utf-8')
).base_names;

const VOWELS = new Set(['a','e','i','o','u','y']);

const patterns = names.map(n =>
    n
        .toLowerCase()
        .split('')
        .map(ch => (VOWELS.has(ch) ? 'V' : 'C'))
        .join('')
);

const vowelPool = names
    .flatMap(n => n.toLowerCase().split(''))
    .filter(ch => VOWELS.has(ch));

const consPool = names
    .flatMap(n => n.toLowerCase().split(''))
    .filter(ch => /[a-z]/.test(ch) && !VOWELS.has(ch));

export function generatePatternName(): string {
  const pat = patterns[Math.floor(Math.random() * patterns.length)];
  let s = '';
  for (const sym of pat) {
    const pool = sym === 'V' ? vowelPool : consPool;
    s += pool[Math.floor(Math.random() * pool.length)];
  }

  return s.charAt(0).toUpperCase() + s.slice(1);
}

export async function generateFiltered(
    count        = 1,
    order        = 1,
    minLen       = 4,
    maxLen       = 12,
    threshold    = -5,
    maxAttempts  = count * 5000000
): Promise<string[]> {
  const out: string[] = [];
  let attempts = 0;
  let thr = threshold;

  while (out.length < count && attempts < maxAttempts) {
    attempts++;
    const cand = generatePatternName();
    //const cand  = genMarkov(1, order, minLen, maxLen)[0];
    const score = await nameScore(cand);
    //console.log(`"${cand}" → score ${score.toFixed(2)}`);
    if (score > thr) out.push(cand);
  }

  // while (out.length < count && thr > -20) {
  //   thr -= 2;
  //   console.warn(`Relaxing threshold to ${thr} and retrying…`);
  //   let innerAttempts = 0;
  //   while (out.length < count && innerAttempts < maxAttempts) {
  //     innerAttempts++;
  //     const cand = genMarkov(1, order, minLen, maxLen)[0];
  //     let score = await nameScore(cand);
  //     console.log(`"${cand}" → score ${score.toFixed(2)}`);
  //     if (score> thr) {
  //       out.push(cand);
  //     }
  //   }
  //   attempts += innerAttempts;
  // }

  // if (out.length < count) {
  //   console.warn(
  //       `ML filter too strict; filling remaining ${count - out.length} slots with only Markov.`
  //   );
  //   while (out.length < count) {
  //     out.push(genMarkov(1, order, minLen, maxLen)[0]);
  //   }
  // }

  console.info(`Generated ${out.length}/${count} names (threshold ended at ${thr}).`);
  return out;
}

export async function generateONNXName(
    session: typeof InferenceSession,
    stoi: Record<string, number>,
    itos: Record<string, string>,
    {
      seqLen    = 10,
      minLen    = 6,
      maxLen    = 12,
      temperature = 0.8
    } = {}
): Promise<string> {
  const eosIdx = stoi['<EOS>'];

  let hidden = new Tensor(
      'float32',
      new Float32Array(32),
      [1, 1, 32]
  );

  const inputSeq: number[] = Array(seqLen).fill(0);
  const chars: string[]   = [];

  while (true) {
    if (chars.length >= maxLen) break;

    const idxArr = new (isNode ? BigInt64Array : Int32Array)([ BigInt(inputSeq[inputSeq.length - 1]) ]);
    const x = new Tensor(isNode ? 'int64' : 'int32', idxArr, [1, 1]);

    const feeds = { x, h: hidden } as any;
    const { logits, h_out } = await session.run(feeds) as {
      logits: typeof Tensor, h_out: typeof Tensor
    };
    hidden = h_out;

    const logitData = logits.data as Float32Array;

    const scaled = logitData.map(v => v / temperature);
    const exps   = scaled.map(v => Math.exp(v));
    const sumExp = exps.reduce((a,b) => a + b, 0);
    const probs  = exps.map(v => v / sumExp);

    const rnd = Math.random();
    let acc = 0, idx = 0;
    for (; idx < probs.length; idx++) {
      acc += probs[idx];
      if (rnd < acc) break;
    }

    if (idx === eosIdx) {
      if (chars.length >= minLen) break;
      else continue;
    }

    if (idx === 0) {
      continue;
    }

    chars.push(itos[idx]);
    inputSeq.push(idx);
  }

  if (chars.length === 0) return '';
  const name = chars.join('');
  return name[0].toUpperCase() + name.slice(1);
}

async function main() {
  const program = new Command()
    .name('cool-name-generator')
    .description('Generate random fantasy names')
    .option('-n, --number     <n>', 'how many names', '1')
    .option('-o, --order      <o>', 'Markov order', '2')
    .option('-m, --min-length <l>', 'minimum name length', '4')
    .option('-x, --max-length <l>', 'maximum name length', '12')
    .option('-t, --threshold  <s>', 'score threshold', '-3');

  program.action(async opts => {
    const n  = parseInt(opts.number,    10) || 1;
    const o  = parseInt(opts.order,     10) || 2;
    const ml = parseInt(opts.minLength, 10) || 4;
    const xl = parseInt(opts.maxLength, 10) || 12;
    const th = parseFloat(opts.threshold) || -3;

    const names = await generateFiltered(n, o, ml, xl, th);
    const name = await generateONNXName(session, stoi, itos);
    console.log(name);
    if (n>1) names.forEach(nm => console.log(nm));
  });

  program.parse(process.argv);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
