#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { Command } from 'commander';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Choose the right runtime depending on environment
const isNode = typeof process !== 'undefined' && !!process.versions?.node;
const ortPkg = isNode ? 'onnxruntime-node' : 'onnxruntime-web';
const { InferenceSession, Tensor } = await import(ortPkg);
// 1) Load vocab
const vocabPath = path.resolve(__dirname, '../data/vocab.json');
const { stoi, itos } = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));
// 2) Load ONNX model
const modelPath = isNode
    ? path.resolve(__dirname, '../data/model.onnx')
    : '/model.onnx'; // or wherever you serve it in your web app
const session = await InferenceSession.create(modelPath);
// 3) Score a name by average log‐prob under the RNN
async function nameScore(name) {
    let prev = 0; // start token
    let hidden = new Tensor('float32', new Float32Array(32), [1, 1, 32]);
    let sumLog = 0, N = 0;
    for (const ch of name.toLowerCase()) {
        const idx = stoi[ch] || 0;
        const x = new Tensor(isNode ? 'int64' : 'int32', isNode
            ? BigInt64Array.from([BigInt(idx)])
            : Int32Array.from([idx]), [1, 1]);
        const feeds = isNode
            ? { x, h: hidden }
            : { x, h: hidden }; // same keys for both runtimes
        const out = await session.run(feeds);
        const logits = out.logits.data;
        // probability of the actual next char
        const freq = logits[idx] ?? 0;
        const total = logits.reduce((s, v) => s + Math.exp(v), 0);
        const prob = Math.exp(freq) / total;
        sumLog += Math.log(prob + Number.EPSILON);
        N++;
        hidden = out.h_out;
    }
    return sumLog / N;
}
// 4) Wrap up a pronounceability check
async function isPronounceable(name, threshold = -3) {
    const score = await nameScore(name);
    return score > threshold;
}
// 6) Combined generator with scoring
/**
 * Generate `count` names with an ML-based filter that auto-relaxes,
 *
 * and finally falls back to pure Markov if needed.
 */
// Load your base names
const names = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../data/base_names.json'), 'utf-8')).base_names;
// Define vowels
const VOWELS = new Set(['a', 'e', 'i', 'o', 'u', 'y']);
// Extract all VC‐patterns and the letter pools
const patterns = names.map(n => n
    .toLowerCase()
    .split('')
    .map(ch => (VOWELS.has(ch) ? 'V' : 'C'))
    .join(''));
// Flatten letter pools (so we preserve the same frequency mix)
const vowelPool = names
    .flatMap(n => n.toLowerCase().split(''))
    .filter(ch => VOWELS.has(ch));
const consPool = names
    .flatMap(n => n.toLowerCase().split(''))
    .filter(ch => /[a-z]/.test(ch) && !VOWELS.has(ch));
export function generatePatternName() {
    // Pick a random pattern
    const pat = patterns[Math.floor(Math.random() * patterns.length)];
    // Build the name
    let s = '';
    for (const sym of pat) {
        const pool = sym === 'V' ? vowelPool : consPool;
        s += pool[Math.floor(Math.random() * pool.length)];
    }
    // Capitalize
    return s.charAt(0).toUpperCase() + s.slice(1);
}
export async function generateFiltered(count = 1, order = 1, minLen = 4, maxLen = 12, threshold = -5, // starting threshold
maxAttempts = count * 5000000) {
    const out = [];
    let attempts = 0;
    let thr = threshold;
    // 1) First pass: use current threshold
    while (out.length < count && attempts < maxAttempts) {
        attempts++;
        const cand = generatePatternName();
        //const cand  = genMarkov(1, order, minLen, maxLen)[0];
        const score = await nameScore(cand);
        //console.log(`"${cand}" → score ${score.toFixed(2)}`);
        if (score > thr)
            out.push(cand);
    }
    // 2) Auto-relax loop
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
    // 3) Final fallback: pure Markov
    // if (out.length < count) {
    //   console.warn(
    //       `ML filter too strict; filling remaining ${count - out.length} slots with pure Markov.`
    //   );
    //   while (out.length < count) {
    //     out.push(genMarkov(1, order, minLen, maxLen)[0]);
    //   }
    // }
    console.info(`Generated ${out.length}/${count} names (threshold ended at ${thr}).`);
    return out;
}
/**
 * Generate a single fantasy name by sampling from the ONNX RNN.
 */
export async function generateONNXName(session, stoi, itos, { seqLen = 10, minLen = 6, maxLen = 12, temperature = 0.8 } = {}) {
    // index of your explicit <EOS> token
    const eosIdx = stoi['<EOS>'];
    // initialize hidden state: [1,1,hiddenSize]
    let hidden = new Tensor('float32', new Float32Array(32 /* your hidden size */), [1, 1, 32]);
    // start with seqLen padding tokens (0)
    const inputSeq = Array(seqLen).fill(0);
    const chars = [];
    while (true) {
        // stop if we've reached maxLen characters
        if (chars.length >= maxLen)
            break;
        // build input tensor [1,1]
        const idxArr = new (isNode ? BigInt64Array : Int32Array)([BigInt(inputSeq[inputSeq.length - 1])]);
        const x = new Tensor(isNode ? 'int64' : 'int32', idxArr, [1, 1]);
        // run the model
        const feeds = { x, h: hidden };
        const { logits, h_out } = await session.run(feeds);
        hidden = h_out;
        // extract the last timestep logits
        const logitData = logits.data;
        // apply temperature & softmax
        const scaled = logitData.map(v => v / temperature);
        const exps = scaled.map(v => Math.exp(v));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(v => v / sumExp);
        // sample one index
        const rnd = Math.random();
        let acc = 0, idx = 0;
        for (; idx < probs.length; idx++) {
            acc += probs[idx];
            if (rnd < acc)
                break;
        }
        // if EOS and we’ve met minLen, stop
        if (idx === eosIdx) {
            if (chars.length >= minLen)
                break;
            else
                continue; // ignore early EOS
        }
        // skip padding
        if (idx === 0) {
            continue;
        }
        // append real character
        chars.push(itos[idx]);
        inputSeq.push(idx);
    }
    // capitalize and return
    if (chars.length === 0)
        return '';
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
    program.action(async (opts) => {
        const n = parseInt(opts.number, 10) || 1;
        const o = parseInt(opts.order, 10) || 2;
        const ml = parseInt(opts.minLength, 10) || 4;
        const xl = parseInt(opts.maxLength, 10) || 12;
        const th = parseFloat(opts.threshold) || -3;
        const names = await generateFiltered(n, o, ml, xl, th);
        const name = await generateONNXName(session, stoi, itos);
        console.log(name);
        if (n > 1)
            names.forEach(nm => console.log(nm));
    });
    program.parse(process.argv);
}
// ESM‐safe “run if invoked directly” check
main().catch(err => {
    console.error(err);
    process.exit(1);
});
