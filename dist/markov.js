#!/usr/bin/env node
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Load base names JSON
const dataPath = path.resolve(__dirname, '../data/base_names.json');
let sampleNames;
try {
    const raw = readFileSync(dataPath, 'utf-8');
    const js = JSON.parse(raw);
    sampleNames = Array.isArray(js.base_names) ? js.base_names : [];
}
catch (err) {
    console.error(`Could not read ${dataPath}:`, err.message);
    process.exit(1);
}
let ORDER = 2;
const START = () => '^'.repeat(ORDER);
const END = '$';
let chain = {};
/**
 * Build the Markov chain with the given order.
 */
export function train(names, order = 2) {
    ORDER = order;
    chain = {};
    const S = START(), E = END;
    for (const n of names) {
        const padded = S + n.toLowerCase() + E;
        for (let i = 0; i <= padded.length - ORDER - 1; i++) {
            const gram = padded.substr(i, ORDER);
            const nxt = padded.charAt(i + ORDER);
            (chain[gram] || (chain[gram] = [])).push(nxt);
        }
    }
}
/**
 * Generate a single name from the current chain.
 */
export function generateName() {
    let res = '', gram = START();
    while (true) {
        const opts = chain[gram];
        if (!opts)
            break;
        const nxt = opts[Math.floor(Math.random() * opts.length)];
        if (nxt === END)
            break;
        res += nxt;
        gram = gram.substr(1) + nxt;
    }
    return res.charAt(0).toUpperCase() + res.slice(1);
}
/**
 * Generate `count` names with the given `order`, each between `minLength` and `maxLength` (inclusive).
 */
export function generateNames(count = 1, order = 2, minLength = 4, maxLength = 12) {
    // rebuild chain if needed
    if (Object.keys(chain).length === 0 || ORDER !== order) {
        train(sampleNames, order);
    }
    const out = [];
    while (out.length < count) {
        const name = generateName();
        if (name.length >= minLength && name.length <= maxLength) {
            out.push(name);
        }
    }
    return out;
}
