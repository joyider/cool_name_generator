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
 * Copyright (c) 2024- Andre Karlsson. All rights reserved.
 *
 * Created on 5/30/25 :: 10:29 BY joyider <andre(-at-)sess.se>
 *
 * This file :: markov.ts is part of the cool_name_generator project.
 */
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
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
export function generateNames(count = 1, order = 2, minLength = 4, maxLength = 12) {
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
