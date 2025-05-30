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
 * Created on 5/30/25 :: 16:48PM BY joyider <andre(-at-)sess.se>
 *
 * This file :: cli.ts is part of the cool_name_generator project.
 */
import { Command } from 'commander';
import { generateName, isPronounceable } from './index.js';

async function main() {
    const program = new Command()
        .name('cool-name-generator')
        .description('Generate random fantasy names')
        .option('-n, --number <n>', 'how many names to generate', '1')
        .option('-s, --sequence-length <n>', 'seqLen', '10')
        .option('-m, --min-length <l>', 'minimum name length', '4')
        .option('-x, --max-length <l>', 'maximum name length', '12')
        .option('-p, --threshold <t>', 'pronounceability threshold', '-4')
        .option('-t, --temperature <t>', 'sampling temperature', '0.8');

    program.action(async (opts) => {
        const count       = +opts.number;
        const seqLen      = +opts.sequenceLength;
        const minLen      = +opts.minLength;
        const maxLen      = +opts.maxLength;
        const threshold   = +opts.threshold;
        const temperature = +opts.temperature;

        let generated = 0;
        while (generated < count) {
            const name = await generateName({ seqLen, minLen, maxLen, temperature });
            if (await isPronounceable(name, threshold)) {
                console.log(name);
                generated++;
            }
        }
    });

    await program.parseAsync(process.argv);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
