# cool-name-generator

A powerful fantasy name generator powered by ONNX models and Markov chains, packaged for CLI use or import into other Node.js projects.

## Features

- Generates pronounceable fantasy names using an ONNX model
- Command-line interface with options for fine-tuning
- Easily importable into other TypeScript/JavaScript projects

---

## Installation

### Option 1: CLI (Global Use)

```bash
npm install -g cool-name-generator
```

### Option 2: Local Project Use

```bash
npm install cool-name-generator
```

---

## Project Structure

```
cool_name_generator/
├── package.json
├── package-lock.json
├── scripts/
│   ├── artistic_names
│   ├── deterministic_names
│   ├── generate_new_names.py
│   ├── get_base_names.py
│   ├── holistic_names
│   ├── requirements.txt
│   └── train_names.py
├── src/
│   ├── bin.js
│   ├── cli.ts
│   ├── index.ts
│   └── markov.ts
├── data/
│   ├── base_names.json
│   ├── model.onnx
│   ├── model.pth
│   └── vocab.json
└── tsconfig.json
```

---

## Usage

### CLI

After installation, use the `cool-name-generator` command:

```bash
cool-name-generator [options]
```

### Options

| Option                     | Description                                                                 | Default |
|---------------------------|-----------------------------------------------------------------------------|---------|
| `-n, --number <n>`        | How many names to generate                                                  | `1`     |
| `-s, --sequence-length <n>` | The fixed-length sliding window (`seqLen`) for input prediction context     | `10`    |
| `-m, --min-length <l>`    | Minimum generated name length                                               | `4`     |
| `-x, --max-length <l>`    | Maximum generated name length                                               | `12`    |
| `-p, --threshold <t>`     | Threshold for pronounceability score (`-4` is good default)                 | `-4`    |
| `-t, --temperature <t>`   | Sampling temperature; lower = safer, higher = crazier names                 | `0.8`   |

### Example

```bash
cool-name-generator -n 5 -m 5 -x 10 -p -4 -t 0.7
```

---

## Programmatic Use

```ts
import { generateName } from 'cool-name-generator';

const name = await generateName({
  seqLen: 10,
  minLen: 5,
  maxLen: 10,
  temperature: 0.8
});

console.log(name);
```

---

## Development

### Build

```bash
npm run build
```

This will compile `src/` into the `dist/` directory and include your `bin.js`.

### Link for Local Testing

```bash
npm link
```

This will make `cool-name-generator` available globally on your system.

To unlink:
```bash
npm unlink -g cool-name-generator
```

---

## Note

Ensure that `model.onnx` and `vocab.json` are present in the `data/` directory before running.

---

## License

EUPL 1.2