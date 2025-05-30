#!/usr/bin/env node
import type { InferenceSession, Tensor } from 'onnxruntime-node';
export type { InferenceSession, Tensor };
export declare const session: any;
/** Returns the average log‐prob per char under your RNN. */
export declare function nameScore(name: string): Promise<number>;
/** True if the RNN thinks `name` is “easy to say.” */
export declare function isPronounceable(name: string, threshold?: number): Promise<boolean>;
/** Generate one fantasy name, respects <EOS>, min/max, temperature… */
export declare function generateName({ seqLen, minLen, maxLen, temperature }?: {
    seqLen?: number | undefined;
    minLen?: number | undefined;
    maxLen?: number | undefined;
    temperature?: number | undefined;
}): Promise<string>;
