#!/usr/bin/env node
declare const InferenceSession: any;
export declare function generatePatternName(): string;
export declare function generateFiltered(count?: number, order?: number, minLen?: number, maxLen?: number, threshold?: number, // starting threshold
maxAttempts?: number): Promise<string[]>;
/**
 * Generate a single fantasy name by sampling from the ONNX RNN.
 */
export declare function generateONNXName(session: typeof InferenceSession, stoi: Record<string, number>, itos: Record<string, string>, { seqLen, minLen, maxLen, temperature }?: {
    seqLen?: number | undefined;
    minLen?: number | undefined;
    maxLen?: number | undefined;
    temperature?: number | undefined;
}): Promise<string>;
export {};
