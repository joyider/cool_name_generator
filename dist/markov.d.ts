#!/usr/bin/env node
/**
 * Build the Markov chain with the given order.
 */
export declare function train(names: string[], order?: number): void;
/**
 * Generate a single name from the current chain.
 */
export declare function generateName(): string;
/**
 * Generate `count` names with the given `order`, each between `minLength` and `maxLength` (inclusive).
 */
export declare function generateNames(count?: number, order?: number, minLength?: number, maxLength?: number): string[];
