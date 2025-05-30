#!/usr/bin/env node
(async () => {
    try {
        await import('./cli.js')
    } catch (e) {
        console.error(e)
        process.exit(1)
    }
})()