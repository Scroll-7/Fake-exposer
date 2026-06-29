import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';

describe('logger', () => {
    let logger;
    let origStdout;
    let origStderr;

    before(async () => {
        const mod = await import('../services/logger.js');
        logger = mod.logger;
        origStdout = process.stdout.write;
        origStderr = process.stderr.write;
    });

    after(() => {
        process.stdout.write = origStdout;
        process.stderr.write = origStderr;
    });

    it('exports error, warn, info, debug methods', () => {
        assert.strictEqual(typeof logger.error, 'function');
        assert.strictEqual(typeof logger.warn, 'function');
        assert.strictEqual(typeof logger.info, 'function');
        assert.strictEqual(typeof logger.debug, 'function');
    });

    it('includes ISO timestamp and level in output', () => {
        let output = '';
        process.stdout.write = (chunk) => { output += chunk; return true; };
        logger.info('test message');
        process.stdout.write = origStdout;
        assert.match(output, /\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
        assert.match(output, /\[INFO\]/);
        assert.ok(output.includes('test message'));
    });

    it('logger.error writes to stderr with ERROR level', () => {
        let output = '';
        process.stderr.write = (chunk) => { output += chunk; return true; };
        logger.error('err test');
        process.stderr.write = origStderr;
        assert.match(output, /\[ERROR\]/);
        assert.ok(output.includes('err test'));
    });

    it('logger.warn writes to stderr with WARN level', () => {
        let output = '';
        process.stderr.write = (chunk) => { output += chunk; return true; };
        logger.warn('warn test');
        process.stderr.write = origStderr;
        assert.match(output, /\[WARN\]/);
        assert.ok(output.includes('warn test'));
    });

    it('logger.info writes to stdout', () => {
        let output = '';
        process.stdout.write = (chunk) => { output += chunk; return true; };
        logger.info('stdout test');
        process.stdout.write = origStdout;
        assert.ok(output.includes('stdout test'));
    });

    it('logger.debug is silent by default (LOG_LEVEL=INFO)', () => {
        let output = '';
        process.stdout.write = (chunk) => { output += chunk; return true; };
        logger.debug('should be silent');
        process.stdout.write = origStdout;
        assert.strictEqual(output, '');
    });
});
