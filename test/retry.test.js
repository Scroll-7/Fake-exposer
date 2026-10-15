import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { withRetry } from '../services/retry.js';

const fast = { baseDelay: 1 };

describe('withRetry', () => {
    it('returns immediately on success without retrying', async () => {
        let calls = 0;
        const out = await withRetry(async () => { calls++; return 'ok'; }, fast);
        assert.equal(out, 'ok');
        assert.equal(calls, 1);
    });

    it('retries a retryable (429) error and eventually succeeds', async () => {
        let calls = 0;
        const out = await withRetry(async () => {
            calls++;
            if (calls < 3) {
                const e = new Error('rate limited');
                e.status = 429;
                throw e;
            }
            return 'ok';
        }, fast);
        assert.equal(out, 'ok');
        assert.equal(calls, 3);
    });

    it('does not retry a non-retryable error', async () => {
        let calls = 0;
        await assert.rejects(
            () => withRetry(async () => {
                calls++;
                const e = new Error('bad request');
                e.status = 400;
                throw e;
            }, fast),
            /bad request/
        );
        assert.equal(calls, 1);
    });

    it('retries on retryable network messages', async () => {
        let calls = 0;
        await assert.rejects(
            () => withRetry(async () => {
                calls++;
                throw new Error('fetch failed');
            }, { ...fast, maxRetries: 2 })
        );
        assert.equal(calls, 2);
    });

    it('throws the last error after exhausting retries', async () => {
        let calls = 0;
        await assert.rejects(
            () => withRetry(async () => {
                calls++;
                const e = new Error('still failing');
                e.status = 503;
                throw e;
            }, { ...fast, maxRetries: 3 }),
            /still failing/
        );
        assert.equal(calls, 3);
    });

    it('calls onRetry once per retry (not for the final failure)', async () => {
        const seen = [];
        await assert.rejects(
            () => withRetry(async () => {
                const e = new Error('boom');
                e.status = 502;
                throw e;
            }, { ...fast, maxRetries: 3, onRetry: (err, attempt) => seen.push(attempt) })
        );
        assert.deepEqual(seen, [1, 2]);
    });

    it('makes no retry when maxRetries is 1', async () => {
        let calls = 0;
        await assert.rejects(
            () => withRetry(async () => {
                calls++;
                const e = new Error('one shot');
                e.status = 429;
                throw e;
            }, { ...fast, maxRetries: 1 })
        );
        assert.equal(calls, 1);
    });

    it('is not retryable when the error has no message or status', async () => {
        let calls = 0;
        await assert.rejects(
            () => withRetry(async () => { calls++; throw {}; }, fast)
        );
        assert.equal(calls, 1);
    });
});
