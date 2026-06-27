/**
 * Retry an async operation with exponential backoff for transient errors.
 * @param {Function} fn — Async function to retry
 * @param {Object} [options]
 * @param {number} [options.maxRetries=3]
 * @param {number} [options.baseDelay=1000]
 * @param {Function} [options.onRetry] — Called with (error, attempt) before each retry
 * @returns {Promise<any>}
 */
export async function withRetry(fn, { maxRetries = 3, baseDelay = 1000, onRetry } = {}) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        } catch (err) {
            if (attempt === maxRetries || !isRetryable(err)) throw err;
            if (onRetry) onRetry(err, attempt);
            const delay = baseDelay * Math.pow(2, attempt - 1);
            await new Promise(r => setTimeout(r, delay));
        }
    }
}

function isRetryable(err) {
    const msg = err?.message || '';
    const status = err?.status;
    if (status === 429 || status === 503 || status === 502) return true;
    if (/fetch failed|ECONNRESET|ETIMEDOUT|ENOTFOUND|econnrefused|network|timeout/i.test(msg)) return true;
    return false;
}
