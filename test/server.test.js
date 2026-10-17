import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';

let BASE = '';
let server = null;
let serverOnline = false;

function skipIfOffline() {
    if (!serverOnline) {
        throw new Error('Server failed to start');
    }
}

before(async () => {
    process.env.NODE_ENV = 'test';
    process.env.PORT = '0';

    const appModule = await import('../server.js');
    const app = appModule.default;

    server = app.listen(0, () => {
        const port = server.address().port;
        BASE = `http://127.0.0.1:${port}`;
        serverOnline = true;
    });

    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Server listen timeout')), 8000);
        server.on('listening', () => {
            clearTimeout(timeout);
            resolve();
        });
        server.on('error', (err) => {
            clearTimeout(timeout);
            reject(err);
        });
    });
});

after(async () => {
    if (server) {
        await new Promise(resolve => server.close(resolve));
        server = null;
    }
});

describe('Server API Endpoints', () => {

    it('serves the frontend HTML', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/`);
        assert.equal(res.status, 200);
        assert.ok(res.headers.get('content-type')?.includes('text/html'));
        const text = await res.text();
        assert.ok(text.includes('<!DOCTYPE html>'));
    });

    it('serves static CSS with cache headers', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/css/style.css`);
        assert.equal(res.status, 200);
        assert.ok(res.headers.get('content-type')?.includes('text/css'));
        assert.ok(res.headers.get('cache-control')?.includes('max-age'));
    });

    it('/api/analyze/text returns 400 for empty body', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'Text is required');
    });

    it('/api/analyze/url returns 400 for empty body', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'URL is required');
    });

    it('/api/analyze/image returns 400 without file', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST' });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'Image is required');
    });

    it('/api/analyze/image rejects GIF uploads with a clear message', async () => {
        skipIfOffline();
        const form = new FormData();
        const gifBytes = Uint8Array.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00]);
        form.append('image', new Blob([gifBytes], { type: 'image/gif' }), 'animated.gif');
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST', body: form });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.match(data.error, /JPG|PNG/);
    });

    it('/api/analyze/image rejects files with unknown magic bytes', async () => {
        skipIfOffline();
        const form = new FormData();
        const txtBytes = Uint8Array.from([0x48, 0x65, 0x6C, 0x6C, 0x6F]); // "Hello"
        form.append('image', new Blob([txtBytes], { type: 'image/png' }), 'fake.png');
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST', body: form });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.match(data.error, /JPEG and PNG are accepted/);
    });

    it('/api/analyze/image returns instant fake verdict for AI-tool filenames', async () => {
        skipIfOffline();
        const form = new FormData();
        const pngBytes = Uint8Array.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        form.append('image', new Blob([pngBytes], { type: 'image/png' }), 'chatgpt_image.png');
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST', body: form });
        assert.equal(res.status, 200);
        const data = await res.json();
        assert.equal(data.verdict, 'Confirmed Fake / AI-Generated');
        assert.equal(data.credibility_score, 5);
    });

    it('/api/analyze/url returns 400 for non-http(s) schemes', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: 'file:///etc/passwd' })
        });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'Only http/https URLs are supported');
    });

    it('returns 404 for unknown routes', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/nonexistent`);
        assert.equal(res.status, 404);
    });

    it('returns JSON error for invalid API route', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/nonexistent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 404);
        const data = await res.json();
        assert.ok(data.error.includes('Not found'));
    });

    it('returns JSON error for unknown API path', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/unknown/route`, {
            method: 'GET'
        });
        assert.equal(res.status, 404);
        const data = await res.json();
        assert.ok(data.error.includes('Not found'));
    });

    it('/api/face/known returns face API status', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/face/known`);
        assert.equal(res.status, 200);
        const data = await res.json();
        assert.ok('available' in data);
        assert.ok('known_faces_count' in data);
        assert.ok(Array.isArray(data.players));
    });

    describe('/api/detect/text (ZeroGPT-style AI text detector)', () => {

        it('returns 400 for empty body', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            assert.equal(res.status, 400);
            const data = await res.json();
            assert.equal(data.error, 'Text is required');
        });

        it('returns valid detection result for short human-like text', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: 'I went to the store yesterday and bought some milk. It was pretty expensive actually.' })
            });
            assert.equal(res.status, 200);
            const data = await res.json();
            assert.ok(typeof data.overallScore === 'number');
            assert.ok(data.overallScore >= 0 && data.overallScore <= 100);
            assert.ok(Array.isArray(data.sentences));
            assert.ok(data.sentences.length > 0);
            assert.ok(typeof data.aiSentencePercentage === 'number');
        });

        it('returns higher AI probability for repetitive/formulaic text', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: 'In today\'s rapidly evolving digital landscape, it is imperative that we leverage innovative solutions to optimize our workflow paradigms. By implementing cutting-edge strategies, we can facilitate synergistic growth and maximize operational efficiency across all verticals. It is crucial to prioritize scalable frameworks that align with our core business objectives and drive sustainable development initiatives forward.'
                })
            });
            assert.equal(res.status, 200);
            const data = await res.json();
            assert.ok(data.overallScore > 30, `Expected score > 30 for AI-like text, got ${data.overallScore}`);
        });

        it('includes all seven metrics in the response', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: 'Test sentence one. Test sentence two.' })
            });
            assert.equal(res.status, 200);
            const data = await res.json();
            const metrics = ['perplexity', 'burstiness', 'vocabulary', 'repetition', 'sentenceStarts', 'formality', 'punctuation'];
            for (const m of metrics) {
                assert.ok(typeof data[m] === 'number', `Missing metric: ${m}`);
                assert.ok(data[m] >= 0 && data[m] <= 100, `${m} out of range: ${data[m]}`);
            }
        });

    });

    it('/api/analyze/text rejects whitespace and tag-only input', async () => {
        skipIfOffline();
        for (const body of ['   ', '<div></div>', '<>']) {
            const res = await fetch(`${BASE}/api/analyze/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: body })
            });
            assert.equal(res.status, 400);
            assert.equal((await res.json()).error, 'Text is required');
        }
    });

    it('/api/analyze/text enforces the length limit', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: 'a '.repeat(8000) })
        });
        assert.equal(res.status, 400);
        assert.match((await res.json()).error, /character limit/);
    });

    it('/api/analyze/image rejects a corrupted PNG (bad magic bytes)', async () => {
        skipIfOffline();
        const form = new FormData();
        form.append('image', new Blob([Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])], { type: 'image/png' }), 'broken.png');
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST', body: form });
        assert.equal(res.status, 400);
        assert.match((await res.json()).error, /invalid or corrupted/i);
    });

    it('/api/analyze/image rejects files over 5MB', async () => {
        skipIfOffline();
        const form = new FormData();
        const big = new Uint8Array(6 * 1024 * 1024);
        big[0] = 0x89; big[1] = 0x50; big[2] = 0x4e; big[3] = 0x47;
        form.append('image', new Blob([big], { type: 'image/png' }), 'big.png');
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST', body: form });
        assert.equal(res.status, 400);
        assert.match((await res.json()).error, /too large/i);
    });

    it('rejects cross-origin API requests with 403', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/detect/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Origin: 'http://evil.example' },
            body: JSON.stringify({ text: 'I went to the store yesterday.' })
        });
        assert.equal(res.status, 403);
    });

    describe('API token auth (API_AUTH_TOKEN set)', () => {
        before(() => { process.env.API_AUTH_TOKEN = 'test-secret'; });
        after(() => { delete process.env.API_AUTH_TOKEN; });

        it('rejects requests without a token', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: 'I went to the store yesterday and bought milk.' })
            });
            assert.equal(res.status, 401);
        });

        it('accepts requests with the correct bearer token', async () => {
            skipIfOffline();
            const res = await fetch(`${BASE}/api/detect/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-secret' },
                body: JSON.stringify({ text: 'I went to the store yesterday and bought some milk.' })
            });
            assert.equal(res.status, 200);
        });
    });
});
