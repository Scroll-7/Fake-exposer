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

after(() => {
    if (server) {
        server.close();
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

    it('returns 404 for unknown routes', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/nonexistent`);
        assert.equal(res.status, 404);
    });

    it('returns JSON error for invalid route', async () => {
        skipIfOffline();
        const res = await fetch(`${BASE}/api/analyze/nonexistent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 404);
    });
});
