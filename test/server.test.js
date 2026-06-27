import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';

const BASE = 'http://127.0.0.1:3001';
let serverOnline = false;

before(async () => {
    try {
        const res = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(2000) });
        serverOnline = res.ok;
    } catch {
        serverOnline = false;
    }
});

describe('Server API Endpoints', () => {
    const skipIfOffline = serverOnline ? false : 'Server not running — start with `npm start` first';

    it('serves the frontend HTML', { skip: skipIfOffline }, async () => {
        const res = await fetch(`${BASE}/`);
        assert.equal(res.status, 200);
        const text = await res.text();
        assert.ok(text.includes('<!DOCTYPE html>'));
    });

    it('/api/analyze/text returns 400 for empty body', { skip: skipIfOffline }, async () => {
        const res = await fetch(`${BASE}/api/analyze/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'Text is required');
    });

    it('/api/analyze/url returns 400 for empty body', { skip: skipIfOffline }, async () => {
        const res = await fetch(`${BASE}/api/analyze/url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'URL is required');
    });

    it('/api/analyze/image returns 400 without file', { skip: skipIfOffline }, async () => {
        const res = await fetch(`${BASE}/api/analyze/image`, { method: 'POST' });
        assert.equal(res.status, 400);
        const data = await res.json();
        assert.equal(data.error, 'Image is required');
    });
});
