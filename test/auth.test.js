import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createApiAuth, extractToken, isSameOrigin, timingSafeEqual } from '../services/auth.js';

function mockRes() {
    return {
        statusCode: null,
        body: null,
        headers: {},
        status(code) { this.statusCode = code; return this; },
        json(payload) { this.body = payload; return this; },
        set(key, value) { this.headers[key] = value; return this; },
    };
}

describe('auth', () => {
    it('extractToken reads Bearer and X-API-Key headers', () => {
        assert.equal(extractToken({ headers: { authorization: 'Bearer abc123' } }), 'abc123');
        assert.equal(extractToken({ headers: { authorization: 'bearer abc123' } }), 'abc123');
        assert.equal(extractToken({ headers: { 'x-api-key': ' key-xyz ' } }), 'key-xyz');
        assert.equal(extractToken({ headers: {} }), null);
        assert.equal(extractToken({ headers: { authorization: 'Basic zzz' } }), null);
    });

    it('timingSafeEqual compares without throwing on length mismatch', () => {
        assert.equal(timingSafeEqual('token', 'token'), true);
        assert.equal(timingSafeEqual('token', 'toke'), false);
        assert.equal(timingSafeEqual('token', 'other'), false);
    });

    it('isSameOrigin allows missing Origin and matching hosts', () => {
        assert.equal(isSameOrigin({ headers: { host: 'localhost:3001' } }), true);
        assert.equal(isSameOrigin({ headers: { host: 'localhost:3001', origin: 'http://localhost:3001' } }), true);
        assert.equal(isSameOrigin({ headers: { host: 'localhost:3001', origin: 'http://evil.example' } }), false);
        assert.equal(isSameOrigin({ headers: { host: 'localhost:3001', origin: 'not a url' } }), false);
    });

    it('passes through when no token is configured', () => {
        const mw = createApiAuth(() => undefined);
        let called = false;
        mw({ headers: { host: 'x' } }, mockRes(), () => { called = true; });
        assert.equal(called, true);
    });

    it('rejects a missing or wrong token when configured', () => {
        const mw = createApiAuth(() => 'secret');
        const noToken = mockRes();
        mw({ headers: { host: 'x' } }, noToken, () => { throw new Error('should not pass'); });
        assert.equal(noToken.statusCode, 401);
        assert.equal(noToken.headers['WWW-Authenticate'], 'Bearer');

        const wrong = mockRes();
        mw({ headers: { host: 'x', authorization: 'Bearer nope' } }, wrong, () => { throw new Error('should not pass'); });
        assert.equal(wrong.statusCode, 401);
    });

    it('accepts a matching token from either header', () => {
        const mw = createApiAuth(() => 'secret');
        let ok = 0;
        mw({ headers: { host: 'x', authorization: 'Bearer secret' } }, mockRes(), () => { ok++; });
        mw({ headers: { host: 'x', 'x-api-key': 'secret' } }, mockRes(), () => { ok++; });
        assert.equal(ok, 2);
    });

    it('rejects cross-origin requests before checking the token', () => {
        const mw = createApiAuth(() => 'secret');
        const res = mockRes();
        mw({ headers: { host: 'localhost:3001', origin: 'http://evil.example', authorization: 'Bearer secret' } }, res, () => {
            throw new Error('should not pass');
        });
        assert.equal(res.statusCode, 403);
    });
});
