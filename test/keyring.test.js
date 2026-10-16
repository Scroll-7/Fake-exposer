import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createKeyRing } from '../services/keyring.js';

describe('createKeyRing', () => {
    it('parses, trims and drops empty keys', () => {
        const ring = createKeyRing(' a , b ,, c ,');
        assert.equal(ring.size, 3);
    });

    it('returns null for an empty or missing value', () => {
        assert.equal(createKeyRing('').current(), null);
        assert.equal(createKeyRing(undefined).current(), null);
        assert.equal(createKeyRing('  ,  ').current(), null);
    });

    it('stays on the current key until it is marked bad (failover-only)', () => {
        const ring = createKeyRing('k1,k2,k3');
        assert.equal(ring.current(), 'k1');
        assert.equal(ring.current(), 'k1');
        assert.equal(ring.current(), 'k1');
    });

    it('fails over to the next key after markBad', () => {
        const ring = createKeyRing('k1,k2,k3');
        assert.equal(ring.current(), 'k1');
        ring.markBad('k1');
        assert.equal(ring.rotate(), 'k2');
        assert.equal(ring.current(), 'k2');
    });

    it('skips keys that are still cooling down', () => {
        const ring = createKeyRing('k1,k2,k3');
        ring.markBad('k2');
        assert.equal(ring.current(), 'k1');
        ring.markBad('k1');
        assert.equal(ring.rotate(), 'k3');
    });

    it('returns null when every key is cooling down', () => {
        const ring = createKeyRing('k1,k2');
        ring.markBad('k1');
        ring.markBad('k2');
        assert.equal(ring.current(), null);
        assert.equal(ring.rotate(), null);
    });

    it('makes a key usable again once its cooldown expires', () => {
        const ring = createKeyRing('k1,k2');
        ring.markBad('k1', 0);
        assert.equal(ring.current(), 'k1');
    });

    it('wraps around when rotating past the last key', () => {
        const ring = createKeyRing('k1,k2');
        assert.equal(ring.rotate(), 'k2');
        assert.equal(ring.rotate(), 'k1');
    });

    it('honours a per-call cooldown override', () => {
        const ring = createKeyRing('k1', { defaultCooldownMs: 60_000 });
        ring.markBad('k1', 0);
        assert.equal(ring.current(), 'k1');
    });

    it('reset clears cooldowns and returns to the first key', () => {
        const ring = createKeyRing('k1,k2');
        ring.markBad('k1');
        assert.equal(ring.current(), 'k2');
        ring.reset();
        assert.equal(ring.current(), 'k1');
    });
});
