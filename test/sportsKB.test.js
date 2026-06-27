import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { detectJerseyMismatch } from '../services/sportsKB.js';

describe('detectJerseyMismatch', () => {
    it('returns null for undefined input', () => {
        assert.equal(detectJerseyMismatch(null), null);
        assert.equal(detectJerseyMismatch(undefined), null);
    });

    it('returns null for empty text', () => {
        assert.equal(detectJerseyMismatch(''), null);
    });

    it('returns null when no player is found', () => {
        const result = detectJerseyMismatch('A random person wearing a Barcelona jersey');
        assert.equal(result, null);
    });

    it('detects valid player-club combo (Messi + Barcelona)', () => {
        const result = detectJerseyMismatch('The image shows Lionel Messi wearing a Barcelona jersey');
        assert.equal(result, null);
    });

    it('detects mismatch: Messi in a Real Madrid jersey', () => {
        const result = detectJerseyMismatch('The image shows Lionel Messi wearing a Real Madrid jersey');
        assert.ok(result);
        assert.equal(result.player, 'Lionel Messi');
        assert.equal(result.jerseyTeam, 'Real Madrid');
        assert.ok(result.knownClubs.includes('Barcelona'));
        assert.ok(result.mismatchMsg.includes('NEVER played'));
    });

    it('detects valid: Neymar in PSG jersey', () => {
        const result = detectJerseyMismatch('Neymar wearing a PSG jersey on the pitch');
        assert.equal(result, null);
    });

    it('detects mismatch: Neymar in a Manchester United jersey', () => {
        const result = detectJerseyMismatch('Neymar wearing a Manchester United jersey');
        assert.ok(result);
        assert.equal(result.player, 'Neymar');
        assert.equal(result.jerseyTeam, 'Manchester United');
    });

    it('handles partial name matches correctly', () => {
        const result = detectJerseyMismatch('Ronaldo wearing a Sporting CP jersey');
        assert.equal(result, null);
    });
});
