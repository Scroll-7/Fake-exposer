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

    it('detects valid: de Jong in Barcelona jersey', () => {
        const result = detectJerseyMismatch('Frenkie de Jong wearing FC Barcelona jersey');
        assert.equal(result, null);
    });

    it('detects mismatch: de Jong in Real Madrid jersey', () => {
        const result = detectJerseyMismatch('Frenkie de Jong playing for Real Madrid');
        assert.ok(result);
        assert.equal(result.player, 'Frenkie De Jong');
        assert.equal(result.jerseyTeam, 'Real Madrid');
    });

    it('detects mismatch: Olise in Barcelona jersey', () => {
        const result = detectJerseyMismatch('Michael Olise posing in a Barcelona kit');
        assert.ok(result);
        assert.equal(result.player, 'Michael Olise');
        assert.equal(result.jerseyTeam, 'FC Barcelona');
        assert.ok(result.mismatchMsg.includes('NEVER played for FC Barcelona'));
    });

    it('detects mismatch: Olise alias in Barcelona jersey', () => {
        const result = detectJerseyMismatch('Olise wearing Barcelona jersey');
        assert.ok(result);
        assert.equal(result.player, 'Olise');
        assert.equal(result.jerseyTeam, 'FC Barcelona');
    });
});
