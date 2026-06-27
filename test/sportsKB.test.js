import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { detectJerseyMismatch, getCelebrityContext, CELEBRITY_CONTEXTS } from '../services/sportsKB.js';
import { detectSportsSuspicion } from '../services/groq.js';
import { findTeamInText } from '../services/sportsKB.js';

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

describe('detectSportsSuspicion', () => {

    it('returns null when Face API is available', () => {
        const result = detectSportsSuspicion('lionel messi', null, 'a man in a jersey on a football pitch');
        assert.equal(result, null);
    });

    it('returns null when jersey mismatch was detected', () => {
        const result = detectSportsSuspicion(null, { player: 'Messi', jerseyTeam: 'Real Madrid' }, 'Messi in a Real Madrid jersey');
        assert.equal(result, null);
    });

    it('returns null for non-sports description', () => {
        const result = detectSportsSuspicion(null, null, 'a cat sitting on a windowsill in the sunlight');
        assert.equal(result, null);
    });

    it('returns null when a team is identified in the description', () => {
        const result = detectSportsSuspicion(null, null, 'a player wearing a Barcelona jersey on the pitch');
        assert.equal(result, null);
    });

    it('returns warning for sports context with no team and no face ID', () => {
        const result = detectSportsSuspicion(null, null, 'a man in a jersey standing on a field');
        assert.ok(result);
        assert.ok(result.includes('Sports image detected'));
        assert.ok(result.includes('player or team could not be identified'));
    });

    it('returns warning for football-specific description without team', () => {
        const result = detectSportsSuspicion(null, null, 'football player on the pitch wearing a white and black kit');
        assert.ok(result);
        assert.ok(result.includes('Sports image detected'));
    });

    it('handles empty description gracefully', () => {
        const result = detectSportsSuspicion(null, null, '');
        assert.equal(result, null);
    });

    it('handles null description gracefully', () => {
        const result = detectSportsSuspicion(null, null, null);
        assert.equal(result, null);
    });

    it('uses custom findTeamFn when provided', () => {
        const customFind = (text) => text.includes('custom_team') ? 'Custom Team' : null;
        const result1 = detectSportsSuspicion(null, null, 'jersey custom_team', customFind);
        assert.equal(result1, null);
        const result2 = detectSportsSuspicion(null, null, 'jersey only', customFind);
        assert.ok(result2);
    });

});

describe('getCelebrityContext', () => {
    it('returns null for undefined input', () => {
        assert.equal(getCelebrityContext(null), null);
        assert.equal(getCelebrityContext(undefined), null);
    });

    it('returns null for empty string', () => {
        assert.equal(getCelebrityContext(''), null);
    });

    it('returns null for unknown celebrity', () => {
        assert.equal(getCelebrityContext('unknown person'), null);
        assert.equal(getCelebrityContext('lionel messi'), null);
    });

    it('returns context for Elon Musk', () => {
        const ctx = getCelebrityContext('elon musk');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Elon Musk');
        assert.ok(ctx.roles.includes('CEO'));
        assert.ok(ctx.organizations.includes('Tesla'));
        assert.ok(ctx.typicalSettings.includes('tech conference'));
        assert.equal(ctx.sport, null);
    });

    it('returns context for Taylor Swift', () => {
        const ctx = getCelebrityContext('taylor swift');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Taylor Swift');
        assert.ok(ctx.roles.includes('singer'));
        assert.ok(ctx.typicalSettings.includes('concert'));
    });

    it('returns context for Donald Trump', () => {
        const ctx = getCelebrityContext('donald trump');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Donald Trump');
        assert.ok(ctx.roles.includes('president'));
        assert.ok(ctx.organizations.includes('White House'));
    });

    it('returns context for The Rock', () => {
        const ctx = getCelebrityContext('the rock');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Dwayne "The Rock" Johnson');
        assert.ok(ctx.roles.includes('wrestler'));
        assert.ok(ctx.organizations.includes('WWE'));
        assert.equal(ctx.sport, 'wrestling');
    });

    it('is case-insensitive', () => {
        const ctx1 = getCelebrityContext('Elon Musk');
        const ctx2 = getCelebrityContext('ELON MUSK');
        assert.ok(ctx1);
        assert.ok(ctx2);
        assert.equal(ctx1.display, ctx2.display);
    });

    it('has all CELEBRITY_CONTEXTS keys match face_api.py naming convention', () => {
        for (const key of Object.keys(CELEBRITY_CONTEXTS)) {
            assert.ok(key === key.toLowerCase(), `Key "${key}" must be lowercase`);
            assert.ok(!key.includes('_'), `Key "${key}" must use spaces, not underscores`);
        }
    });

    it('returns party and opponents for Donald Trump', () => {
        const ctx = getCelebrityContext('donald trump');
        assert.equal(ctx.party, 'Republican');
        assert.ok(ctx.opponents.includes('Joe Biden'));
        assert.ok(ctx.opponents.includes('Democratic Party'));
    });

    it('returns party and opponents for Joe Biden', () => {
        const ctx = getCelebrityContext('joe biden');
        assert.equal(ctx.party, 'Democratic');
        assert.ok(ctx.opponents.includes('Donald Trump'));
        assert.ok(ctx.opponents.includes('Republican Party'));
    });

    it('returns party and opponents for Barack Obama', () => {
        const ctx = getCelebrityContext('barack obama');
        assert.equal(ctx.party, 'Democratic');
        assert.ok(ctx.opponents.includes('John McCain'));
        assert.ok(ctx.opponents.includes('Mitt Romney'));
    });

    it('returns party and opponents for Vladimir Putin', () => {
        const ctx = getCelebrityContext('putin');
        assert.equal(ctx.party, 'United Russia');
        assert.ok(ctx.opponents.includes('NATO'));
        assert.ok(ctx.opponents.includes('Ukraine'));
    });

    it('returns party and opponents for Xi Jinping', () => {
        const ctx = getCelebrityContext('xi jinping');
        assert.equal(ctx.party, 'Chinese Communist Party');
        assert.ok(ctx.opponents.includes('Taiwan'));
        assert.ok(ctx.opponents.includes('United States'));
    });

    it('returns party and opponents for Kamala Harris', () => {
        const ctx = getCelebrityContext('kamala harris');
        assert.equal(ctx.party, 'Democratic');
        assert.ok(ctx.opponents.includes('Donald Trump'));
        assert.ok(ctx.opponents.includes('Republican Party'));
    });

    it('returns null party/opponents for non-politician (Pope Francis)', () => {
        const ctx = getCelebrityContext('pope francis');
        assert.equal(ctx.party, null);
        assert.deepEqual(ctx.opponents, []);
    });

    it('returns null party/opponents for non-politician (Kim Kardashian)', () => {
        const ctx = getCelebrityContext('kim kardashian');
        assert.equal(ctx.party, null);
        assert.deepEqual(ctx.opponents, []);
    });

    it('returns context for all 20 celebrity entries', () => {
        const keys = Object.keys(CELEBRITY_CONTEXTS);
        assert.equal(keys.length, 20);
        for (const key of keys) {
            const ctx = getCelebrityContext(key);
            assert.ok(ctx, `Missing context for "${key}"`);
            assert.ok(ctx.display, `Missing display for "${key}"`);
            assert.ok(ctx.roles.length > 0, `Missing roles for "${key}"`);
            assert.ok(ctx.typicalSettings.length > 0, `Missing typicalSettings for "${key}"`);
            assert.ok('party' in ctx, `Missing party for "${key}"`);
            assert.ok('opponents' in ctx, `Missing opponents for "${key}"`);
        }
    });

    it('produces a note string that includes party for politicians', () => {
        const ctx = getCelebrityContext('donald trump');
        const note = [
            `\n[CELEBRITY PROFILE] ${ctx.display} is known as a ${ctx.roles.join(', ')}. `,
            ctx.organizations.length ? `Associated with: ${ctx.organizations.join(', ')}. ` : '',
            `Typical settings: ${ctx.typicalSettings.join(', ')}. `,
            ctx.party ? `Political party: ${ctx.party}. ` : '',
            ctx.opponents?.length ? `Political opponents: ${ctx.opponents.join(', ')}. ` : '',
        ].join('');
        assert.ok(note.includes('Political party: Republican'));
        assert.ok(note.includes('Political opponents: Democratic Party'));
    });

    it('returns context for LeBron James', () => {
        const ctx = getCelebrityContext('lebron james');
        assert.ok(ctx);
        assert.equal(ctx.display, 'LeBron James');
        assert.ok(ctx.roles.includes('basketball player'));
        assert.equal(ctx.sport, 'basketball');
        assert.equal(ctx.party, null);
    });

    it('returns context for Serena Williams', () => {
        const ctx = getCelebrityContext('serena williams');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Serena Williams');
        assert.ok(ctx.roles.includes('tennis player'));
        assert.equal(ctx.sport, 'tennis');
    });

    it('returns context for Roger Federer', () => {
        const ctx = getCelebrityContext('roger federer');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Roger Federer');
        assert.ok(ctx.roles.includes('tennis player'));
        assert.equal(ctx.sport, 'tennis');
    });

    it('returns context for Leonardo DiCaprio', () => {
        const ctx = getCelebrityContext('leonardo dicaprio');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Leonardo DiCaprio');
        assert.ok(ctx.roles.includes('actor'));
    });

    it('returns context for Keanu Reeves', () => {
        const ctx = getCelebrityContext('keanu reeves');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Keanu Reeves');
        assert.ok(ctx.roles.includes('actor'));
    });

    it('returns context for Oprah Winfrey', () => {
        const ctx = getCelebrityContext('oprah winfrey');
        assert.ok(ctx);
        assert.equal(ctx.display, 'Oprah Winfrey');
        assert.ok(ctx.roles.includes('talk show host'));
    });
});
