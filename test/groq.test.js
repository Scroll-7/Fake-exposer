import { describe, it, after, before } from 'node:test';
import assert from 'node:assert/strict';

describe('Groq API key validation', () => {
    const originalKey = process.env.GROQ_API_KEY;

    it('analyzeContent throws descriptive error when GROQ_API_KEY is empty', async () => {
        process.env.GROQ_API_KEY = '';
        const { analyzeContent } = await import('../services/groq.js');
        await assert.rejects(
            () => analyzeContent('test text'),
            { message: /GROQ_API_KEY is not configured/ }
        );
    });

    it('analyzeImage throws descriptive error when GROQ_API_KEY is empty', async () => {
        process.env.GROQ_API_KEY = '';
        const { analyzeImage } = await import('../services/groq.js');
        await assert.rejects(
            () => analyzeImage('dGVzdC1pbWFnZQ==', 'image/png'),
            { message: /GROQ_API_KEY is not configured/ }
        );
    });

    after(() => {
        process.env.GROQ_API_KEY = originalKey;
    });
});

describe('applyLocalMlSignal (ML confidence gate)', () => {
    const emptyResult = () => ({ credibility_score: 90, verdict: 'Likely Credible' });

    it('caps the score to 50 and sets verdict when confident fake', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = emptyResult();
        applyLocalMlSignal(result, { is_fake: true, fake_probability: 0.95, real_probability: 0.05 });
        assert.equal(result.credibility_score, 50);
        assert.match(result.verdict, /Disputed/);
        assert.ok(result.red_flags.some(f => f.includes('95%')));
    });

    it('does not raise a below-50 score but still records the flag', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = { credibility_score: 30, verdict: 'Likely Fake' };
        applyLocalMlSignal(result, { is_fake: true, fake_probability: 0.9, real_probability: 0.1 });
        assert.equal(result.credibility_score, 30);
        assert.equal(result.verdict, 'Likely Fake');
        assert.equal(result.red_flags.length, 1);
    });

    it('abstains (no cap) when fake probability is below the gate', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = emptyResult();
        applyLocalMlSignal(result, { is_fake: true, fake_probability: 0.65, real_probability: 0.35 });
        assert.equal(result.credibility_score, 90);
        assert.equal(result.verdict, 'Likely Credible');
        assert.match(result.red_flags[0], /below the 80% confidence gate/);
    });

    it('adds a green flag when the model predicts real', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = emptyResult();
        applyLocalMlSignal(result, { is_fake: false, fake_probability: 0.1, real_probability: 0.9 });
        assert.equal(result.credibility_score, 90);
        assert.match(result.green_flags[0], /90%/);
        assert.equal(result.red_flags, undefined);
    });

    it('is a no-op for missing/incomplete ML output', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = emptyResult();
        applyLocalMlSignal(result, undefined);
        applyLocalMlSignal(result, { is_fake: true });
        applyLocalMlSignal(null, { is_fake: true, fake_probability: 0.99 });
        assert.equal(result.credibility_score, 90);
        assert.equal(result.red_flags, undefined);
        assert.equal(result.green_flags, undefined);
    });

    it('honours a custom gate value', async () => {
        const { applyLocalMlSignal } = await import('../services/groq.js');
        const result = emptyResult();
        applyLocalMlSignal(result, { is_fake: true, fake_probability: 0.7, real_probability: 0.3 }, 0.6);
        assert.equal(result.credibility_score, 50);
    });
});

describe('nonescapeAdvisory (weak-signal wrapper)', () => {
    it('returns an empty advisory for missing/invalid probabilities', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        for (const v of [undefined, null, NaN]) {
            const a = nonescapeAdvisory(v);
            assert.equal(a.level, null);
            assert.equal(a.note, '');
            assert.equal(a.redFlag, '');
        }
    });

    it('marks >0.7 as high but frames it as a weak signal (no hard verdict)', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.98);
        assert.equal(a.level, 'high');
        assert.match(a.note, /weak, low-confidence signal/);
        assert.match(a.redFlag, /not a hard verdict/);
    });

    it('marks 0.5-0.7 as moderate with no red flag', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.6);
        assert.equal(a.level, 'moderate');
        assert.equal(a.redFlag, '');
        assert.match(a.note, /weak signal/);
    });

    it('marks <0.3 as pass', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.05);
        assert.equal(a.level, 'pass');
        assert.match(a.note, /nothing suspicious/);
        assert.equal(a.redFlag, '');
    });

    it('returns a neutral advisory in the 0.3-0.5 ambiguous band', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.4);
        assert.equal(a.level, null);
        assert.equal(a.note, '');
        assert.equal(a.redFlag, '');
    });
});

describe('extractTrustedSource', () => {
    let extractTrustedSource;
    before(async () => {
        ({ extractTrustedSource } = await import('../services/groq.js'));
    });

    it('returns null for empty or unmatched text', () => {
        assert.equal(extractTrustedSource(null), null);
        assert.equal(extractTrustedSource(''), null);
        assert.equal(extractTrustedSource('just some random sentence'), null);
    });

    it('matches known journalists and outlets case-insensitively', () => {
        assert.equal(extractTrustedSource('According to Fabrizio Romano, here we go!'), 'Fabrizio Romano');
        assert.equal(extractTrustedSource('reuters is reporting this'), 'Reuters');
        assert.equal(extractTrustedSource('espn confirmed the trade'), 'ESPN');
    });
});

describe('detectSportsSuspicion', () => {
    let detectSportsSuspicion;
    before(async () => {
        ({ detectSportsSuspicion } = await import('../services/groq.js'));
    });
    const noTeam = () => null;
    const SPORTS_DESC = 'a player celebrates a goal during the match at the stadium';

    it('returns null when a face override or jersey mismatch is present', () => {
        assert.equal(detectSportsSuspicion('player-1', null, SPORTS_DESC, noTeam), null);
        assert.equal(detectSportsSuspicion(null, { player: 'x' }, SPORTS_DESC, noTeam), null);
    });

    it('returns null without a description', () => {
        assert.equal(detectSportsSuspicion(null, null, '', noTeam), null);
    });

    it('skips selfies and portraits via the Gemini category', () => {
        assert.equal(detectSportsSuspicion(null, null, SPORTS_DESC, noTeam, 'Mirror Selfie'), null);
        assert.equal(detectSportsSuspicion(null, null, SPORTS_DESC, noTeam, 'Portrait'), null);
    });

    it('does not flag descriptions containing anti-sports keywords', () => {
        assert.equal(detectSportsSuspicion(null, null, 'a player stands on the field near the goal during the match', noTeam), null);
    });

    it('requires at least three sports keywords', () => {
        assert.equal(detectSportsSuspicion(null, null, 'a player at the stadium', noTeam), null);
    });

    it('flags an unidentifiable sports image', () => {
        const out = detectSportsSuspicion(null, null, SPORTS_DESC, noTeam);
        assert.match(out, /could not be identified/);
    });

    it('does not flag when a team can be identified', () => {
        assert.equal(detectSportsSuspicion(null, null, SPORTS_DESC, () => 'Barcelona'), null);
    });
});
