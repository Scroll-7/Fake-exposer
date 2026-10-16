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

describe('nonescapeAdvisory (advisory + high-confidence cap)', () => {
    it('returns an empty advisory for missing/invalid probabilities', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        for (const v of [undefined, null, NaN]) {
            const a = nonescapeAdvisory(v);
            assert.equal(a.level, null);
            assert.equal(a.note, '');
            assert.equal(a.redFlag, '');
            assert.equal(a.cap, false);
        }
    });

    it('caps only above 0.9 and calls it high confidence', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const hi = nonescapeAdvisory(0.95);
        assert.equal(hi.level, 'high');
        assert.equal(hi.cap, true);
        assert.match(hi.note, /high confidence/);

        const edge = nonescapeAdvisory(0.9);
        assert.equal(edge.cap, false); // strict inequality
    });

    it('does not cap in the moderate band and warns about domain sensitivity', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.65);
        assert.equal(a.level, 'moderate');
        assert.equal(a.cap, false);
        assert.equal(a.redFlag, '');
        assert.match(a.note, /domain-sensitive/);
    });

    it('marks <0.3 as pass with no cap', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.05);
        assert.equal(a.level, 'pass');
        assert.equal(a.cap, false);
        assert.match(a.note, /nothing suspicious/);
    });

    it('returns a neutral (non-capping) advisory in the 0.3-0.5 band', async () => {
        const { nonescapeAdvisory } = await import('../services/groq.js');
        const a = nonescapeAdvisory(0.4);
        assert.equal(a.level, null);
        assert.equal(a.note, '');
        assert.equal(a.redFlag, '');
        assert.equal(a.cap, false);
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

// ── analyzeImage: mocked network pipeline ────────────────────────────────────
// analyzeImage orchestrates 4 independent network calls (local Face API,
// local Nonescape API, Gemini, and the Groq vision model twice). We mock
// global fetch so the *pure* post-processing — heuristic caps, jersey match,
// Nonescape cap, and the Gemini AI-override priority — can be exercised without
// any keys or running services.
describe('analyzeImage (mocked network pipeline)', () => {
    const originalGroqKey = process.env.GROQ_API_KEY;
    const originalGeminiKeys = process.env.GEMINI_API_KEYS;
    const originalVisionModels = process.env.GROQ_VISION_MODELS;
    const originalFetch = globalThis.fetch;
    let analyzeImage;
    let analyzeContent;
    let resetGroqKeyring;

    const baseAnalysis = () => ({
        credibility_score: 90,
        verdict: 'Highly Credible',
        bias: 'None',
        sentiment: 'Neutral',
        red_flags: [],
        green_flags: ['real world context'],
        summary: 'looks authentic',
        recommendations: [],
        is_trusted_source: false,
        trusted_source_name: '',
    });

    const ok = (obj, status = 200) => ({
        ok: true,
        status,
        json: async () => obj,
        text: async () => JSON.stringify(obj),
    });

    function mockFetch({ description = '', analysis, gemini = null, face = null, nonescape = 0.05 } = {}) {
        const body = analysis || baseAnalysis();
        return async (url, init = {}) => {
            const u = String(url);
            if (u.includes(':8002/recognize')) return ok({ player: face });
            if (u.includes(':8001/detect_ai')) return ok({ ai_probability: nonescape });
            if (u.includes('generativelanguage.googleapis.com')) {
                return ok({ candidates: [{ content: { parts: [{ text: JSON.stringify(gemini) }] } }] });
            }
            if (u.includes('api.groq.com')) {
                const payload = JSON.parse(init.body);
                const content = payload.max_tokens === 250 ? description : JSON.stringify(body);
                return ok({ choices: [{ message: { content } }] });
            }
            throw new Error(`Unexpected fetch in test: ${u}`);
        };
    }

    async function withGeminiKey(fn) {
        const prev = process.env.GEMINI_API_KEYS;
        process.env.GEMINI_API_KEYS = 'test-key';
        try { return await fn(); } finally { process.env.GEMINI_API_KEYS = prev; }
    }

    before(async () => {
        process.env.GROQ_API_KEY = 'test-key';
        process.env.GEMINI_API_KEYS = '';
        process.env.GROQ_VISION_MODELS = 'mock-vision-model';
        ({ analyzeImage, analyzeContent, resetGroqKeyring } = await import('../services/groq.js'));
    });

    after(() => {
        process.env.GROQ_API_KEY = originalGroqKey;
        process.env.GEMINI_API_KEYS = originalGeminiKeys;
        if (originalVisionModels === undefined) delete process.env.GROQ_VISION_MODELS;
        else process.env.GROQ_VISION_MODELS = originalVisionModels;
        globalThis.fetch = originalFetch;
    });

    it('passes the LLM result through untouched when no signal fires', async () => {
        globalThis.fetch = mockFetch();
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 90);
        assert.equal(r.verdict, 'Highly Credible');
    });

    it('applies the portrait heuristic penalty on a studio-portrait description', async () => {
        globalThis.fetch = mockFetch({
            description: 'A professional portrait of a young man facing the camera against a plain grey background. Clear skin, studio lighting, sharp and photorealistic.',
        });
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 50);
        assert.ok(r.red_flags.some(f => /portrait signature/i.test(f)));
    });

    it('hard-caps the score for an extreme physique in a casual setting', async () => {
        globalThis.fetch = mockFetch({
            description: 'A shredded bodybuilder poses with a six-pack in a bathroom mirror.',
        });
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 20);
        assert.match(r.verdict, /Likely AI-Enhanced/);
    });

    it('hard-caps the score when the filename reveals an AI tool', async () => {
        globalThis.fetch = mockFetch();
        const r = await analyzeImage('dGVzdA==', 'image/png', '', 'sunset_midjourney.png');
        assert.equal(r.credibility_score, 5);
        assert.match(r.verdict, /Confirmed AI-Generated/);
    });

    it('hard-caps the score for a known jersey mismatch', async () => {
        globalThis.fetch = mockFetch();
        const r = await analyzeImage('dGVzdA==', 'image/png', 'Lionel Messi wearing a Real Madrid jersey');
        assert.equal(r.credibility_score, 8);
        assert.match(r.verdict, /never played for/);
    });

    it('caps an unverifiable identity (team jersey, no player name)', async () => {
        globalThis.fetch = mockFetch();
        const r = await analyzeImage('dGVzdA==', 'image/png', 'A man wearing a Barcelona jersey.');
        assert.equal(r.credibility_score, 15);
        assert.match(r.verdict, /Unverifiable/);
    });

    it('caps an unidentifiable sports image', async () => {
        globalThis.fetch = mockFetch({
            description: 'A football player celebrates a goal at the stadium during the match.',
        });
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 55);
        assert.match(r.verdict, /Unverifiable — Sports/);
    });

    it('caps the score when the Nonescape model is >= 0.9 confident', async () => {
        globalThis.fetch = mockFetch({ nonescape: 0.95 });
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 30);
        assert.match(r.verdict, /ML Detector/);
        assert.ok(r.red_flags.some(f => /95% AI probability/.test(f)));
    });

    it('does not cap on a moderate Nonescape probability', async () => {
        globalThis.fetch = mockFetch({ nonescape: 0.65 });
        const r = await analyzeImage('dGVzdA==', 'image/png');
        assert.equal(r.credibility_score, 90);
    });

    it('lets a Gemini hard AI override take priority over every other cap', async () => {
        await withGeminiKey(async () => {
            const analysis = { ...baseAnalysis(), is_trusted_source: true, trusted_source_name: 'Reuters' };
            globalThis.fetch = mockFetch({
                analysis,
                gemini: { image_category: 'Other', ai_probability: 90, verdict: 'AI-Generated', confidence: 'High', artifacts_detected: ['x'], reasoning: 'r', description: '' },
            });
            const r = await analyzeImage('dGVzdA==', 'image/png', 'Lionel Messi wearing a Real Madrid jersey');
            assert.equal(r.credibility_score, 10);
            assert.match(r.verdict, /AI-Generated \/ Fake/);
            assert.equal(r.is_trusted_source, false);
            assert.equal(r.trusted_source_name, null);
        });
    });

    it('softens the score but keeps the verdict for a mid-range Gemini signal', async () => {
        await withGeminiKey(async () => {
            globalThis.fetch = mockFetch({
                gemini: { image_category: 'Other', ai_probability: 30, verdict: 'Suspicious', confidence: 'High', artifacts_detected: [], reasoning: '', description: '' },
            });
            const r = await analyzeImage('dGVzdA==', 'image/png');
            assert.equal(r.credibility_score, 72);
            assert.equal(r.verdict, 'Highly Credible');
            assert.ok(r.red_flags.some(f => /Gemini AI Detector/.test(f)));
        });
    });

    it('fails over to the next Groq key when the vision key is denied (403)', async () => {
        const prevKeys = process.env.GROQ_API_KEYS;
        process.env.GROQ_API_KEYS = 'gk1,gk2';
        resetGroqKeyring();
        const seenAuth = [];
        globalThis.fetch = (url, init = {}) => {
            const u = String(url);
            if (u.includes(':8002/recognize')) return Promise.resolve(ok({ player: null }));
            if (u.includes(':8001/detect_ai')) return Promise.resolve(ok({ ai_probability: 0.05 }));
            if (u.includes('api.groq.com')) {
                const auth = init.headers?.Authorization;
                seenAuth.push(auth);
                if (auth === 'Bearer gk1') {
                    return Promise.resolve({ ok: false, status: 403, text: async () => 'denied', json: async () => ({}) });
                }
                const payload = JSON.parse(init.body);
                const content = payload.max_tokens === 250 ? '' : JSON.stringify(baseAnalysis());
                return Promise.resolve(ok({ choices: [{ message: { content } }] }));
            }
            throw new Error(`Unexpected fetch in test: ${u}`);
        };
        try {
            const r = await analyzeImage('dGVzdA==', 'image/png');
            assert.equal(r.credibility_score, 90);
            assert.equal(seenAuth[0], 'Bearer gk1');
            assert.equal(seenAuth[1], 'Bearer gk2');
        } finally {
            if (prevKeys === undefined) delete process.env.GROQ_API_KEYS;
            else process.env.GROQ_API_KEYS = prevKeys;
            resetGroqKeyring();
        }
    });

    it('fails over to the next Groq key when the text key is denied (403)', async () => {
        const prevKeys = process.env.GROQ_API_KEYS;
        const prevTextModels = process.env.GROQ_TEXT_MODELS;
        process.env.GROQ_API_KEYS = 'tk1,tk2';
        process.env.GROQ_TEXT_MODELS = 'mock-text-model';
        resetGroqKeyring();
        const seenAuth = [];
        const ddgHtml = '<td class="result__snippet">The World Health Organization declared COVID-19 a global pandemic on March 11, 2020, a widely reported public-health milestone.</td>';
        globalThis.fetch = (url, init = {}) => {
            const u = String(url);
            if (u.includes('duckduckgo.com')) return Promise.resolve({ ok: true, status: 200, text: async () => ddgHtml });
            if (u.includes('api.groq.com')) {
                const auth = init.headers?.Authorization;
                seenAuth.push(auth);
                if (auth === 'Bearer tk1') {
                    return Promise.resolve({ ok: false, status: 403, text: async () => 'denied', json: async () => ({}) });
                }
                return Promise.resolve(ok({ choices: [{ message: { content: JSON.stringify(baseAnalysis()) } }] }));
            }
            throw new Error(`Unexpected fetch in test: ${u}`);
        };
        try {
            const r = await analyzeContent('The World Health Organization declared COVID-19 a global pandemic on March 11 2020 according to official public health records and news reports from around the world that day.');
            assert.equal(seenAuth[0], 'Bearer tk1');
            assert.equal(seenAuth[1], 'Bearer tk2');
            assert.equal(typeof r.credibility_score, 'number');
        } finally {
            if (prevKeys === undefined) delete process.env.GROQ_API_KEYS;
            else process.env.GROQ_API_KEYS = prevKeys;
            if (prevTextModels === undefined) delete process.env.GROQ_TEXT_MODELS;
            else process.env.GROQ_TEXT_MODELS = prevTextModels;
            resetGroqKeyring();
        }
    });
});
