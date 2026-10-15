import { describe, it, mock, after } from 'node:test';
import assert from 'node:assert/strict';
import { computeAiOverride, detectAiImage } from '../services/aiDetector.js';

describe('computeAiOverride', () => {
    it('returns null for no detection', () => {
        assert.equal(computeAiOverride(null), null);
        assert.equal(computeAiOverride(undefined), null);
    });

    describe('TIER 1 — High/Medium confidence', () => {
        it('hard-overrides definite AI (>=80%)', () => {
            const o = computeAiOverride({
                aiProbability: 90,
                confidence: 'High',
                artifacts: ['halo'],
                reasoning: 'rendered skin',
            });
            assert.equal(o.credibility_score, 10); // max(2, 100-90)
            assert.match(o.verdict, /AI-Generated \/ Fake/);
            assert.equal(o.softWarning, false);
            assert.deepEqual(o.ai_artifacts, ['halo']);
            assert.equal(o.ai_reasoning, 'rendered skin');
        });

        it('clamps definite-AI score to a floor of 2', () => {
            const o = computeAiOverride({ aiProbability: 100, confidence: 'High' });
            assert.equal(o.credibility_score, 2);
        });

        it('overrides likely AI (45-79%) at half weight', () => {
            const o = computeAiOverride({ aiProbability: 60, confidence: 'Medium' });
            assert.equal(o.credibility_score, 20); // round((100-60)*0.5)
            assert.match(o.verdict, /Likely AI-Generated/);
            assert.equal(o.softWarning, false);
        });

        it('soft-warns for suspicious (25-44%) without overriding', () => {
            const o = computeAiOverride({ aiProbability: 30, confidence: 'High' });
            assert.equal(o.credibility_score, null);
            assert.equal(o.verdict, null);
            assert.equal(o.softWarning, true);
        });

        it('returns null below the suspicion floor (<25%)', () => {
            assert.equal(computeAiOverride({ aiProbability: 10, confidence: 'High' }), null);
            assert.equal(computeAiOverride({ aiProbability: 24, confidence: 'Medium' }), null);
        });

        it('treats the 80% boundary as definite and 45% as likely', () => {
            assert.equal(computeAiOverride({ aiProbability: 80, confidence: 'High' }).softWarning, false);
            assert.match(computeAiOverride({ aiProbability: 80, confidence: 'High' }).verdict, /Fake/);
            assert.match(computeAiOverride({ aiProbability: 45, confidence: 'High' }).verdict, /Likely/);
        });
    });

    describe('TIER 2 — Low confidence', () => {
        it('overrides >=80% at 1.5x with a floor of 10', () => {
            const o = computeAiOverride({ aiProbability: 90, confidence: 'Low' });
            assert.equal(o.credibility_score, 15); // max(10, round(10*1.5))
            assert.match(o.verdict, /Possibly AI-Generated/);
            assert.equal(o.softWarning, false);
        });

        it('soft-warns for 50-79%', () => {
            const o = computeAiOverride({ aiProbability: 60, confidence: 'Low' });
            assert.equal(o.credibility_score, null);
            assert.equal(o.softWarning, true);
        });

        it('returns null below 50%', () => {
            assert.equal(computeAiOverride({ aiProbability: 49, confidence: 'Low' }), null);
        });
    });
});

describe('detectAiImage', () => {
    const origKeys = process.env.GEMINI_API_KEYS;

    after(() => {
        if (origKeys === undefined) delete process.env.GEMINI_API_KEYS;
        else process.env.GEMINI_API_KEYS = origKeys;
        mock.reset();
    });

    function geminiPayload(obj) {
        return { candidates: [{ content: { parts: [{ text: JSON.stringify(obj) }] } }] };
    }

    function okResponse(payload) {
        return {
            status: 200,
            ok: true,
            text: () => Promise.resolve(''),
            json: () => Promise.resolve(payload),
        };
    }

    const validBody = {
        image_category: 'Mirror Selfie',
        ai_probability: 90,
        verdict: 'AI-Generated',
        confidence: 'High',
        artifacts_detected: ['halo'],
        reasoning: 'rendered skin',
        description: 'a man in a mirror',
    };

    it('returns null without calling the network when no keys are set', async () => {
        delete process.env.GEMINI_API_KEYS;
        const spy = mock.method(global, 'fetch', () => { throw new Error('should not be called'); });
        assert.equal(await detectAiImage('b64', 'image/png'), null);
        assert.equal(spy.mock.callCount(), 0);
        mock.reset();
    });

    it('parses a successful Gemini JSON response', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(okResponse(geminiPayload(validBody))));
        const r = await detectAiImage('b64', 'image/png');
        assert.equal(r.aiProbability, 90);
        assert.equal(r.isAI, true);
        assert.equal(r.verdict, 'AI-Generated');
        assert.equal(r.imageCategory, 'Mirror Selfie');
        assert.equal(r.confidence, 'High');
        assert.deepEqual(r.artifacts, ['halo']);
        assert.equal(r.description, 'a man in a mirror');
        mock.reset();
    });

    it('flags isAI at the 45% boundary', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(
            okResponse(geminiPayload({ ...validBody, ai_probability: 45 }))
        ));
        assert.equal((await detectAiImage('b', 'image/png')).isAI, true);
        mock.reset();

        mock.method(global, 'fetch', () => Promise.resolve(
            okResponse(geminiPayload({ ...validBody, ai_probability: 44 }))
        ));
        assert.equal((await detectAiImage('b', 'image/png')).isAI, false);
        mock.reset();
    });

    it('defaults optional fields when absent', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(
            okResponse(geminiPayload({ ai_probability: 70, verdict: 'Likely', confidence: 'Medium' }))
        ));
        const r = await detectAiImage('b', 'image/png');
        assert.equal(r.imageCategory, 'Other');
        assert.deepEqual(r.artifacts, []);
        assert.equal(r.reasoning, '');
        mock.reset();
    });

    it('advances to the next model when one returns 503', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        let calls = 0;
        mock.method(global, 'fetch', () => {
            calls++;
            if (calls === 1) {
                return Promise.resolve({ status: 503, ok: false, text: () => Promise.resolve('busy') });
            }
            return Promise.resolve(okResponse(geminiPayload(validBody)));
        });
        const r = await detectAiImage('b', 'image/png');
        assert.equal(r.aiProbability, 90);
        assert.equal(calls, 2);
        mock.reset();
    });

    it('returns null when every model is unavailable', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(
            { status: 503, ok: false, text: () => Promise.resolve('busy') }
        ));
        assert.equal(await detectAiImage('b', 'image/png'), null);
        mock.reset();
    });

    it('returns null on a non-ok response', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(
            { status: 500, ok: false, text: () => Promise.resolve('server error') }
        ));
        assert.equal(await detectAiImage('b', 'image/png'), null);
        mock.reset();
    });

    it('returns null when the model output is not valid JSON', async () => {
        process.env.GEMINI_API_KEYS = 'k1';
        mock.method(global, 'fetch', () => Promise.resolve(
            { status: 200, ok: true, text: () => Promise.resolve(''), json: () => Promise.resolve({ candidates: [{ content: { parts: [{ text: 'not json' }] } }] }) }
        ));
        assert.equal(await detectAiImage('b', 'image/png'), null);
        mock.reset();
    });
});
