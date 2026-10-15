import { describe, it, after } from 'node:test';
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
