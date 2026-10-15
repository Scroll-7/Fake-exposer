import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeAiOverride } from '../services/aiDetector.js';

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
