import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeText } from '../services/textDetector.js';

const AI_TEXT = 'Furthermore, the comprehensive framework leverages innovative methodologies to facilitate seamless integration across the ecosystem. Moreover, stakeholders must utilize scalable solutions to optimize outcomes and prioritize robust implementation. Consequently, the transformative paradigm demonstrates significant enhancements that ultimately underscore the multifaceted nature of the initiative. Additionally, this holistic approach fosters synergy and navigates the intricate complexities of the evolving landscape.';

const CASUAL_TEXT = 'ok so honestly i was kinda tired today and the whole thing felt pretty weird lol. my buddy said hey dude wanna grab some food but i wasnt really hungry at all. we ended up just chilling and talking about random stuff for a while. anyway it was a good time i guess and thats pretty much it you know.';

const ZERO_KEYS = ['overallScore', 'perplexity', 'burstiness', 'vocabulary', 'repetition', 'sentenceStarts'];

describe('analyzeText', () => {
    it('returns all-zero results for empty / non-string / whitespace input', () => {
        for (const input of ['', '   ', null, undefined, 42]) {
            const r = analyzeText(input);
            for (const k of ZERO_KEYS) assert.equal(r[k], 0, `${k} for ${JSON.stringify(input)}`);
            assert.deepEqual(r.sentences, []);
        }
    });

    it('returns all-zero results for fewer than 3 words', () => {
        const r = analyzeText('hi there');
        for (const k of ZERO_KEYS) assert.equal(r[k], 0);
        assert.deepEqual(r.sentences, []);
    });

    it('produces a normalised score and sentence breakdown', () => {
        const r = analyzeText(AI_TEXT);
        assert.ok(r.overallScore > 0 && r.overallScore <= 100);
        assert.ok(Array.isArray(r.sentences) && r.sentences.length > 0);
        for (const s of r.sentences) {
            assert.equal(typeof s.sentence, 'string');
            assert.equal(typeof s.score, 'number');
            assert.equal(typeof s.isAi, 'boolean');
        }
        assert.ok(r.aiSentencePercentage >= 0 && r.aiSentencePercentage <= 100);
    });

    it('scores AI-formal prose higher than casual prose', () => {
        const ai = analyzeText(AI_TEXT);
        const casual = analyzeText(CASUAL_TEXT);
        assert.ok(
            ai.overallScore > casual.overallScore,
            `expected AI ${ai.overallScore} > casual ${casual.overallScore}`
        );
        assert.ok(ai.aiSentencePercentage > casual.aiSentencePercentage);
    });

    it('gives neutral 50 to very short sentences', () => {
        const r = analyzeText('Hi there. This is a much longer sentence that should be analyzed properly by the detector.');
        assert.equal(r.sentences[0].score, 50); // "Hi there." is < 10 chars
    });

    it('gives neutral 50 to sentences with fewer than 3 tokens', () => {
        const r = analyzeText('Hello everyone. Furthermore, this longer sentence contains enough tokens to be scored properly.');
        assert.equal(r.sentences[0].score, 50); // "Hello everyone." -> 2 tokens
    });

    it('handles a range of text lengths without throwing', () => {
        const short = analyzeText('one two three four five six seven eight nine');
        const medium = analyzeText('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi');
        const long = analyzeText(AI_TEXT);
        for (const r of [short, medium, long]) {
            assert.equal(typeof r.overallScore, 'number');
            assert.ok(r.overallScore >= 0 && r.overallScore <= 100);
        }
    });
});
