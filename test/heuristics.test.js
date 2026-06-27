import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { detectExtremePhysiqueCasualSetting, detectAiPortrait, detectAiFilename } from '../services/heuristics.js';

describe('detectExtremePhysiqueCasualSetting', () => {
    it('returns null for undefined input', () => {
        assert.equal(detectExtremePhysiqueCasualSetting(null), null);
        assert.equal(detectExtremePhysiqueCasualSetting(undefined), null);
    });

    it('triggers on extreme physique in casual setting', () => {
        const desc = 'A shredded man with visible six-pack taking a mirror selfie in a bathroom';
        assert.ok(detectExtremePhysiqueCasualSetting(desc));
    });

    it('returns null for normal description', () => {
        const desc = 'A fit person jogging in a park on a sunny day';
        assert.equal(detectExtremePhysiqueCasualSetting(desc), null);
    });

    it('returns null for extreme physique in competition setting', () => {
        const desc = 'A bodybuilder on stage with competition trunks under bright lights';
        assert.equal(detectExtremePhysiqueCasualSetting(desc), null);
    });
});

describe('detectAiPortrait', () => {
    it('returns null for undefined input', () => {
        assert.equal(detectAiPortrait(null), null);
    });

    it('triggers on classic AI portrait signature', () => {
        const desc = 'A young woman with smooth skin facing directly at camera against a plain white background studio lighting';
        assert.ok(detectAiPortrait(desc));
    });

    it('returns null when real-world context is present', () => {
        const desc = 'A woman smiling in a park with friends behind her, outdoors natural lighting';
        assert.equal(detectAiPortrait(desc), null);
    });
});

describe('detectAiFilename', () => {
    it('returns null for undefined input', () => {
        assert.equal(detectAiFilename(null), null);
    });

    it('triggers on known AI generator names', () => {
        assert.ok(detectAiFilename('photo_midjourney_v6.png'));
        assert.ok(detectAiFilename('dall-e_output.jpg'));
        assert.ok(detectAiFilename('chatgpt_generated_image.png'));
    });

    it('returns null for normal filenames', () => {
        assert.equal(detectAiFilename('IMG_20240601_123456.jpg'), null);
        assert.equal(detectAiFilename('Screenshot_2024.png'), null);
    });
});
