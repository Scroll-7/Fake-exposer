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
