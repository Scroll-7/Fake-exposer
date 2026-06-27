import { describe, it, mock } from 'node:test';
import assert from 'node:assert/strict';
import { searchWeb, scrapeUrl } from '../services/scraper.js';

describe('searchWeb', () => {
    it('returns fallback message when fetch fails', async () => {
        mock.method(global, 'fetch', () => Promise.reject(new Error('Network error')));
        const result = await searchWeb('test query');
        assert.equal(result, 'No recent news found.');
        mock.reset();
    });

    it('returns fallback on empty HTML', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            text: () => Promise.resolve('<html></html>'),
        }));
        const result = await searchWeb('test query');
        assert.equal(result, 'No recent news found.');
        mock.reset();
    });

    it('extracts snippets from DuckDuckGo HTML', async () => {
        // Realistic DuckDuckGo HTML snippet with proper length
        const html = `<!DOCTYPE html><html><body><div class="results"><div class="result"><a class="result__snippet" href="http://example.com">This is a test snippet about fake news detection using AI and machine learning methods</a></div></div></body></html>`;
        mock.method(global, 'fetch', () => Promise.resolve({
            text: () => Promise.resolve(html),
        }));
        const result = await searchWeb('test query');
        assert.ok(result.includes('This is a test snippet'));
        mock.reset();
    });

    it('falls back to lite endpoint when html endpoint fails', async () => {
        let callCount = 0;
        mock.method(global, 'fetch', () => {
            callCount++;
            if (callCount === 1) return Promise.reject(new Error('html failed'));
            return Promise.resolve({
                text: () => Promise.resolve('<html><a class="result__snippet" href="http://example.com">Lite fallback snippet about fake news detection technologies</a></html>'),
            });
        });
        const result = await searchWeb('test query');
        assert.equal(callCount, 2);
        assert.ok(result.includes('Lite fallback snippet'));
        mock.reset();
    });
});

describe('scrapeUrl', () => {
    it('throws on failed Jina response', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            ok: false,
            statusText: 'Bad Gateway',
        }));
        await assert.rejects(
            () => scrapeUrl('https://example.com/article'),
            /Could not extract text/
        );
        mock.reset();
    });

    it('throws on empty Jina response', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            ok: true,
            text: () => Promise.resolve('   '),
        }));
        await assert.rejects(
            () => scrapeUrl('https://example.com/article'),
            /Could not extract text/
        );
        mock.reset();
    });

    it('returns markdown on successful Jina response', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            ok: true,
            text: () => Promise.resolve('# Article Title\n\nSome content here.'),
        }));
        const result = await scrapeUrl('https://example.com/article');
        assert.equal(result, '# Article Title\n\nSome content here.');
        mock.reset();
    });
});
