import { describe, it, mock } from 'node:test';
import assert from 'node:assert/strict';
import { searchWeb, scrapeUrl, validatePublicUrl, isPrivateIp } from '../services/scraper.js';

describe('searchWeb', () => {
    it('returns fallback message when fetch fails', async () => {
        mock.method(global, 'fetch', () => Promise.reject(new Error('Network error')));
        const result = await searchWeb('test query');
        assert.ok(result.includes('Search engines could not find'));
        mock.reset();
    });

    it('returns fallback on empty HTML', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            text: () => Promise.resolve('<html></html>'),
        }));
        const result = await searchWeb('test query');
        assert.ok(result.includes('Search engines could not find'));
        mock.reset();
    });

    it('extracts snippets from DuckDuckGo HTML', async () => {
        // Realistic DuckDuckGo HTML snippet with proper length
        const html = '<!DOCTYPE html><html><body><div class="results"><div class="result"><a class="result__snippet" href="http://example.com">This is a test snippet about fake news detection using AI and machine learning methods</a></div></div></body></html>';
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

describe('SSRF guard (validatePublicUrl / isPrivateIp)', () => {
    it('classifies private and loopback ranges as unsafe', () => {
        assert.equal(isPrivateIp('127.0.0.1'), true);
        assert.equal(isPrivateIp('10.0.0.1'), true);
        assert.equal(isPrivateIp('172.16.0.1'), true);
        assert.equal(isPrivateIp('172.31.255.255'), true);
        assert.equal(isPrivateIp('192.168.1.42'), true);
        assert.equal(isPrivateIp('169.254.169.254'), true);
        assert.equal(isPrivateIp('::1'), true);
        assert.equal(isPrivateIp('fe80::1'), true);
    });

    it('allows public IPs', () => {
        assert.equal(isPrivateIp('8.8.8.8'), false);
        assert.equal(isPrivateIp('1.1.1.1'), false);
    });

    it('rejects non-http(s) URLs without DNS lookups', async () => {
        assert.equal((await validatePublicUrl('file:///etc/passwd')).ok, false);
        assert.equal((await validatePublicUrl('ftp://example.com/file')).ok, false);
        assert.equal((await validatePublicUrl('data:text/plain,hello')).ok, false);
    });

    it('rejects hosts that resolve to private IPs', async () => {
        const resolvePrivate = async () => ['127.0.0.1', '10.0.0.5'];
        const result = await validatePublicUrl('http://internal.local/page', resolvePrivate);
        assert.equal(result.ok, false);
    });

    it('rejects hosts where any resolved IP is private', async () => {
        const resolveMixed = async () => ['8.8.8.8', '169.254.169.254'];
        const result = await validatePublicUrl('http://mixed.example/', resolveMixed);
        assert.equal(result.ok, false);
    });

    it('accepts public hosts', async () => {
        const resolvePublic = async () => ['93.184.216.34'];
        const result = await validatePublicUrl('https://example.com/article', resolvePublic);
        assert.equal(result.ok, true);
        assert.equal(result.url.href, 'https://example.com/article');
    });

    it('fails closed when DNS resolution fails', async () => {
        const resolveFail = async () => null;
        assert.equal((await validatePublicUrl('https://example.com/', resolveFail)).ok, false);
    });

    it('scrapeUrl throws for unsupported protocols', async () => {
        await assert.rejects(
            () => scrapeUrl('ftp://example.com/file'),
            /Could not extract text/
        );
    });
});
