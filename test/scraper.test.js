import { describe, it, mock } from 'node:test';
import assert from 'node:assert/strict';
import { searchWeb, scrapeUrl, validatePublicUrl, isPrivateIp, cleanHtml, extractSnippets, extractKeyTerms, extractMetaContent, extractBodyText } from '../services/scraper.js';

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
    const resolvePublic = async () => ['93.184.216.34'];
    const resolvePrivate = async () => ['127.0.0.1'];
    const rejectRequest = async () => { throw new Error('no direct fetch in tests'); };

    it('throws on failed Jina response', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            ok: false,
            statusText: 'Bad Gateway',
        }));
        await assert.rejects(
            () => scrapeUrl('https://example.com/article', { resolveHost: resolvePublic, request: rejectRequest }),
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
            () => scrapeUrl('https://example.com/article', { resolveHost: resolvePublic, request: rejectRequest }),
            /Could not extract text/
        );
        mock.reset();
    });

    it('returns markdown on successful Jina response', async () => {
        mock.method(global, 'fetch', () => Promise.resolve({
            ok: true,
            text: () => Promise.resolve('# Article Title\n\nSome content here.'),
        }));
        const result = await scrapeUrl('https://example.com/article', { resolveHost: resolvePublic });
        assert.equal(result, '# Article Title\n\nSome content here.');
        mock.reset();
    });

    it('blocks private targets before they reach the Jina reader', async () => {
        let jinaCalled = false;
        mock.method(global, 'fetch', () => {
            jinaCalled = true;
            return Promise.resolve({ ok: true, text: () => Promise.resolve('secret') });
        });
        await assert.rejects(
            () => scrapeUrl('http://localhost:8000/admin', { resolveHost: resolvePrivate }),
            /private, unresolvable, or invalid/
        );
        assert.equal(jinaCalled, false);
        mock.reset();
    });

    it('rejects an IP-literal loopback target', async () => {
        await assert.rejects(
            () => scrapeUrl('http://[::1]:8000/router', { resolveHost: resolvePrivate }),
            /private, unresolvable, or invalid/
        );
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

    it('flags additional reserved and mapped ranges', () => {
        assert.equal(isPrivateIp('0.0.0.1'), true);
        assert.equal(isPrivateIp('100.64.0.1'), true);   // CGNAT lower bound
        assert.equal(isPrivateIp('100.127.255.255'), true); // CGNAT upper bound
        assert.equal(isPrivateIp('::ffff:127.0.0.1'), true);
        assert.equal(isPrivateIp('fc00::1'), true);      // unique local
        assert.equal(isPrivateIp('fd12:3456::1'), true); // unique local
    });

    it('allows addresses just outside private ranges', () => {
        assert.equal(isPrivateIp('100.63.0.1'), false);
        assert.equal(isPrivateIp('100.128.0.0'), false);
        assert.equal(isPrivateIp('172.15.0.1'), false);
        assert.equal(isPrivateIp('172.32.0.1'), false);
        assert.equal(isPrivateIp('::ffff:8.8.8.8'), false);
        assert.equal(isPrivateIp('2001:4860:4860::8888'), false);
    });

    it('fails closed for unrecognised input', () => {
        assert.equal(isPrivateIp('not-an-ip'), true);
        assert.equal(isPrivateIp(''), true);
    });

    it('classifies reserved, multicast and documentation IPv4 ranges', () => {
        assert.equal(isPrivateIp('0.0.0.0'), true);
        assert.equal(isPrivateIp('192.0.2.1'), true);    // TEST-NET-1
        assert.equal(isPrivateIp('192.88.99.1'), true);  // 6to4 relay
        assert.equal(isPrivateIp('198.18.0.1'), true);   // benchmarking
        assert.equal(isPrivateIp('198.51.100.1'), true); // TEST-NET-2
        assert.equal(isPrivateIp('203.0.113.1'), true);  // TEST-NET-3
        assert.equal(isPrivateIp('224.0.0.1'), true);    // multicast
        assert.equal(isPrivateIp('240.0.0.1'), true);    // reserved
        assert.equal(isPrivateIp('255.255.255.255'), true);
    });

    it('unwraps IPv6 encodings that hide private IPv4', () => {
        assert.equal(isPrivateIp('::ffff:7f00:1'), true);    // hex IPv4-mapped loopback
        assert.equal(isPrivateIp('::ffff:127.0.0.1'), true);
        assert.equal(isPrivateIp('[::1]'), true);            // bracketed literal
        assert.equal(isPrivateIp('64:ff9b::7f00:1'), true);  // NAT64 -> 127.0.0.1
        assert.equal(isPrivateIp('64:ff9b::808:808'), false); // NAT64 -> 8.8.8.8
        assert.equal(isPrivateIp('2002:7f00:1::'), true);    // 6to4 -> 127.0.0.1
        assert.equal(isPrivateIp('2001:db8::1'), true);      // documentation
        assert.equal(isPrivateIp('ff02::1'), true);          // multicast
    });
});

describe('cleanHtml', () => {
    it('strips tags and collapses whitespace', () => {
        assert.equal(cleanHtml('<p>Hello   <b>world</b></p>'), 'Hello world');
        assert.equal(cleanHtml('  a\n\n  b  '), 'a b');
    });

    it('decodes common HTML entities', () => {
        assert.equal(cleanHtml('&quot;q&quot; &amp; &#39;a&#39; &lt;x&gt;'), '"q" & \'a\' <x>');
    });
});

describe('extractSnippets', () => {
    it('returns [] for empty or non-matching HTML', () => {
        assert.deepEqual(extractSnippets(''), []);
        assert.deepEqual(extractSnippets('<div>nothing here</div>'), []);
    });

    it('extracts result__snippet anchors', () => {
        const html = '<a class="result__snippet">First good snippet</a>'
            + '<a class="result__snippet extra">Second good snippet</a>';
        assert.deepEqual(extractSnippets(html), ['First good snippet', 'Second good snippet']);
    });

    it('ignores snippets shorter than 10 characters', () => {
        assert.deepEqual(extractSnippets('<a class="result__snippet">tiny</a>'), []);
    });

    it('falls back to long generic links', () => {
        const html = '<a href="https://example.com/a/b">This is a long enough link text</a>';
        assert.deepEqual(extractSnippets(html), ['This is a long enough link text']);
    });
});

describe('extractKeyTerms', () => {
    it('lowercases, strips stopwords, short words and punctuation', () => {
        assert.equal(extractKeyTerms('The Quick Brown Fox jumps!'), 'quick brown fox jumps');
        assert.equal(extractKeyTerms("Mbappe's amazing goal"), 'mbappe amazing goal');
    });

    it('caps the result at six terms', () => {
        const terms = extractKeyTerms('alpha bravo charlie delta echo foxtrot golf hotel');
        assert.equal(terms.split(' ').length, 6);
    });
});

describe('extractMetaContent', () => {
    it('reads content in either attribute order', () => {
        assert.equal(extractMetaContent('<meta property="og:description" content="Hello desc">', 'og:description'), 'Hello desc');
        assert.equal(extractMetaContent('<meta content="Reverse order" name="description">', 'description'), 'Reverse order');
    });

    it('decodes entities and returns null when absent', () => {
        assert.equal(extractMetaContent('<meta name="description" content="A &amp; B">', 'description'), 'A & B');
        assert.equal(extractMetaContent('<html></html>', 'description'), null);
    });
});

describe('extractBodyText', () => {
    const long = 'This is a sufficiently long piece of article body text used for testing extraction.';

    it('removes scripts, styles and markup, returning plain text', () => {
        const html = `<body><script>var x = 1;</script><style>.a{}</style><p>${long}</p></body>`;
        const out = extractBodyText(html);
        assert.ok(out.includes('sufficiently long piece'));
        assert.ok(!out.includes('var x'));
        assert.ok(!out.includes('<p>'));
    });

    it('returns null when there is no body or the text is too short', () => {
        assert.equal(extractBodyText('<div>no body here</div>'), null);
        assert.equal(extractBodyText('<body><p>too short</p></body>'), null);
    });
});
