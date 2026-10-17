import { withRetry } from './retry.js';
import { logger } from './logger.js';
import net from 'net';
import http from 'http';
import https from 'https';
import { lookup } from 'dns/promises';

export function cleanHtml(text) {
    return text
        .replace(/<[^>]+>/g, '')
        .replace(/&quot;/g, '"')
        .replace(/&#x27;/g, "'")
        .replace(/&#39;/g, "'")
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/\s+/g, ' ')
        .trim();
}

export function extractSnippets(html) {
    const patterns = [
        /<a[^>]*class="result__snippet[^"]*"[^>]*>(.*?)<\/a>/gi,
        /<span[^>]*class="result__snippet[^"]*"[^>]*>(.*?)<\/span>/gi,
        /<td[^>]*class="result__snippet[^"]*"[^>]*>(.*?)<\/td>/gi,
        /class="result__body"[^>]*>[\s\S]*?<a[^>]*>(.*?)<\/a>/gi,
        /class="snippet"[^>]*>(.*?)<\/[^>]+>/gi,
        /class="result"[^>]*>[\s\S]*?<a[^>]*href="https?:\/\/[^"]*"[^>]*>(.*?)<\/a>/gi,
    ];

    for (const pattern of patterns) {
        pattern.lastIndex = 0;
        const matches = [];
        let match;
        while ((match = pattern.exec(html)) !== null && matches.length < 3) {
            const text = cleanHtml(match[1]);
            if (text && text.length > 10) matches.push(text);
        }
        if (matches.length > 0) return matches;
    }

    const genericLinks = html.match(/<a[^>]*href="https?:\/\/[^"]*"[^>]*>([^<]{20,})<\/a>/gi);
    if (genericLinks) {
        return genericLinks.slice(0, 3).map(l => cleanHtml(l.replace(/<a[^>]*>/i, '').replace(/<\/a>/i, ''))).filter(Boolean);
    }

    return [];
}

async function searchJina(query) {
    try {
        const response = await fetch('https://s.jina.ai/' + encodeURIComponent(query), {
            headers: {
                'User-Agent': 'FakeNewsDetector/1.0',
                'Accept': 'text/plain'
            },
            signal: AbortSignal.timeout(15000)
        });
        if (!response.ok) return null;
        const text = await response.text();
        if (!text || text.length < 50) return null;
        // Jina returns markdown with line-item results — take first 3 non-empty lines
        const lines = text.split('\n').filter(l => l.trim().length > 20 && !l.startsWith('!') && !l.startsWith('[')).slice(0, 3);
        return lines.length > 0 ? lines.join('\n- ') : null;
    } catch (err) {
        logger.warn('Jina search failed:', err?.message || err);
        return null;
    }
}

async function tryQuery(query) {
    for (const endpoint of [
        { url: 'https://lite.duckduckgo.com/lite/?q=' + encodeURIComponent(query), method: 'GET' },
        { url: 'https://html.duckduckgo.com/html/', method: 'POST', body: 'q=' + encodeURIComponent(query) },
    ]) {
        try {
            const response = await fetch(endpoint.url, {
                method: endpoint.method,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                body: endpoint.body,
                signal: AbortSignal.timeout(10000)
            });

            const html = await response.text();
            if (!html || html.length < 50) continue;

            const snippets = extractSnippets(html);
            if (snippets.length > 0) {
                return snippets.join('\n- ');
            }
        } catch {
            // Ignore, try next
        }
    }
    return null;
}

const STOP_WORDS = new Set(['the', 'a', 'an', 'is', 'it', 'was', 'to', 'in', 'for', 'of', 'on', 'at', 'by', 'with', 'and', 'or', 'but', 'not', 'are', 'were', 'has', 'had', 'have', 'been', 'its', 'this', 'that', 'from', 's']);

export function extractKeyTerms(text) {
    const cleaned = text.replace(/[^a-zA-Z\s]/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
    const words = cleaned.split(' ').filter(w => w.length > 2 && !STOP_WORDS.has(w));
    return words.slice(0, 6).join(' ');
}

export async function searchWeb(query) {
    // Try the original query first
    let result = await tryQuery(query);
    if (result) return result;

    // Fallback: try Jina with original query
    const jinaResult = await searchJina(query);
    if (jinaResult) return jinaResult;

    // Fallback: extract key terms and retry (handles typos in non-essential words)
    const keyTerms = extractKeyTerms(query);
    if (keyTerms && keyTerms !== query.toLowerCase().trim()) {
        result = await tryQuery(keyTerms);
        if (result) return result;

        const jinaFallback = await searchJina(keyTerms);
        if (jinaFallback) return jinaFallback;
    }

    return 'Search engines could not find relevant web results for this query. This does not mean the claim is false — only that no matching pages were found online. Fact-check based on general knowledge and internal reasoning.';
}

export function extractMetaContent(html, property) {
    const patterns = [
        new RegExp(`<meta[^>]+property="${property}"[^>]+content="([^"]*)"`, 'i'),
        new RegExp(`<meta[^>]+name="${property}"[^>]+content="([^"]*)"`, 'i'),
        new RegExp(`<meta[^>]+content="([^"]*)"[^>]+property="${property}"`, 'i'),
        new RegExp(`<meta[^>]+content="([^"]*)"[^>]+name="${property}"`, 'i'),
    ];
    for (const pattern of patterns) {
        const match = html.match(pattern);
        if (match) return match[1].replace(/&#\d+;/g, '').replace(/&amp;/g, '&').replace(/&quot;/g, '"').trim();
    }
    return null;
}

export function extractBodyText(html) {
    const body = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
    if (!body) return null;
    const text = body[1]
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<style[\s\S]*?<\/style>/gi, '')
        .replace(/<nav[\s\S]*?<\/nav>/gi, '')
        .replace(/<footer[\s\S]*?<\/footer>/gi, '')
        .replace(/<header[\s\S]*?<\/header>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&#\d+;/g, '')
        .replace(/&amp;/g, '&')
        .replace(/\s+/g, ' ')
        .trim();
    return text.length > 50 ? text : null;
}

// ── SSRF PROTECTION ──
// The /api/analyze/url endpoint lets users submit arbitrary URLs which the server
// fetches (scrapeDirect). Without validation, a malicious user could point the
// server at http://127.0.0.1:8000, cloud metadata endpoints (169.254.169.254),
// or other internal services. We validate every hop: the target host and each
// redirect destination must resolve to a public (non-private) IP, or the request
// is abandoned. DNS is re-checked per hop to close most rebinding gaps, and
// failures fail CLOSED (blocked).

function ipv4ToInt(ip) {
    const parts = ip.split('.').map(Number);
    return (((parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]) >>> 0);
}

function isPrivateIpv4(n) {
    n = n >>> 0;
    const a = (n >>> 24) & 0xff;
    const b = (n >>> 16) & 0xff;
    const c = (n >>> 8) & 0xff;
    if (a === 0) return true;                              // 0.0.0.0/8 "this network"
    if (a === 10) return true;                             // 10.0.0.0/8 private
    if (a === 127) return true;                            // 127.0.0.0/8 loopback
    if (a === 100 && b >= 64 && b <= 127) return true;     // 100.64.0.0/10 CGNAT
    if (a === 169 && b === 254) return true;               // 169.254.0.0/16 link-local
    if (a === 172 && b >= 16 && b <= 31) return true;      // 172.16.0.0/12 private
    if (a === 192 && b === 0 && (c === 0 || c === 2)) return true;  // 192.0.0.0/24, 192.0.2.0/24
    if (a === 192 && b === 88 && c === 99) return true;    // 192.88.99.0/24 6to4
    if (a === 192 && b === 168) return true;               // 192.168.0.0/16 private
    if (a === 198 && (b === 18 || b === 19)) return true;  // 198.18.0.0/15 benchmarking
    if (a === 198 && b === 51 && c === 100) return true;   // 198.51.100.0/24 TEST-NET-2
    if (a === 203 && b === 0 && c === 113) return true;    // 203.0.113.0/24 TEST-NET-3
    if (a >= 224) return true;                             // multicast + reserved + broadcast
    return false;
}

function ipv6Hextets(ip) {
    let s = String(ip).trim().replace(/^\[|\]$/g, '');
    const zone = s.indexOf('%');
    if (zone !== -1) s = s.slice(0, zone);
    s = s.toLowerCase();
    if (s.includes('.')) {
        const lastColon = s.lastIndexOf(':');
        const octets = s.slice(lastColon + 1).split('.').map(Number);
        if (octets.length !== 4 || octets.some(o => !Number.isInteger(o) || o < 0 || o > 255)) return null;
        const hi = ((octets[0] << 8) | octets[1]).toString(16);
        const lo = ((octets[2] << 8) | octets[3]).toString(16);
        s = s.slice(0, lastColon + 1) + hi + ':' + lo;
    }
    const halves = s.split('::');
    if (halves.length > 2) return null;
    const head = halves[0] ? halves[0].split(':') : [];
    const tail = halves.length === 2 && halves[1] ? halves[1].split(':') : [];
    if (halves.length === 2) {
        if (head.length + tail.length > 7) return null; // "::" must cover >= 1 group
    } else if (head.length !== 8) {
        return null;
    }
    const hextets = new Array(8).fill(0);
    head.forEach((g, i) => { hextets[i] = parseInt(g, 16); });
    tail.forEach((g, i) => { hextets[8 - tail.length + i] = parseInt(g, 16); });
    if (hextets.some(h => Number.isNaN(h) || h < 0 || h > 0xffff)) return null;
    return hextets;
}

function isPrivateIpv6(h) {
    const [a, b, c, d, e, f, g, last] = h;
    if (h.every(x => x === 0)) return true;                                     // :: unspecified
    if (a === 0 && b === 0 && c === 0 && d === 0 && e === 0 && f === 0 && g === 0 && last === 1) return true; // ::1
    if (a === 0 && b === 0 && c === 0 && d === 0 && e === 0 && f === 0xffff) return isPrivateIpv4(((g << 16) | last) >>> 0); // ::ffff:0:0/96
    if (a === 0 && b === 0 && c === 0 && d === 0 && e === 0 && f === 0) return true;  // ::/96 IPv4-compatible (deprecated)
    if (a === 0x64 && b === 0xff9b && c === 0 && d === 0 && e === 0 && f === 0) return isPrivateIpv4(((g << 16) | last) >>> 0); // NAT64
    if (a === 0x2002) return isPrivateIpv4(((b << 16) | c) >>> 0);             // 2002::/16 6to4
    if (a === 0x2001 && b === 0x0000) return true;                             // 2001::/32 Teredo
    if ((a & 0xfe00) === 0xfc00) return true;                                  // fc00::/7 ULA
    if ((a & 0xffc0) === 0xfe80) return true;                                  // fe80::/10 link-local
    if ((a & 0xff00) === 0xff00) return true;                                  // ff00::/8 multicast
    if (a === 0x2001 && b === 0x0db8) return true;                             // 2001:db8::/32 documentation
    return false;
}

export function isPrivateIp(ip) {
    const addr = String(ip).trim().replace(/^\[|\]$/g, '');
    if (net.isIPv4(addr)) return isPrivateIpv4(ipv4ToInt(addr));
    if (net.isIPv6(addr)) {
        const hextets = ipv6Hextets(addr);
        return hextets ? isPrivateIpv6(hextets) : true; // fail closed
    }
    return true; // unknown format — treat as unsafe
}

async function lookupHost(hostname) {
    try {
        const addresses = await lookup(hostname, { all: true, verbatim: true });
        return addresses.map(entry => entry.address);
    } catch {
        return null;
    }
}

export async function validatePublicUrl(raw, resolveHost = lookupHost) {
    let url;
    try { url = new URL(raw.trim()); } catch { return { ok: false, url: null }; }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return { ok: false, url: null };
    if (!url.hostname) return { ok: false, url: null };
    const addresses = await resolveHost(url.hostname);
    if (!addresses || addresses.length === 0) return { ok: false, url: null };
    if (addresses.some(isPrivateIp)) return { ok: false, url: null };
    return { ok: true, url };
}

// Validate the peer IP at connect time so a hostile DNS answer cannot slip
// between the pre-check and the actual connection (DNS-rebinding / TOCTOU).
function safeLookup(hostname, options, callback) {
    lookup(hostname, { ...options, all: true })
        .then((addresses) => {
            const list = Array.isArray(addresses) ? addresses : [addresses];
            if (list.length === 0) return callback(new Error(`No address for ${hostname}`));
            for (const entry of list) {
                if (isPrivateIp(entry.address)) {
                    return callback(new Error(`Blocked private address ${entry.address}`));
                }
            }
            callback(null, list[0].address, list[0].family);
        })
        .catch(callback);
}

const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const BROWSER_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

// Minimal GET that validates the peer IP at connect time (safeLookup) and caps
// the response size, replacing fetch() so the SSRF guard cannot be bypassed.
function requestOnce(url) {
    return new Promise((resolve, reject) => {
        const mod = url.protocol === 'https:' ? https : http;
        const req = mod.request(url, {
            method: 'GET',
            headers: { 'User-Agent': BROWSER_UA },
            lookup: safeLookup,
            signal: AbortSignal.timeout(15000),
        }, (res) => {
            const chunks = [];
            let size = 0;
            res.on('data', (chunk) => {
                size += chunk.length;
                if (size > MAX_RESPONSE_BYTES) {
                    req.destroy(new Error('Response too large'));
                    return;
                }
                chunks.push(chunk);
            });
            res.on('end', () => resolve({
                status: res.statusCode,
                headers: res.headers,
                body: Buffer.concat(chunks).toString('utf8'),
            }));
            res.on('error', reject);
        });
        req.on('error', reject);
        req.end();
    });
}

async function scrapeDirect(url, options = {}) {
    const request = options.request || requestOnce;
    // Validate each hop (target + redirects) against the SSRF guard. Redirects
    // are followed manually so a hostile server cannot bounce us to an internal
    // address, and safeLookup re-checks the IP of every connection we open.
    let current;
    try { current = new URL(url); } catch { return null; }

    for (let hop = 0; hop < 5; hop++) {
        const checked = await validatePublicUrl(current.href, options.resolveHost);
        if (!checked.ok) return null;
        current = checked.url;

        const res = await request(current).catch(() => null);
        if (!res) return null;

        if ([301, 302, 303, 307, 308].includes(res.status)) {
            const loc = res.headers?.location;
            if (!loc) return null;
            try { current = new URL(loc, current); } catch { return null; }
            continue;
        }

        if (res.status < 200 || res.status >= 300) return null;
        const html = res.body || '';
        if (html.length < 100) return null;
        // Try meta tags first (works for Twitter/X, most social media, news sites)
        const ogDesc = extractMetaContent(html, 'og:description');
        if (ogDesc && ogDesc.length > 20) return ogDesc;
        const twitterDesc = extractMetaContent(html, 'twitter:description');
        if (twitterDesc && twitterDesc.length > 20) return twitterDesc;
        const metaDesc = extractMetaContent(html, 'description');
        if (metaDesc && metaDesc.length > 50) return metaDesc;
        // Fallback: extract body text
        const bodyText = extractBodyText(html);
        if (bodyText) return bodyText;
        return null;
    }
    return null;
}

export async function scrapeUrl(url, options = {}) {
    let parsed;
    try { parsed = new URL(String(url).trim()); } catch { parsed = null; }
    if (!parsed || (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') || !parsed.hostname) {
        logger.warn(`scrapeUrl rejected URL (unsupported protocol): ${String(url).slice(0, 120)}`);
        throw new Error(`Could not extract text from unsupported URL: ${url}`);
    }

    // SSRF guard up front so a private/invalid target is never handed to the
    // Jina reader (which would otherwise fetch it on our behalf, bypassing the
    // direct-fetch guard) nor to scrapeDirect.
    const guard = await validatePublicUrl(parsed.href, options.resolveHost);
    if (!guard.ok) {
        logger.warn(`scrapeUrl blocked URL (SSRF guard): ${parsed.href.slice(0, 120)}`);
        throw new Error('Could not extract text: the URL is private, unresolvable, or invalid.');
    }

    // Preferred reader: renders JS-heavy pages. Only reached for public URLs.
    const jinaUrl = `https://r.jina.ai/${parsed.href}`;
    try {
        const response = await withRetry(() => fetch(jinaUrl, {
            headers: {
                'User-Agent': 'FakeNewsDetector/1.0',
                'X-Return-Format': 'markdown'
            },
            signal: AbortSignal.timeout(15000)
        }), { maxRetries: 2, onRetry: (err, a) => logger.warn(`scrapeUrl retry ${a}: ${err.message}`) });

        if (response.ok) {
            const markdown = await response.text();
            if (markdown && markdown.trim().length > 0) {
                return markdown.trim();
            }
        }
    } catch (error) {
        logger.warn('Jina scrape failed, trying direct fetch:', error?.message || error);
    }

    const directResult = await scrapeDirect(parsed.href, options);
    if (directResult) return directResult;

    throw new Error('Could not extract text from the provided URL. It might be protected or invalid.');
}
