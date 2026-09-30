import { withRetry } from './retry.js';
import { logger } from './logger.js';
import net from 'net';
import { lookup } from 'dns/promises';

function cleanHtml(text) {
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

function extractSnippets(html) {
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

function extractKeyTerms(text) {
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

function extractMetaContent(html, property) {
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

function extractBodyText(html) {
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

export function isPrivateIp(ip) {
    if (net.isIPv4(ip)) {
        const [a, b] = ip.split('.').map(Number);
        if (a === 0 || a === 10 || a === 127) return true;
        if (a === 100 && b >= 64 && b <= 127) return true;     // CGNAT
        if (a === 169 && b === 254) return true;                // link-local (AWS metadata)
        if (a === 172 && b >= 16 && b <= 31) return true;       // private
        if (a === 192 && b === 168) return true;                // private
        return false;
    }
    if (net.isIPv6(ip)) {
        const lower = ip.toLowerCase();
        if (lower === '::1') return true;
        if (/^fe80:/.test(lower) || /^f[cd]/.test(lower)) return true; // link-local + ULA (fc00::/7)
        const v4Mapped = lower.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
        if (v4Mapped) return isPrivateIp(v4Mapped[1]);
        return false;
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

async function scrapeDirect(url) {
    // Validate each hop (target + redirects) against the SSRF guard. 'manual'
    // redirect mode means we control every destination rather than trusting
    // fetch() to follow redirects to wherever a hostile server points us.
    let current;
    try { current = new URL(url); } catch { return null; }

    for (let hop = 0; hop < 5; hop++) {
        const checked = await validatePublicUrl(current.href);
        if (!checked.ok) return null;
        current = checked.url;

        const res = await fetch(current, {
            redirect: 'manual',
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            signal: AbortSignal.timeout(15000)
        }).catch(() => null);
        if (!res) return null;

        if ([301, 302, 303, 307, 308].includes(res.status)) {
            const loc = res.headers.get('location');
            if (!loc) return null;
            try { current = new URL(loc, current); } catch { return null; }
            continue;
        }

        if (!res.ok) return null;
        const html = await res.text();
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

export async function scrapeUrl(url) {
    let lastError;
    try {
        // Reject non-http(s) schemes up front (the full DNS guard lives in scrapeDirect)
        let parsed;
        try { parsed = new URL(url.trim()); } catch { parsed = null; }
        if (!parsed || (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') || !parsed.hostname) {
            logger.warn(`scrapeUrl rejected URL (unsupported protocol): ${String(url).slice(0, 120)}`);
            throw new Error(`Unsupported URL protocol: ${url}`);
        }

        const jinaUrl = `https://r.jina.ai/${url}`;

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
        lastError = new Error(`Jina API failed: ${response.statusText}`);
    } catch (error) {
        lastError = error;
        logger.warn('Jina scrape failed, trying direct fetch:', error.message);
    }

    // Fallback: direct HTTP fetch + meta tag extraction
    const directResult = await scrapeDirect(url);
    if (directResult) return directResult;

    logger.error('Scraping error:', lastError);
    throw new Error('Could not extract text from the provided URL. It might be protected or invalid.');
}
