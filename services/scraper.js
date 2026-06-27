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
    } catch {
        return null;
    }
}

export async function searchWeb(query) {
    const errors = [];

    // Try DuckDuckGo (lite endpoint first — simpler HTML, less likely to break)
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
        } catch (err) {
            errors.push(err.message);
        }
    }

    // Fallback: Jina AI search API
    const jinaResult = await searchJina(query);
    if (jinaResult) return jinaResult;

    console.error("Search failed:", errors.join('; '));
    return "No recent news found.";
}

export async function scrapeUrl(url) {
    try {
        const jinaUrl = `https://r.jina.ai/${url}`;

        const response = await fetch(jinaUrl, {
            headers: {
                'User-Agent': 'FakeNewsDetector/1.0',
                'X-Return-Format': 'markdown'
            }
        });

        if (!response.ok) {
            throw new Error(`Jina API failed: ${response.statusText}`);
        }

        const markdown = await response.text();

        if (!markdown || markdown.trim() === '') {
            throw new Error('No content returned from Jina API');
        }

        return markdown.trim();
    } catch (error) {
        console.error('Scraping error:', error);
        throw new Error('Could not extract text from the provided URL. It might be protected or invalid.');
    }
}
