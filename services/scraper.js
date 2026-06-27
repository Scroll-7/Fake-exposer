export async function searchWeb(query) {
    try {
        const response = await fetch('https://html.duckduckgo.com/html/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            body: 'q=' + encodeURIComponent(query),
            signal: AbortSignal.timeout(10000)
        });
        const html = await response.text();
        const snippetRegex = /<a class="result__snippet[^>]*>(.*?)<\/a>/gi;
        let match;
        const snippets = [];
        while ((match = snippetRegex.exec(html)) !== null && snippets.length < 3) {
            let text = match[1].replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').trim();
            if (text) snippets.push(text);
        }
        return snippets.length ? snippets.join('\n- ') : "No recent news found.";
    } catch (err) {
        console.error("Search failed:", err.message);
        return "No recent news found.";
    }
}

export async function scrapeUrl(url) {
    try {
        // Use Jina Reader API to fetch and parse the URL into clean Markdown.
        // This automatically handles JS-rendered sites and strips out boilerplate.
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
        
        // Jina returns markdown format. This is actually perfect for Gemini analysis.
        return markdown.trim();
    } catch (error) {
        console.error('Scraping error:', error);
        throw new Error('Could not extract text from the provided URL. It might be protected or invalid.');
    }
}
