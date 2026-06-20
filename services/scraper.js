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
