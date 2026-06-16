import * as cheerio from 'cheerio';

export async function scrapeUrl(url) {
    try {
        // We use native fetch in Node 18+
        const response = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch URL: ${response.statusText}`);
        }
        
        const html = await response.text();
        const $ = cheerio.load(html);
        
        // Remove scripts, styles, nav, footer, etc to get cleaner text
        $('script, style, nav, footer, header, aside, .ad, .advertisement').remove();
        
        // Try to find the main article content. If not, just get body text.
        let articleText = $('article').text();
        if (!articleText || articleText.trim() === '') {
            articleText = $('body').text();
        }
        
        // Clean up whitespace
        return articleText.replace(/\s+/g, ' ').trim();
    } catch (error) {
        console.error('Scraping error:', error);
        throw new Error('Could not extract text from the provided URL.');
    }
}
