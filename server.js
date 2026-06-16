import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

import { analyzeContent, analyzeImage } from './services/gemini.js';
import { scrapeUrl } from './services/scraper.js';
import fs from 'fs';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for image uploads
const upload = multer({ dest: 'uploads/' });

// If the AI identified a highly trusted source, boost credibility score by 50% (capped at 100)
// and inject a green flag noting the source.
function applyTrustedBoost(result) {
    if (!result.is_trusted_source) return result;

    const originalScore = result.credibility_score;
    // Add a flat 50 points to the score (capped at 100) instead of multiplying.
    // This ensures that even if the AI gives a low base score due to sensational wording,
    // the verified source status pulls it up into the highly credible range (>50%).
    const boostedScore = Math.min(100, originalScore + 50);
    const sourceName = result.trusted_source_name || 'a verified account';

    const trustedGreenFlag = `⭐ Published by ${sourceName} — a highly reputable, verified official source (+50 pt trust boost)`;

    return {
        ...result,
        credibility_score: boostedScore,
        trusted_boost_applied: true,
        green_flags: [trustedGreenFlag, ...(result.green_flags || [])],
        summary: result.summary
    };
}

// Routes
app.post('/api/analyze/text', async (req, res) => {
    try {
        const { text } = req.body;
        if (!text) return res.status(400).json({ error: 'Text is required' });
        
        const result = await analyzeContent(text);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        console.error('Error analyzing text:', error);
        res.status(500).json({ error: 'Failed to analyze text' });
    }
});

app.post('/api/analyze/url', async (req, res) => {
    try {
        const { url } = req.body;
        if (!url) return res.status(400).json({ error: 'URL is required' });

        const text = await scrapeUrl(url);
        const result = await analyzeContent(text);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        console.error('Error analyzing URL:', error);
        res.status(500).json({ error: 'Failed to analyze URL' });
    }
});

app.post('/api/analyze/image', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'Image is required' });

        const imagePath = req.file.path;
        const imageData = fs.readFileSync(imagePath);
        const base64 = imageData.toString('base64');

        // Detect mime type from magic bytes
        let mimeType = 'image/jpeg';
        if (imageData[0] === 0x89 && imageData[1] === 0x50) mimeType = 'image/png';
        else if (imageData[0] === 0x47 && imageData[1] === 0x49) mimeType = 'image/gif';
        else if (imageData[0] === 0x52 && imageData[1] === 0x49) mimeType = 'image/webp';

        // Clean up temp file
        fs.unlink(imagePath, () => {});

        // Single Gemini call: reads image text + analyzes credibility (saves quota)
        const result = await analyzeImage(base64, mimeType);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        console.error('Error analyzing image:', error);
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a moment and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze image' });
        }
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
