import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

import { analyzeContent, analyzeImage } from './services/groq.js';
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
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a minute and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze text' });
        }
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
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a minute and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze URL' });
        }
    }
});

app.post('/api/analyze/image', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Image is required' });
    
    const imagePath = req.file.path;
    
    try {
        const imageData = await fs.promises.readFile(imagePath);
        const base64 = imageData.toString('base64');
        const mimeType = req.file.mimetype;

        // Pass optional user context (e.g. "this is an AI generated photo of me")
        const userContext = req.body.context || '';
        const result = await analyzeImage(base64, mimeType, userContext);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        console.error('Error analyzing image:', error);
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a moment and try again.' });
        } else {
            res.status(500).json({ error: error.message || 'Failed to analyze image' });
        }
    } finally {
        // Clean up temp file safely
        try {
            await fs.promises.unlink(imagePath);
        } catch (unlinkError) {
            console.error('Error deleting temp image file:', unlinkError);
        }
    }
});

function startServer(port) {
    const server = app.listen(port, () => {
        console.log(`Server running on http://localhost:${port}`);
    });

    server.on('error', async (err) => {
        if (err.code === 'EADDRINUSE') {
            console.log(`Port ${port} is busy. Killing the old process and retrying...`);
            const { exec } = await import('child_process');
            // Find and kill whatever is using the port
            exec(`for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port} ^| findstr LISTENING') do taskkill /F /PID %a`, { shell: 'cmd.exe' }, (error) => {
                setTimeout(() => startServer(port), 1500);
            });
        } else {
            console.error('Server error:', err);
        }
    });
}

startServer(PORT);
