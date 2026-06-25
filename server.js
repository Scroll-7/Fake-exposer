import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';

import { analyzeContent, analyzeImage } from './services/groq.js';
import { scrapeUrl } from './services/scraper.js';
import fs from 'fs';
import crypto from 'crypto';

// Initialize simple file-based cache for expensive image analysis
const CACHE_FILE = path.join(__dirname, 'image_cache.json');
let imageCache = {};
try {
    if (fs.existsSync(CACHE_FILE)) {
        imageCache = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf-8'));
    }
} catch (e) {
    console.error("Failed to load image cache:", e);
}

function saveCache() {
    try {
        fs.writeFileSync(CACHE_FILE, JSON.stringify(imageCache, null, 2));
    } catch (e) {
        console.error("Failed to save image cache:", e);
    }
}

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// --- Rate Limiting ---
// Protect against bot spam and API cost drain by limiting IPs to 15 requests per hour.
const apiLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour window
    max: 15, // limit each IP to 15 requests per windowMs
    message: { error: "Too many requests from this IP. Please try again after an hour." },
    standardHeaders: true,
    legacyHeaders: false,
});

// Apply rate limiting specifically to the analysis endpoints
app.use('/api/analyze/', apiLimiter);

// Configure multer for image uploads
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const upload = multer({ dest: UPLOADS_DIR });

// --- Background Cleanup Job ---
// Automatically clean up any stuck files in the uploads folder older than 1 hour.
setInterval(() => {
    fs.readdir(UPLOADS_DIR, (err, files) => {
        if (err) return;
        const now = Date.now();
        files.forEach(file => {
            const filePath = path.join(UPLOADS_DIR, file);
            fs.stat(filePath, (err, stats) => {
                if (err) return;
                // If file is older than 1 hour (3600000 ms), delete it
                if (now - stats.mtimeMs > 3600000) {
                    fs.unlink(filePath, unlinkErr => {
                        if (!unlinkErr) console.log(`🧹 Background cleanup: deleted stale file ${file}`);
                    });
                }
            });
        });
    });
}, 3600000); // Run every hour

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

// Keywords in filenames that strongly indicate AI-generated images
const AI_FILENAME_KEYWORDS = [
    'chatgpt', 'chat_gpt', 'chat-gpt',
    'gemini', 'google_gemini', 'gemini_image',
    'midjourney', 'mid_journey', 'mid-journey',
    'dall-e', 'dalle', 'dall_e',
    'stable_diffusion', 'stablediffusion', 'stable-diffusion',
    'ai_generated', 'ai-generated', 'aigenerated', 'ai_image', 'ai-image',
    'deepfake', 'deep_fake', 'deep-fake',
    'generated', 'gen_ai', 'genai',
    'flux', 'firefly', 'adobe_firefly', 'sora', 'ideogram', 'leonardo',
    'bing_image', 'imagefx', 'image_fx', 'imagen',
    'gpt', 'openai',
    'artificial', 'synthetic',
    'fake', 'edited', 'enhanced', 'photoshop', 'ps_edit',
    'portrait_ai', 'portraitai', 'remini', 'faceapp',
];

function checkAiFilename(originalName) {
    if (!originalName) return null;
    const lower = originalName.toLowerCase().replace(/[^a-z0-9]/g, ' ');
    const matched = AI_FILENAME_KEYWORDS.filter(kw => lower.includes(kw.replace(/[-_]/g, ' ').replace(/[-_]/g, '')));
    if (matched.length > 0) {
        return matched[0]; // return the first matched keyword
    }
    return null;
}

app.post('/api/analyze/image', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Image is required' });
    
    const imagePath = req.file.path;
    const originalName = req.file.originalname || '';
    
    try {
        const imageData = await fs.promises.readFile(imagePath);
        const base64 = imageData.toString('base64');
        const mimeType = req.file.mimetype;

        // Check if the filename itself reveals AI origin
        const aiKeywordMatch = checkAiFilename(originalName);
        if (aiKeywordMatch) {
            console.log(`⚠️ AI keyword detected in filename: "${originalName}" (matched: "${aiKeywordMatch}") — returning immediate fake verdict.`);
            res.json({
                credibility_score: 5,
                verdict: 'Confirmed Fake / AI-Generated',
                bias: 'None',
                sentiment: 'Neutral',
                red_flags: [
                    `🚨 Filename reveals AI origin: "${originalName}" contains "${aiKeywordMatch}"`,
                    'File names from AI tools (ChatGPT, Midjourney, DALL-E, etc.) are a strong indicator of AI generation.',
                ],
                green_flags: [],
                summary: `The filename "${originalName}" contains the keyword "${aiKeywordMatch}", which is a direct indicator that this image was generated or processed by an AI tool. The image is almost certainly fake or AI-generated.`,
                recommendations: ['Do not share this image as genuine.', 'Reverse image search to find the original if one exists.'],
                is_trusted_source: false,
                trusted_source_name: '',
                filename_flag: aiKeywordMatch,
            });
            return;
        }

        // Generate SHA-256 hash of the image to check cache
        const hash = crypto.createHash('sha256').update(imageData).digest('hex');
        
        if (imageCache[hash]) {
            console.log(`⚡ Cache hit for image: ${hash} (Returning instantly)`);
            return res.json(applyTrustedBoost(imageCache[hash]));
        }

        // Pass optional user context (e.g. "this is an AI generated photo of me")
        const userContext = req.body.context || '';
        const result = await analyzeImage(base64, mimeType, userContext, originalName);
        
        // Save result to cache
        imageCache[hash] = result;
        saveCache();
        
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
