import express from 'express';
import compression from 'compression';
import cors from 'cors';
import helmet from 'helmet';
import hpp from 'hpp';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

import { analyzeContent, analyzeImage } from './services/groq.js';
import { analyzeText as detectText } from './services/textDetector.js';
import { scrapeUrl } from './services/scraper.js';
import { AI_FILENAME_KEYWORDS } from './services/heuristics.js';
import { logger } from './services/logger.js';
import fs from 'fs';

// In test mode, do NOT load .env so the test-controlled env vars are used
if (process.env.NODE_ENV !== 'test') {
    dotenv.config();
} else {
    logger.info('[Test mode] Skipping .env loading.');
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Groq API Key Check (non-fatal) ---
// Server starts regardless — Groq-dependent endpoints return a descriptive error.
if (!process.env.GROQ_API_KEY) {
    logger.warn('GROQ_API_KEY is not set. Text/image analysis via Groq will be unavailable.');
    logger.warn('   The server will still serve static files and Python ML APIs.');
    logger.warn('   Set GROQ_API_KEY in .env to enable full analysis features.');
    process.env.GROQ_API_KEY = ''; // ensure it's defined (empty) so groq-sdk doesn't throw on construction
}

const app = express();
const PORT = process.env.PORT || 3001;

// Middlewares
app.use(compression()); // gzip all responses — smaller payloads, faster loads
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", 'https://fonts.googleapis.com', "'unsafe-inline'"],
            fontSrc: ['https://fonts.gstatic.com'],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", 'data:'],
            connectSrc: ["'self'"],
            baseUri: ["'self'"],
            frameAncestors: ["'none'"],
            objectSrc: ["'none'"],
        },
    },
    hidePoweredBy: true,
    crossOriginResourcePolicy: { policy: 'same-origin' },
    permissionsPolicy: {
        directives: {
            camera: [],
            microphone: [],
            geolocation: [],
            usb: [],
            bluetooth: [],
            midi: [],
            'sync-xhr': [],
            accelerometer: [],
            gyroscope: [],
            magnetometer: [],
            payment: [],
            fullscreen: [],
        },
    },
}));
app.use(cors({ origin: false })); // Same-origin only — frontend is served by Express

// HTTPS redirect — skip for localhost dev and already-secure requests
app.use((req, res, next) => {
    if (req.secure || req.headers['x-forwarded-proto'] === 'https') return next();
    const host = req.headers.host || '';
    if (host.startsWith('localhost') || host.startsWith('127.0.0.1')) return next();
    res.redirect(301, `https://${host}${req.originalUrl}`);
});

app.use(express.json({ limit: '10mb' })); // Prevent huge payload DoS
app.use(hpp()); // Prevent HTTP Parameter Pollution attacks
app.use(morgan('short')); // Minimal request logging
app.use(express.static('public', { maxAge: '7d' })); // Cache static assets 7 days

// --- Rate Limiting ---
// Protect against bot spam and API cost drain by limiting IPs to 60 requests per hour.
// Image analysis takes ~30s, so 60/hr (≈1/min) leaves room for interactive use.
const apiLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour window
    max: 60, // limit each IP to 60 requests per windowMs
    message: { error: 'Too many requests from this IP. Please try again after an hour.' },
    standardHeaders: true,
    legacyHeaders: false,
});

// Apply rate limiting specifically to the analysis endpoints
app.use('/api/analyze/', apiLimiter);

// Stricter rate limiter for image analysis (costs real API $$)
// 15 uploads/hr per IP is plenty for interactive use
const imageLimiter = rateLimit({
    windowMs: 60 * 60 * 1000,
    max: 15,
    message: { error: 'Too many image uploads from this IP. Please try again after an hour.' },
    standardHeaders: true,
    legacyHeaders: false,
});

// Rate limiter for the local text detector (cheap, but prevent spam)
const detectTextLimiter = rateLimit({
    windowMs: 60 * 60 * 1000,
    max: 120,
    message: { error: 'Too many requests from this IP. Please try again after an hour.' },
    standardHeaders: true,
    legacyHeaders: false,
});

// Configure multer for image uploads with a strict 5MB limit and MIME type filtering
// GIF and WebP are rejected: Groq vision and Gemini cannot process them, which caused
// every GIF/WebP upload to silently degrade to an "unable to analyze" result.
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const upload = multer({ 
    dest: UPLOADS_DIR,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5 Megabytes limit
    fileFilter: (req, file, cb) => {
        // Strict MIME type checking: only allow JPEG and PNG
        const allowedTypes = ['image/jpeg', 'image/png'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Invalid file type. Only JPG and PNG are supported.'));
        }
    }
});

// --- Background Cleanup Job ---
// Automatically clean up any stuck files in the uploads folder older than 1 hour.
// `.unref()` allows the process to exit cleanly even if this timer is still active.
const cleanupTimer = setInterval(() => {
    fs.readdir(UPLOADS_DIR, (err, files) => {
        if (err) return;
        const now = Date.now();
        files.forEach(file => {
            const filePath = path.join(UPLOADS_DIR, file);
            fs.stat(filePath, (err, stats) => {
                if (err) return;
                if (now - stats.mtimeMs > 3600000) {
                    fs.unlink(filePath, unlinkErr => {
                        if (!unlinkErr) logger.info(`Background cleanup: deleted stale file ${file}`);
                    });
                }
            });
        });
    });
}, 3600000);
cleanupTimer.unref();

// Magic byte signatures for allowed image types (JPEG/PNG only — see multer note)
const IMAGE_MAGIC_BYTES = {
    jpeg: [[0xFF, 0xD8, 0xFF]],
    png: [[0x89, 0x50, 0x4E, 0x47]],
};

function validateImageMagicBytes(buffer) {
    const header = buffer.slice(0, 12);
    for (const [fmt, sigs] of Object.entries(IMAGE_MAGIC_BYTES)) {
        for (const sig of sigs) {
            if (sig.every((b, i) => header[i] === b)) return fmt;
        }
    }
    return null;
}

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
// ── Face API known identities ──
// Returns the list of players the local Face API can recognize.
app.get('/api/face/known', async (req, res) => {
    try {
        const faceRes = await fetch('http://127.0.0.1:8002/health', { signal: AbortSignal.timeout(3000) });
        if (faceRes.ok) {
            const data = await faceRes.json();
            res.json({
                available: true,
                known_faces_count: data.known_faces_count,
                players: [
                    'cristiano ronaldo', 'de bruyne', 'haaland', 'harry kane',
                    'jude bellingham', 'lewandowski',
                    'lionel messi', 'mbappe', 'michael olise', 'neymar',
                    'salah', 'vinicius jr'
                ]
            });
        } else {
            res.json({ available: false, known_faces_count: 0, players: [] });
        }
    } catch {
        res.json({ available: false, known_faces_count: 0, players: [] });
    }
});

// ── Standalone ZeroGPT-style AI text detector ──
// Does NOT require GROQ_API_KEY. Uses local statistical analysis only.
app.post('/api/detect/text', detectTextLimiter, (req, res) => {
    let { text } = req.body;
    if (!text) return res.status(400).json({ error: 'Text is required' });
    text = sanitize(text);
    if (!text) return res.status(400).json({ error: 'Text is required' });
    const result = detectText(text);
    res.json(result);
});

const MAX_TEXT_LENGTH = 15000;

function sanitize(str) {
    // eslint-disable-next-line no-control-regex -- control-char stripping is intentional
    return str.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '').replace(/<[^>]*>/g, '').trim();
}

app.post('/api/analyze/text', async (req, res) => {
    try {
        let { text } = req.body;
        if (!text) return res.status(400).json({ error: 'Text is required' });
        text = sanitize(text);
        if (!text) return res.status(400).json({ error: 'Text is required' });
        if (text.length > MAX_TEXT_LENGTH) return res.status(400).json({ error: `Text exceeds ${MAX_TEXT_LENGTH} character limit` });
        
        const result = await analyzeContent(text);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        logger.error('Error analyzing text:', error);
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a minute and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze text' });
        }
    }
});

app.post('/api/analyze/url', async (req, res) => {
    try {
        let { url } = req.body;
        if (!url) return res.status(400).json({ error: 'URL is required' });
        url = sanitize(url);
        if (!url) return res.status(400).json({ error: 'URL is required' });
        if (url.length > 5000) return res.status(400).json({ error: 'URL exceeds 5000 character limit' });
        try {
            const proto = new URL(url).protocol;
            if (proto !== 'http:' && proto !== 'https:') {
                return res.status(400).json({ error: 'Only http/https URLs are supported' });
            }
        } catch {
            return res.status(400).json({ error: 'Invalid URL provided' });
        }

        const text = await scrapeUrl(url);
        const result = await analyzeContent(text);
        res.json(applyTrustedBoost(result));
    } catch (error) {
        logger.error('Error analyzing URL:', error);
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a minute and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze URL' });
        }
    }
});

// Filename keywords that strongly indicate AI-generated images.
// Shared strict list from heuristics.js — only unambiguous AI tool names.
function checkAiFilename(originalName) {
    if (!originalName) return null;
    const lower = originalName.toLowerCase().replace(/[^a-z0-9]/g, ' ');
    for (const kw of AI_FILENAME_KEYWORDS) {
        const normalized = kw.toLowerCase().replace(/[^a-z0-9]/g, ' ');
        if (lower.includes(normalized)) return kw;
    }
    return null;
}

app.post('/api/analyze/image', imageLimiter, upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Image is required' });
    
    const imagePath = req.file.path;
    const originalName = req.file.originalname || '';
    
    try {
        const imageData = await fs.promises.readFile(imagePath);

        // Validate magic bytes — client-supplied MIME type is untrustworthy
        const detected = validateImageMagicBytes(imageData);
        if (!detected) {
            await fs.promises.unlink(imagePath).catch(() => {});
            return res.status(400).json({ error: 'Invalid or corrupted image file. Only JPEG and PNG are accepted.' });
        }

        const base64 = imageData.toString('base64');
        // Use validated format from magic bytes, NOT client-supplied MIME
        const MIME_MAP = { jpeg: 'image/jpeg', png: 'image/png' };
        const mimeType = MIME_MAP[detected] || req.file.mimetype;

        // Check if the filename itself reveals AI origin
        const aiKeywordMatch = checkAiFilename(originalName);
        if (aiKeywordMatch) {
            logger.warn(`AI keyword detected in filename: "${originalName}" (matched: "${aiKeywordMatch}") — returning immediate fake verdict.`);
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

        // Pass optional user context (e.g. "this is an AI generated photo of me")
        const userContext = req.body.context || '';
        const result = await analyzeImage(base64, mimeType, userContext, originalName);
        
        res.json(applyTrustedBoost(result));
    } catch (error) {
        logger.error('Error analyzing image:', error);
        if (error?.status === 429) {
            res.status(429).json({ error: 'API rate limit reached. Please wait a moment and try again.' });
        } else {
            res.status(500).json({ error: 'Failed to analyze image' });
        }
    } finally {
        // Clean up temp file safely
        try {
            await fs.promises.unlink(imagePath);
        } catch (unlinkError) {
            logger.error('Error deleting temp image file:', unlinkError);
        }
    }
});

// --- 404 Handler ---
// Return JSON for unknown API routes, fall through to static for non-API
app.use('/api/*', (req, res) => {
    res.status(404).json({ error: `Not found: ${req.method} ${req.originalUrl}` });
});

// --- Global Error Handler ---
// Catch Multer limit errors and prevent HTML crash pages
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large. Maximum size is 5MB.' });
        }
        if (err.code === 'LIMIT_UNEXPECTED_FILE') {
            return res.status(400).json({ error: err.field || 'Invalid file uploaded. Please upload a valid image (JPG, PNG).' });
        }
        return res.status(400).json({ error: `Upload error: ${err.message}` });
    } else if (err) {
        logger.error('Global Error:', err);
        return res.status(500).json({ error: 'An unexpected server error occurred.' });
    }
    next();
});

function startServer(port) {
    const server = app.listen(port, () => {
        const actualPort = server.address().port;
        logger.info(`Server running on http://localhost:${actualPort}`);
    });

    server.on('error', async (err) => {
        if (err.code === 'EADDRINUSE') {
            logger.warn(`Port ${port} is busy. Killing the old process and retrying...`);
            const portNum = parseInt(port, 10);
            if (portNum > 0 && portNum <= 65535) {
                exec(`for /f "tokens=5" %a in ('netstat -aon ^| findstr :${portNum} ^| findstr LISTENING') do taskkill /F /PID %a`, { shell: 'cmd.exe' }, () => {
                    setTimeout(() => startServer(port), 1500);
                });
            } else {
                setTimeout(() => startServer(port), 3000);
            }
        } else {
            logger.error('Server error:', err);
        }
    });
}

// Python ML services (face_api.py, animal_api.py, python_api.py) are NOT auto-spawned.
// Run them manually if needed: python services/face_api.py, etc.

// Only start when run directly (not imported as a module)
// Decode URL-encoded path (e.g., %20 for spaces on Windows) before comparing
const decodedPath = decodeURIComponent(import.meta.url);
const isMainModule = process.argv[1] && (decodedPath === `file:///${process.argv[1].replace(/\\/g, '/')}` || !process.argv[1]);
if (isMainModule || process.env.START_SERVER === '1') {
    startServer(PORT);
}

export default app;
