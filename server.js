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
import { spawn } from 'child_process';

import { analyzeContent, analyzeImage } from './services/groq.js';
import { analyzeText as detectText } from './services/textDetector.js';
import { scrapeUrl } from './services/scraper.js';
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
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const upload = multer({ 
    dest: UPLOADS_DIR,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5 Megabytes limit
    fileFilter: (req, file, cb) => {
        // Strict MIME type checking: only allow JPEG, PNG, WEBP, and GIF
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'Invalid file type. Only JPG, PNG, WEBP, and GIF are allowed.'));
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

// Magic byte signatures for allowed image types
const IMAGE_MAGIC_BYTES = {
    jpeg: [[0xFF, 0xD8, 0xFF]],
    png: [[0x89, 0x50, 0x4E, 0x47]],
    gif: [[0x47, 0x49, 0x46, 0x38]],
    webp: [[0x52, 0x49, 0x46, 0x46]],
};

function validateImageMagicBytes(buffer) {
    const header = buffer.slice(0, 12);
    for (const [fmt, sigs] of Object.entries(IMAGE_MAGIC_BYTES)) {
        for (const sig of sigs) {
            if (sig.every((b, i) => header[i] === b)) return fmt;
        }
    }
    // WEBP: RIFF header at 0-3, WEBP at 8-11
    if (header[0] === 0x52 && header[1] === 0x49 && header[2] === 0x46 && header[3] === 0x46 &&
        header[8] === 0x57 && header[9] === 0x45 && header[10] === 0x42 && header[11] === 0x50) {
        return 'webp';
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
            return res.status(400).json({ error: 'Invalid or corrupted image file. Only JPEG, PNG, WEBP, and GIF are accepted.' });
        }

        const base64 = imageData.toString('base64');
        const mimeType = req.file.mimetype;

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
            return res.status(400).json({ error: err.field || 'Invalid file uploaded. Please upload a valid image (JPG, PNG, WEBP).' });
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
                const { exec } = await import('child_process');
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

// --- Auto-Training Check ---
// If any model file is missing, automatically run auto_train.py in the background.
// The server boots immediately and starts serving — ML endpoints gracefully degrade
// until training finishes and the Python APIs reload the new model files.
// Skipped in test mode to keep tests fast and avoid hanging on missing data files.
let textApiProcess = null, imageApiProcess = null, faceApiProcess = null;

if (process.env.NODE_ENV !== 'test') {
    const MODEL_FILES = [
        { path: path.join(__dirname, 'fake_news_model.pkl'),      label: 'Text fake-news model' },
        { path: path.join(__dirname, 'fake_news_vectorizer.pkl'), label: 'Text vectorizer' },
        { path: path.join(__dirname, 'animal_model.onnx'),        label: 'Image classifier model' },
    ];

    const missingModels = MODEL_FILES.filter(m => !fs.existsSync(m.path));

    if (missingModels.length > 0) {
        const trainer = spawn('python', ['auto_train.py'], {
            cwd: __dirname,
            stdio: 'ignore',
        });
        trainer.on('close', (code) => {
            if (code === 0) {
                try { textApiProcess.kill(); } catch {}
                try { imageApiProcess.kill(); } catch {}
                try { faceApiProcess.kill(); } catch {}
                textApiProcess = spawnPythonApi('ML Text API', 'python_api.py', 8000);
                imageApiProcess = spawnPythonApi('ML Image API', 'animal_api.py', 8001);
                faceApiProcess = spawnPythonApi('Face API', 'face_api.py', 8002);
            }
        });
    }
}

// --- Start Python ML APIs Automatically (quiet mode) ---
// Skip spawning in test mode to keep integration tests fast
function spawnPythonApi(name, script, port) {
    const proc = spawn('python', [script], { cwd: __dirname });
    proc.stdout.on('data', () => {}); // suppress
    proc.stderr.on('data', (data) => {
        // Only log actual errors, not startup info
        const s = data.toString().trim();
        if (/error|traceback|exception|fail|warn/i.test(s)) logger.error(`[${name}] ${s}`);
    });
    proc.on('exit', (code) => {
        if (code !== 0 && code !== null) {
            setTimeout(() => {
                const newProc = spawnPythonApi(name, script, port);
                if (name === 'ML Text API') textApiProcess = newProc;
                else if (name === 'ML Image API') imageApiProcess = newProc;
                else if (name === 'Face API') faceApiProcess = newProc;
            }, 3000);
        }
    });
    return proc;
}

if (process.env.NODE_ENV !== 'test') {
    textApiProcess = spawnPythonApi('ML Text API', 'python_api.py', 8000);
    imageApiProcess = spawnPythonApi('ML Image API', 'animal_api.py', 8001);
    faceApiProcess = spawnPythonApi('Face API', 'face_api.py', 8002);
} else {
    logger.info('[Test mode] Skipping Python API spawning.');
}

const API_HEALTH_URLS = [
    { name: 'Text ML', url: 'http://127.0.0.1:8000/health' },
    { name: 'Image ML', url: 'http://127.0.0.1:8001/health' },
    { name: 'Face ID', url: 'http://127.0.0.1:8002/health' },
];

async function checkApiHealth(retries = 3, delay = 5000) {
    for (const api of API_HEALTH_URLS) {
        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                const res = await fetch(api.url, { signal: AbortSignal.timeout(5000) });
                if (res.ok) break;
            } catch {
                if (attempt < retries) {
                    await new Promise(r => setTimeout(r, delay));
                }
            }
        }
    }
}

// Health check every 60 seconds, first check after 5s
// Skip in test mode so the process exits cleanly after integration tests
if (process.env.NODE_ENV !== 'test') {
    setTimeout(async () => {
        await checkApiHealth(6, 7000);
        const healthInterval = setInterval(() => checkApiHealth(1, 0), 60000);
        healthInterval.unref();
    }, 5000);
}

// Kill Python processes when Node.js shuts down
function killAll() {
    for (const p of [textApiProcess, imageApiProcess, faceApiProcess]) {
        try { p.kill(); } catch {}
    }
}
process.on('exit', killAll);
process.on('SIGINT', () => { killAll(); process.exit(); });

// Only start when run directly (not imported as a module)
// Decode URL-encoded path (e.g., %20 for spaces on Windows) before comparing
const decodedPath = decodeURIComponent(import.meta.url);
const isMainModule = process.argv[1] && (decodedPath === `file:///${process.argv[1].replace(/\\/g, '/')}` || !process.argv[1]);
if (isMainModule || process.env.START_SERVER === '1') {
    startServer(PORT);
}

export default app;
