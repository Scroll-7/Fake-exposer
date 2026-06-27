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
import { scrapeUrl } from './services/scraper.js';
import fs from 'fs';

// In test mode, do NOT load .env so the test-controlled env vars are used
if (process.env.NODE_ENV !== 'test') {
    dotenv.config();
} else {
    console.log('[Test mode] Skipping .env loading.');
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Groq API Key Check (non-fatal) ---
// Server starts regardless — Groq-dependent endpoints return a descriptive error.
if (!process.env.GROQ_API_KEY) {
    console.warn("⚠️ GROQ_API_KEY is not set. Text/image analysis via Groq will be unavailable.");
    console.warn("   The server will still serve static files and Python ML APIs.");
    console.warn("   Set GROQ_API_KEY in .env to enable full analysis features.");
    process.env.GROQ_API_KEY = ''; // ensure it's defined (empty) so groq-sdk doesn't throw on construction
}

const app = express();
const PORT = process.env.PORT || 3001;

// Middlewares
app.use(compression()); // gzip all responses — smaller payloads, faster loads
app.use(helmet());
app.use(cors({ origin: false })); // Same-origin only — frontend is served by Express
app.use(express.json({ limit: '10mb' })); // Prevent huge payload DoS
app.use(hpp()); // Prevent HTTP Parameter Pollution attacks
app.use(morgan('combined')); // Detailed access logging for security audits
app.use(express.static('public', { maxAge: '7d' })); // Cache static assets 7 days

// --- Rate Limiting ---
// Protect against bot spam and API cost drain by limiting IPs to 60 requests per hour.
// Image analysis takes ~30s, so 60/hr (≈1/min) leaves room for interactive use.
const apiLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour window
    max: 60, // limit each IP to 60 requests per windowMs
    message: { error: "Too many requests from this IP. Please try again after an hour." },
    standardHeaders: true,
    legacyHeaders: false,
});

// Apply rate limiting specifically to the analysis endpoints
app.use('/api/analyze/', apiLimiter);

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
                        if (!unlinkErr) console.log(`🧹 Background cleanup: deleted stale file ${file}`);
                    });
                }
            });
        });
    });
}, 3600000);
cleanupTimer.unref();

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

        // Pass optional user context (e.g. "this is an AI generated photo of me")
        const userContext = req.body.context || '';
        const result = await analyzeImage(base64, mimeType, userContext, originalName);
        
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
        console.error("Global Error:", err);
        return res.status(500).json({ error: 'An unexpected server error occurred.' });
    }
    next();
});

function startServer(port) {
    const server = app.listen(port, () => {
        const actualPort = server.address().port;
        console.log(`Server running on http://localhost:${actualPort}`);
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

// --- Auto-Training Check ---
// If any model file is missing, automatically run auto_train.py in the background.
// The server boots immediately and starts serving — ML endpoints gracefully degrade
// until training finishes and the Python APIs reload the new model files.
const MODEL_FILES = [
    { path: path.join(__dirname, 'fake_news_model.pkl'),      label: 'Text fake-news model' },
    { path: path.join(__dirname, 'fake_news_vectorizer.pkl'), label: 'Text vectorizer' },
    { path: path.join(__dirname, 'animal_model.onnx'),        label: 'Image classifier model' },
];

const missingModels = MODEL_FILES.filter(m => !fs.existsSync(m.path));

if (missingModels.length > 0) {
    console.log('\n🤖 Auto-Trainer: The following model files are missing:');
    missingModels.forEach(m => console.log(`   • ${m.label}  (${path.basename(m.path)})`));
    console.log('🤖 Auto-Trainer: Starting auto_train.py in the background...');
    console.log('🤖 Auto-Trainer: The server is live — ML features will activate once training completes.\n');

    const trainer = spawn('python', ['auto_train.py'], {
        cwd: __dirname,
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    trainer.stdout.on('data', (data) => {
        data.toString().split('\n').filter(Boolean).forEach(line =>
            console.log(`[Auto-Trainer] ${line}`)
        );
    });
    trainer.stderr.on('data', (data) => {
        data.toString().split('\n').filter(Boolean).forEach(line =>
            console.error(`[Auto-Trainer ERR] ${line}`)
        );
    });
    trainer.on('close', (code) => {
        if (code === 0) {
            console.log('\n✅ Auto-Trainer: All models trained successfully! Restarting Python APIs...');
        } else {
            console.error(`\n❌ Auto-Trainer: Training exited with code ${code}. Check logs above.`);
        }
        // Restart Python APIs so they pick up the freshly trained model files
        textApiProcess.kill();
        imageApiProcess.kill();
        const newTextApi = spawn('python', ['python_api.py'], { cwd: __dirname });
        newTextApi.stdout.on('data', (d) => console.log(`[ML Text API] ${d.toString().trim()}`));
        newTextApi.stderr.on('data', (d) => console.error(`[ML Text API Error] ${d.toString().trim()}`));
        const newImageApi = spawn('python', ['animal_api.py'], { cwd: __dirname });
        newImageApi.stdout.on('data', (d) => console.log(`[ML Image API] ${d.toString().trim()}`));
        newImageApi.stderr.on('data', (d) => console.error(`[ML Image API Error] ${d.toString().trim()}`));
        faceApiProcess.kill();
        const newFaceApi = spawn('python', ['face_api.py'], { cwd: __dirname });
        newFaceApi.stdout.on('data', (d) => console.log(`[Face API] ${d.toString().trim()}`));
        newFaceApi.stderr.on('data', (d) => console.error(`[Face API Error] ${d.toString().trim()}`));
    });
} else {
    console.log('✅ Auto-Trainer: All model files present — no training needed.');
}

// --- Start Python ML APIs Automatically ---
// Skip spawning in test mode to keep integration tests fast
let textApiProcess = null, imageApiProcess = null, faceApiProcess = null;
if (process.env.NODE_ENV !== 'test') {
    textApiProcess = spawn('python', ['python_api.py'], { cwd: __dirname });
    textApiProcess.stdout.on('data', (data) => console.log(`[ML Text API] ${data.toString().trim()}`));
    textApiProcess.stderr.on('data', (data) => console.error(`[ML Text API Error] ${data.toString().trim()}`));

    imageApiProcess = spawn('python', ['animal_api.py'], { cwd: __dirname });
    imageApiProcess.stdout.on('data', (data) => console.log(`[ML Image API] ${data.toString().trim()}`));
    imageApiProcess.stderr.on('data', (data) => console.error(`[ML Image API Error] ${data.toString().trim()}`));

    faceApiProcess = spawn('python', ['face_api.py'], { cwd: __dirname });
    faceApiProcess.stdout.on('data', (data) => console.log(`[Face API] ${data.toString().trim()}`));
    faceApiProcess.stderr.on('data', (data) => console.error(`[Face API Error] ${data.toString().trim()}`));
} else {
    console.log('[Test mode] Skipping Python API spawning.');
}

const API_HEALTH_URLS = [
    { name: 'Text ML', url: 'http://127.0.0.1:8000/health' },
    { name: 'Image ML', url: 'http://127.0.0.1:8001/health' },
    { name: 'Face ID', url: 'http://127.0.0.1:8002/health' },
];

async function checkApiHealth() {
    for (const api of API_HEALTH_URLS) {
        try {
            const res = await fetch(api.url, { signal: AbortSignal.timeout(3000) });
            if (res.ok) {
                const data = await res.json();
                console.log(`[Health] ${api.name} API OK: ${JSON.stringify(data)}`);
            } else {
                console.error(`[Health] ${api.name} API returned ${res.status}`);
            }
        } catch {
            console.warn(`[Health] ${api.name} API unreachable`);
        }
    }
}

// Health check every 60 seconds, first check after 5s
// Skip in test mode so the process exits cleanly after integration tests
if (process.env.NODE_ENV !== 'test') {
    setTimeout(async () => {
        await checkApiHealth();
        const healthInterval = setInterval(checkApiHealth, 60000);
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
const isMainModule = process.argv[1] && (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}` || !process.argv[1]);
if (isMainModule || process.env.START_SERVER === '1') {
    startServer(PORT);
}

export default app;
