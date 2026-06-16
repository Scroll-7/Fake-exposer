import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
dotenv.config();

// Support a single key or a comma-separated list of keys
const keysString = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const apiKeys = keysString.split(',').map(k => k.trim()).filter(k => k.length > 0);

if (apiKeys.length === 0) {
    console.warn("[Warning] No GEMINI API keys found in environment variables!");
}

const FALLBACK_MODELS = [
    'gemini-2.0-flash',        // Primary (if quota refreshed)
    'gemini-1.5-flash-latest', // Standard 1.5 flash
    'gemini-1.5-pro-latest',   // Pro version
    'gemini-pro-vision',       // Older stable vision model
    'gemini-3.5-flash',        // The one that had 20 limits
    'gemini-1.5-flash-8b'      // Very fast, light fallback
];

// Automatically try multiple keys AND multiple models if quota hits 429
async function executeWithFallback(promptConfig, isImage = false) {
    for (let i = 0; i < apiKeys.length; i++) {
        const key = apiKeys[i];
        const genAI = new GoogleGenerativeAI(key);
        const keyLabel = `Key-${i + 1}`;

        for (const modelName of FALLBACK_MODELS) {
            try {
                console.log(`[Gemini] Attempting ${modelName} on ${keyLabel}...`);
                const model = genAI.getGenerativeModel({
                    model: modelName,
                    generationConfig: { responseMimeType: 'application/json' }
                });
                return await model.generateContent(promptConfig);
            } catch (err) {
                const status = err?.status;
                // If it's a quota limit (429) or model not found (404), try the next model on this key
                if (status === 429 || status === 404) {
                    console.log(`[Gemini] ${modelName} on ${keyLabel} failed (${status}).`);
                    continue; 
                }
                // Real failure (e.g. invalid key auth)
                throw err;
            }
        }
        console.log(`[Gemini] ALL models failed for ${keyLabel}. Switching to next API key...`);
    }
    // If all keys and all models fail
    throw new Error('All AI models across ALL provided API keys have exhausted their quota.');
}

const ANALYSIS_SCHEMA = `
    You must respond ONLY with a valid JSON object in this exact format:
    {
      "credibility_score": <number 0-100>,
      "verdict": "<string, e.g., 'Likely Fake', 'Highly Credible'>",
      "bias": "<string, e.g., 'Right-Wing', 'Center', 'None'>",
      "sentiment": "<string, e.g., 'Fear-inducing', 'Neutral'>",
      "red_flags": ["<string>", ...],
      "green_flags": ["<string>", ...],
      "summary": "<string, short explanation of verdict>",
      "recommendations": ["<string>", ...],
      "is_trusted_source": <true or false>,
      "trusted_source_name": "<string, name of the identified trusted source, or empty string if not trusted>"
    }`;

const SOURCE_INSTRUCTIONS = `
    IMPORTANT — Source Reputation Detection (Dynamic Verification):
    Try to identify WHO published or posted this content (the author, outlet, organization, or account).
    Use your visual, contextual, and internal knowledge to assess the source's credibility.

    How to determine if a source is "Highly Trusted":
    1. Visual Verification (Images): If you are analyzing a screenshot, look closely at the profile. If the account has a "verified badge" (e.g., a blue checkmark on X/Twitter, Instagram, TikTok) AND the handle/name matches a known public figure, journalist, or organization, trust it.
    2. Contextual Recognition: Recognize world-class, reputable sources across various domains. Examples include (but are not limited to):
        - Global News: BBC, Reuters, AP News, AFP, The Guardian, NYT, Washington Post, Al Jazeera, Bloomberg, WSJ.
        - Sports & Transfers: Fabrizio Romano, David Ornstein, Florian Plettenberg, Shams Charania, Adrian Wojnarowski, Sky Sports, ESPN.
        - Tech & Gaming: Marques Brownlee (MKBHD), IGN, The Verge, TechCrunch, Jason Schreier, Geoff Keighley.
        - Science & Health: WHO, CDC, NASA, Nature, Science Magazine, Neil deGrasse Tyson, Andrew Huberman.
        - Finance & Politics: Financial Times, The Economist, official government verified accounts.

    CRITICAL RULE FOR SCREENSHOTS: If the image visually contains a blue checkmark/verified badge next to the account name, you MUST set "is_trusted_source" to true. DO NOT fact-check the actual claim in the post to determine if the screenshot is a photoshop. Even if you know the claim is factually false (e.g., a fake transfer), assume the screenshot is authentic for the purpose of the "is_trusted_source" flag if the visual badge is present.`;

// Analyze plain text content
export async function analyzeContent(text) {
    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, and source reputation researcher.
    Analyze the following text for signs of fake news, misinformation, bias, and manipulation.
    ${SOURCE_INSTRUCTIONS}

    Text to analyze:
    "${text}"

    ${ANALYSIS_SCHEMA}
    `;

    const result = await executeWithFallback(prompt, false);
    let responseText = result.response.text();
    responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
    return JSON.parse(responseText);
}

// Analyze an image in ONE single Gemini call (no separate OCR step — saves quota)
export async function analyzeImage(imageBase64, mimeType) {
    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, and source reputation researcher.
    You are given an image (screenshot, photo, or document).
    First, read and extract all the text visible in the image.
    Then analyze that text for signs of fake news, misinformation, bias, and manipulation.
    ${SOURCE_INSTRUCTIONS}

    ${ANALYSIS_SCHEMA}
    `;

    const result = await executeWithFallback([
        { inlineData: { data: imageBase64, mimeType } },
        prompt
    ], true);
    let responseText = result.response.text();
    responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
    return JSON.parse(responseText);
}
