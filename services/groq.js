import Groq from 'groq-sdk';
import dotenv from 'dotenv';
import { detectAiImage, computeAiOverride } from './aiDetector.js';
dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

async function searchWeb(query) {
    try {
        const response = await fetch('https://html.duckduckgo.com/html/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            body: 'q=' + encodeURIComponent(query),
            signal: AbortSignal.timeout(10000) // 10s timeout
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

const ANALYSIS_SCHEMA = `
    You must respond ONLY with a valid JSON object in this exact format. Do not include markdown formatting or extra text:
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

const QUERY_TOLERANCE_INSTRUCTIONS = `
    IMPORTANT — Casual Query Tolerance:
    Users will often submit short, casual search queries or rumors (e.g., "mbape to real madrid"). 
    DO NOT flag minor spelling mistakes (e.g., "mbape" instead of "Mbappé"), lack of capitalization, or missing punctuation as "red flags". 
    Focus entirely on the factual accuracy of the core claim, not the user's grammar. If a user is asking about a known rumor or fact, evaluate the fact itself instead of criticizing the text format.
`;

export async function analyzeContent(text) {
    // Truncate query for DuckDuckGo to avoid massive payload errors (DuckDuckGo expects short queries)
    const searchQuery = text.length > 300 ? text.substring(0, 300) : text;
    const searchResults = await searchWeb(searchQuery);

    // Truncate text for Llama model to prevent 413 Payload Too Large / Token Rate Limits
    // The TPM limit is 12000 tokens. Safely truncate to 15,000 characters.
    let contentToAnalyze = text;
    if (contentToAnalyze.length > 15000) {
        contentToAnalyze = contentToAnalyze.substring(0, 15000) + "\n...[Content truncated due to length]...";
    }

    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, and source reputation researcher.
    Analyze the following text for signs of fake news, misinformation, bias, and manipulation.
    ${SOURCE_INSTRUCTIONS}
    ${QUERY_TOLERANCE_INSTRUCTIONS}

    --- LIVE WEB SEARCH CONTEXT (USE THIS TO FACT CHECK) ---
    The following are live search results from the internet regarding the query. 
    Use these facts to determine if the user's claim is true or false, especially for recent events:
    - ${searchResults}
    --------------------------------------------------------

    Text to analyze:
    "${contentToAnalyze}"

    ${ANALYSIS_SCHEMA}
    `;

    try {
        const completion = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.3-70b-versatile",
            response_format: { type: "json_object" },
            temperature: 0.1,
        });

        const responseText = completion.choices[0]?.message?.content || '{}';
        return JSON.parse(responseText);
    } catch (error) {
        console.error("Groq text analysis error:", error);
        throw error;
    }
}

export async function analyzeImage(imageBase64, mimeType, userContext = '') {
    // Groq enforces a 4 MB limit on base64-encoded images
    const base64SizeBytes = Math.ceil(imageBase64.length * 3 / 4);
    const MAX_BASE64_BYTES = 4 * 1024 * 1024; // 4 MB
    if (base64SizeBytes > MAX_BASE64_BYTES) {
        throw new Error(`Image too large (${(base64SizeBytes / (1024 * 1024)).toFixed(1)} MB). Maximum is 4 MB. Please resize or compress the image.`);
    }

    const dataUrl = `data:${mimeType};base64,${imageBase64}`;

    // ── PRE-STEP: Dedicated AI image detection (Gemini Vision forensics) ──
    // This runs a specialized 10-point AI artifact detection BEFORE the general LLM analysis.
    // If Gemini says the image is AI-generated with high confidence, we hard-override the score.
    const aiDetection = await detectAiImage(imageBase64, mimeType);
    const aiOverride = computeAiOverride(aiDetection);

    // ── STEP 1: Rich visual description for context + search ──
    let imageDescription = '';
    console.log(`Starting Groq step 1 (Description)...`);
    const descStartTime = Date.now();
    try {
        const userContextNote = userContext
            ? `\n\nIMPORTANT — The user says this image: "${userContext}". Keep this in mind.`
            : '';

        const descCompletion = await Promise.race([
            groq.chat.completions.create({
                messages: [
                    {
                        role: "user",
                        content: [
                            {
                                type: "text",
                                text: `You are a visual forensics expert. Analyze this image thoroughly and answer ALL of the following questions:${userContextNote}

--- GENERAL AI / MANIPULATION DETECTION ---
1. Does this look AI-generated or AI-edited? Look for:
   - Unnaturally smooth or waxy skin texture (Note: account for bodybuilding stage oil/spray tan which looks waxy)
   - Exaggerated or impossible muscle definition / body proportions (Note: extreme conditioning in professional bodybuilders is real, consider context)
   - Inconsistent lighting between body parts (e.g. face lit differently from torso)
   - Blurring or smearing at body edges / background seams
   - Unnatural background consistency or bokeh patterns
   - Finger/hand deformities or repetition
   - Skin color or tone mismatches between face and body
2. Does the body look naturally proportioned, or are features (muscles, face, limbs) exaggerated beyond what is humanly typical for the person's frame?
3. Are there any blending artifacts or unnatural boundary transitions between body parts or between the person and the background?

--- IDENTITY CHECK (SPORTS IMAGES ONLY) ---
4. If the person is wearing a sports jersey: name who you think they are (only if you are highly confident — if uncertain, say "unidentified person"). Also state what team jersey they wear, and whether that person actually plays for that team.
5. If you are NOT confident about the person's identity, say so explicitly. DO NOT guess a famous person's name unless you are at least 80% certain.

--- SUMMARY ---
6. Write a 2-3 sentence summary of what the image shows and whether it appears authentic or manipulated.

Be very specific and honest about uncertainty.`
                            },
                            { type: "image_url", image_url: { url: dataUrl } }
                        ]
                    }
                ],
                model: "meta-llama/llama-4-scout-17b-16e-instruct",
                temperature: 0.1,
                max_tokens: 500,
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Groq Description Timeout')), 45000))
        ]);
        imageDescription = descCompletion.choices[0]?.message?.content || '';
        console.log(`Groq Description completed in ${Date.now() - descStartTime}ms. Output length: ${imageDescription.length}`);
    } catch (err) {
        console.error(`Groq Description failed after ${Date.now() - descStartTime}ms:`, err.message);
        // Continue without description — fall back to visual-only analysis
    }

    // ── STEP 2: Web search to fact-check the image content ──
    let searchResults = "No recent news found.";
    if (imageDescription) {
        searchResults = await searchWeb(imageDescription);
    }

    // ── STEP 3: Full analysis with both visual + factual context ──
    const userContextSection = userContext
        ? `--- USER-PROVIDED CONTEXT (TREAT AS A STRONG SIGNAL) ---
    The person who uploaded this image says: "${userContext}"
    If the user admits the image is AI-generated, edited, or fake, TRUST THEM and reflect this in your verdict and score.
    --------------------------------------------------------`
        : '';

    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, visual forensics expert, and source reputation researcher.
    You are given an image (screenshot, photo, or document).

    ${userContextSection}

    STEP 1: Determine the image type:
    - Type A: Contains readable text (social media post, news headline, article, caption)
    - Type B: Primarily a photograph or graphic without significant text

    STEP 2A — If TYPE A (text-based):
      Extract all visible text and analyze it for fake news, misinformation, bias, and manipulation.
      ${SOURCE_INSTRUCTIONS}
      ${QUERY_TOLERANCE_INSTRUCTIONS}

    STEP 2B — If TYPE B (photo/graphic):
      You MUST perform ALL of the following checks:

      ── AI GENERATION / MANIPULATION CHECK (CRITICAL) ──
      Look for these signs of AI editing or generation:
      • Unnaturally smooth, waxy, or plastic-looking skin (IMPORTANT: Do not penalize if the person is clearly a professional bodybuilder wearing stage oil/tan)
      • Exaggerated muscle definition or body proportions that look physically impossible or inconsistent with the person's face/frame (IMPORTANT: Real professional bodybuilders on stage have extreme, shredded muscle definition)
      • Lighting inconsistencies — e.g. face lit from one direction, body from another
      • Blurry, smeared, or artificially sharpened body edges
      • Background that looks overly uniform, smeared, or mismatched to the foreground
      • Skin color or texture differences between face and body (suggesting a face swap or body replacement)
      • Repeated or warped features (extra fingers, asymmetric limbs)
      • Unusual sharpness or detail in body/muscle areas compared to the face or surroundings
      If ANY of these signs are present, the image is LIKELY AI-GENERATED OR EDITED.

      ── IDENTITY CAUTION RULE ──
      - If the person in the image is NOT a well-known public figure you are highly confident about (80%+), say "unidentified person" — do NOT guess a celebrity name.
      - Many real people look slightly similar to athletes or celebrities. Similarity is NOT identification.
      - If a person's muscles/body looks AI-enhanced compared to a realistic build for their face/frame, say so.

      ── SPORTS JERSEY FACT-CHECK (only if applicable) ──
      If the person IS a highly-recognized athlete in a sports jersey:
      1. Confirm their identity (only if 80%+ certain)
      2. Confirm what team jersey they wear
      3. Cross-reference: does this player actually play for that team?
      4. If there's a mismatch → score 5-20, verdict "Fake / Manipulated - Player Not at This Club"

      ── SCORING GUIDE ──
      • No AI artifacts + factually accurate → HIGH (80-100)
      • Subtle AI artifacts OR minor factual uncertainty → MEDIUM (40-70)
      • Clear AI artifacts (muscle/body enhancement, skin inconsistencies, etc.) → LOW (5-30), verdict "Likely AI-Generated / Edited"
      • User admitted it is AI-generated or fake → VERY LOW (5-15), verdict "Confirmed Fake / AI-Generated"
      • Wrong sports jersey for identified player → VERY LOW (5-20), verdict "Fake / Manipulated"
      • Visually authentic photo of unknown person with no suspicious context → HIGH (75-90)

    --- VISUAL ANALYSIS FROM STEP 1 ---
    ${imageDescription || 'No description available.'}

    --- LIVE WEB SEARCH CONTEXT ---
    ${searchResults}
    --------------------------------------------------------

    ${ANALYSIS_SCHEMA}
    `;

    try {
        console.log(`Starting Groq step 3 (Final Analysis)...`);
        const analysisStartTime = Date.now();
        const completion = await Promise.race([
            groq.chat.completions.create({
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            { type: "image_url", image_url: { url: dataUrl } }
                        ]
                    }
                ],
                model: "meta-llama/llama-4-scout-17b-16e-instruct",
                temperature: 0.1,
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Groq Analysis Timeout')), 45000))
        ]);
        console.log(`Groq Analysis completed in ${Date.now() - analysisStartTime}ms.`);

        let responseText = completion.choices[0]?.message?.content || '{}';
        // Clean markdown blocks if vision model ignores json_object format
        responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
        const llmResult = JSON.parse(responseText);

        // ── MERGE AI OVERRIDE: If Gemini detected AI with high confidence, hard-override the score ──
        if (aiOverride) {
            const artifactFlags = (aiOverride.ai_artifacts || []).map(a => `⚠️ AI artifact: ${a}`);
            return {
                ...llmResult,
                credibility_score: aiOverride.credibility_score,
                verdict: aiOverride.verdict,
                red_flags: [
                    ...artifactFlags,
                    ...(llmResult.red_flags || []),
                    `🤖 Gemini AI Detector: ${aiOverride.ai_probability}% probability of AI generation`
                ],
                green_flags: [],  // Clear green flags — the image is fake
                summary: aiOverride.ai_reasoning || llmResult.summary,
                ai_detection: {
                    probability: aiOverride.ai_probability,
                    verdict: aiDetection?.verdict,
                    confidence: aiDetection?.confidence,
                    artifacts: aiOverride.ai_artifacts,
                }
            };
        }

        return llmResult;
    } catch (error) {
        const detail = error?.error?.message || error?.message || 'Unknown error';
        console.error("Groq image analysis error:", detail, error);
        throw new Error(`Image analysis failed: ${detail}`);
    }
}
