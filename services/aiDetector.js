/**
 * AI Image Detector using Google Gemini Vision
 * Uses Gemini's strong visual analysis to detect AI-generated/edited images.
 * Returns { aiProbability: 0-100, isAI: boolean, reasoning: string, artifacts: string[] }
 */

// Rotate through available API keys
let keyIndex = 0;
function getNextGeminiKey() {
    const rawKeys = (process.env.GEMINI_API_KEYS || '').split(',').map(k => k.trim()).filter(Boolean);
    if (!rawKeys.length) return null;
    const key = rawKeys[keyIndex % rawKeys.length];
    keyIndex++;
    return key;
}

export async function detectAiImage(imageBase64, mimeType) {
    const apiKey = getNextGeminiKey();
    if (!apiKey) {
        console.warn('No GEMINI_API_KEYS set — skipping dedicated AI detection step.');
        return null;
    }

    try {
        const prompt = `You are a specialized AI image forensics detector with expertise in detecting subtle AI body enhancement edits AND AI-generated portraits. Your ONLY job is to determine if this image was AI-generated or AI-edited.

═══════════════════════════════════════════════
STEP 1: CLASSIFY THE IMAGE TYPE
═══════════════════════════════════════════════
First, identify which category this image falls into:
A) Professional bodybuilding competition (stage lights, posing trunks, audience, competition banner, heavy spray tan, formal posing)
B) Casual mirror selfie / gym selfie / bathroom selfie (person holding phone in a mirror, casual setting)
C) Portrait / Headshot / Passport Photo (person facing camera, mostly face/shoulders visible, often against a plain, neutral, or simple background)
D) Group photo / Crowd scene (multiple people, events, parties, background characters)
E) Politician, celebrity, or public figure in a highly dramatic or unusual situation
F) Other (street photo, sports action shot, etc.)

The category determines how strict you should be.

═══════════════════════════════════════════════
STEP 2: FORENSIC ANALYSIS (do ALL checks)
═══════════════════════════════════════════════

CHECK 1 — MUSCLE-TO-SKELETON RATIO (most important for body edits):
  → Look at the person's wrist/forearm width, collarbone width, jaw width, and neck thickness.
  → Now compare those bone-anchored measurements against the muscle volume on the arms, chest, shoulders, and abs.
  → If the muscles look disproportionately large or defined RELATIVE TO the visible bone structure, this is a PRIMARY indicator of AI muscle enhancement.
  → For example: if wrists look slender (normal) but upper arms and chest look extremely thick and defined — that mismatch is a red flag.

CHECK 2 — SKIN TEXTURE CONSISTENCY:
  → Compare the skin texture on the FACE (forehead, cheeks) vs the TORSO (chest, abs, arms).
  → Natural photos: same grain, pores, and noise across all skin areas.
  → AI-edited photos: torso skin often looks smoother, more "airbrushed", or more rendered than the face.
  → Check if muscle areas look more "plasticky" or "rendered" than the face.

CHECK 3 — LIGHTING & SHADOW GEOMETRY:
  → Identify where the key light source is coming from (look at shadows on the face).
  → Now check: do the muscle highlights and shadows on the torso follow the SAME direction?
  → AI muscle edits often add volume/shading inconsistent with the actual ambient light in the room.
  → Look for: muscles that look more "3D" or "lit from within" compared to the rest of the scene.

CHECK 4 — EDGE ARTIFACTS AT BODY CONTOURS:
  → Look carefully at the silhouette edges of the arms, shoulders, and torso.
  → Zoom into where skin meets the background — is there any halos, blurring, unnatural sharpening, or "glow" at the edges?
  → AI body edits often leave subtle edge artifacts where the muscle shape was digitally enlarged.

CHECK 5 — SKIN TONE MISMATCH:
  → Is the skin tone on the torso slightly more saturated, more orange/red/tan than the face?
  → Natural selfies have consistent skin tone everywhere.
  → AI body enhancements often subtly shift the torso color.

CHECK 6 — VEIN AND DETAIL HYPER-REALISM:
  → Are veins or muscle striations hyper-visible in a way that looks slightly "rendered" or "digitally sharpened"?
  → Real veins in photos look natural and slightly blurry at the edges. Digitally enhanced ones look too sharp or perfectly shaped.

CHECK 7 — BACKGROUND CONSISTENCY:
  → Does the background look natural, or does it show smearing, repetitive patterns, or blurring that doesn't match a real camera lens?

CHECK 8 — OVERALL GESTALT ("UNCANNY VALLEY"):
  → Does anything feel "off" about the body proportions, even if you can't pinpoint it?
  → Trust your instincts — if something feels too perfect or too impressive for the casual setting, flag it.

═══════════════════════════════════════════════
STEP 3: PORTRAIT-SPECIFIC CHECKS (If Category C)
═══════════════════════════════════════════════
If this is a portrait, headshot, or close-up of a face, AI generators (like Midjourney, DALL-E, Gemini, ThisPersonDoesNotExist) consistently produce:
  → A perfectly uniform grey/neutral/muted background.
  → Flawless, symmetrical lighting with "studio" quality that looks slightly too perfect.
  → Skin texture that lacks real-world flaws, blemishes, or asymmetric pores (often looks plastic or hyper-real).
  → Lack of real-world environmental context (no messy room, no outdoors).
  → Eyes that might have mismatched catchlights (reflections) or unnaturally perfect irises.
If a portrait looks like a "perfect passport photo" or "perfect professional headshot" with a uniform background and no context, treat it as HIGHLY suspicious of being AI-generated.

═══════════════════════════════════════════════
STEP 4: GROUP & CROWD CHECKS (If Category D)
═══════════════════════════════════════════════
→ Look closely at the faces of people in the background. AI models often generate mangled, melting, or featureless faces for background characters.
→ Count fingers and look at hands. Are hands merging into other people's bodies or clothing? Are there extra limbs?
→ Look at background text or signs. Are they written in a nonsensical alien language (common AI artifact)?

═══════════════════════════════════════════════
STEP 5: PUBLIC FIGURE CHECKS (If Category E)
═══════════════════════════════════════════════
→ Does the public figure look overly glossy, dramatized, or caricatured compared to real press photos?
→ IF this is a public figure in an unusual situation (e.g. Pope in a puffy jacket, politician getting arrested), it is almost certainly an AI deepfake. Flag it heavily.

═══════════════════════════════════════════════
STEP 6: CALIBRATED SCORING GUIDE
═══════════════════════════════════════════════

Use these EXACT scoring anchors depending on the category:

  0–15% (Real): Completely natural — realistic proportions, natural skin texture with pores, consistent tone, real-world context and imperfections.

  16–29% (Probably Real): Very minor suspicions — maybe slight smoothness in one area, but overall looks natural.

  30–49% (Suspicious / Possibly Edited): Something feels off — minor skin texture inconsistency, or minor edge softness. Not definitive but warrants a flag.

  50–69% (Likely AI-Edited or Generated): Clear signs in 2+ checks — AI-smooth skin, subtle lighting inconsistency, or a portrait that looks overly perfect and uniform without real-world flaws. Lean toward flagging.

  70–89% (Likely AI-Generated): Multiple strong indicators — obvious AI-smooth skin, uncanny valley feeling, mangled background faces in crowds, or a classic AI-generated studio portrait with flawless rendering and a completely uniform background.

  90–100% (Definitely AI): Unambiguous AI artifacts — impossible proportions, clear edge halos, face-body texture mismatch, rendered-looking elements, mismatched eye reflections, alien text in the background, or a quintessential AI-generated portrait with zero real-world context and hyper-real "plastic" perfection.

IMPORTANT: For Category C (Portrait / Headshot), be extremely strict. A huge percentage of modern fake images are simply AI-generated faces on plain backgrounds. If it looks like an AI generator made it (perfect lighting, perfect skin, neutral background), score it 80-100%. If the background is a perfectly uniform solid color (like flat grey), this is a massive red flag for an AI generation.

IMPORTANT: For Category B (casual mirror selfie), a person who is genuinely lean and athletic will STILL show:
  ✓ Some body fat visible (even at 10% body fat, you see a layer of skin over muscles)
  ✓ Slightly imperfect, asymmetric muscle shapes
  ✓ Natural skin texture with pores on the torso
  ✓ Muscles that match the bone structure proportionally
  ✗ NOT: Extremely shredded, deeply cut abs with near-zero visible body fat in a bathroom/bedroom setting
  ✗ NOT: Muscles that look "inflated" or "pumped" beyond what the skeleton could support

For Category A (bodybuilding competition): be lenient — scores should typically be 0–30% unless there are obvious AI artifacts beyond the expected extreme conditioning.

═══════════════════════════════════════════════
STEP 7: RESPOND WITH JSON ONLY
═══════════════════════════════════════════════

You MUST respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{
  "image_category": "<'Competition' | 'Mirror Selfie' | 'Other'>",
  "ai_probability": <integer 0-100>,
  "verdict": "<'AI-Generated' | 'Likely AI-Generated' | 'Suspicious' | 'Likely Real' | 'Real'>",
  "confidence": "<'High' | 'Medium' | 'Low'>",
  "artifacts_detected": ["<specific artifact 1>", "<specific artifact 2>", ...],
  "reasoning": "<3-4 sentence explanation referencing specific checks above>"
}`;

        const GEMINI_MODELS = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-1.5-flash-latest',
        ];

        let response = null;
        let usedModel = null;
        for (const model of GEMINI_MODELS) {
            console.log(`Sending image to Gemini model: ${model} (key index ${keyIndex - 1})...`);
            const startTime = Date.now();
            const r = await fetch(
                `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        contents: [{
                            parts: [
                                { text: prompt },
                                { inline_data: { mime_type: mimeType, data: imageBase64 } }
                            ]
                        }],
                        generationConfig: { 
                            temperature: 0.05, 
                            maxOutputTokens: 2048,
                            responseMimeType: "application/json"
                        }
                    }),
                    signal: AbortSignal.timeout(45000)
                }
            );
            console.log(`Gemini ${model} fetch completed in ${Date.now() - startTime}ms, status=${r.status}`);

            if (r.status === 503 || r.status === 429) {
                const errBody = await r.text();
                console.warn(`Gemini model ${model} unavailable (${r.status}): ${errBody.slice(0, 150)} — trying next...`);
                continue;
            }

            response = r;
            usedModel = model;
            break;
        }

        if (!response) {
            console.warn('All Gemini models returned 503/429 — skipping AI detection.');
            return null;
        }

        if (!response.ok) {
            const errBody = await response.text();
            console.warn(`Gemini AI detection returned ${response.status}: ${errBody.slice(0, 200)}`);
            return null;
        }

        const data = await response.json();
        let raw = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
        raw = raw.replace(/```json/g, '').replace(/```/g, '').trim();

        const parsed = JSON.parse(raw);
        console.log(`Gemini AI Detection [${usedModel}]: ${parsed.ai_probability}% AI probability — ${parsed.verdict} [${parsed.image_category}] confidence=${parsed.confidence}`);
        return {
            aiProbability: parsed.ai_probability,
            isAI: parsed.ai_probability >= 45,
            verdict: parsed.verdict,
            confidence: parsed.confidence,
            imageCategory: parsed.image_category || 'Other',
            artifacts: parsed.artifacts_detected || [],
            reasoning: parsed.reasoning || ''
        };
    } catch (err) {
        console.warn('Gemini AI detection failed:', err.message);
        return null;
    }
}

/**
 * Given a detection result, compute credibility score + verdict override.
 * Returns null if no override should be applied.
 * Returns { credibility_score, verdict, softWarning, ... } — softWarning=true means
 * we inject red flags but don't fully override the score.
 */
export function computeAiOverride(detection) {
    if (!detection) return null;
    const { aiProbability, confidence } = detection;

    // ── TIER 1: High/Medium confidence hard overrides ──
    if (confidence !== 'Low') {
        // Definite AI
        if (aiProbability >= 80) {
            return {
                credibility_score: Math.max(2, Math.round(100 - aiProbability)),
                verdict: `AI-Generated / Fake (${aiProbability}% confidence)`,
                ai_probability: aiProbability,
                ai_artifacts: detection.artifacts,
                ai_reasoning: detection.reasoning,
                softWarning: false,
            };
        }
        // Likely AI
        if (aiProbability >= 45) {
            return {
                credibility_score: Math.round((100 - aiProbability) * 0.5),
                verdict: `Likely AI-Generated / Edited (${aiProbability}% confidence)`,
                ai_probability: aiProbability,
                ai_artifacts: detection.artifacts,
                ai_reasoning: detection.reasoning,
                softWarning: false,
            };
        }
        // Suspicious — soft warning (don't override score, but inject red flags)
        if (aiProbability >= 25) {
            return {
                credibility_score: null, // don't override
                verdict: null,           // don't override
                ai_probability: aiProbability,
                ai_artifacts: detection.artifacts,
                ai_reasoning: detection.reasoning,
                softWarning: true,
            };
        }
    }

    // ── TIER 2: Low confidence overrides ──
    if (confidence === 'Low') {
        if (aiProbability >= 80) {
            return {
                credibility_score: Math.max(10, Math.round((100 - aiProbability) * 1.5)),
                verdict: `Possibly AI-Generated (${aiProbability}% AI probability, low confidence)`,
                ai_probability: aiProbability,
                ai_artifacts: detection.artifacts,
                ai_reasoning: detection.reasoning,
                softWarning: false,
            };
        }
        if (aiProbability >= 50) {
            return {
                credibility_score: null,
                verdict: null,
                ai_probability: aiProbability,
                ai_artifacts: detection.artifacts,
                ai_reasoning: detection.reasoning,
                softWarning: true,
            };
        }
    }

    return null;
}
