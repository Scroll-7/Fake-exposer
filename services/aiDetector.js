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
        const prompt = `You are a specialized AI image forensics detector. Your ONLY job is to determine if this image was AI-generated or AI-edited (e.g. by Midjourney, DALL-E, Stable Diffusion, ChatGPT image edit, Photoshop AI, etc).

Examine the image CAREFULLY for these telltale signs of AI generation or AI editing:
1. SKIN TEXTURE: Is it too smooth, waxy, plastic-looking, or overly "perfect"? Real photos have pores, slight unevenness, natural variation.
2. MUSCLE DEFINITION: Are muscles unnaturally defined, shredded, or volumetrically impossible for the person's apparent age/frame? AI often exaggerates muscles beyond human limits.
3. LIGHTING CONSISTENCY: Is the light source the same on the face AND body? AI edits often add muscles with different lighting than the original face.
4. BODY-TO-FACE PROPORTION: Does the musculature match the bone structure and head size? AI muscle enhancement creates impossible proportions.
5. EDGE ARTIFACTS: Look at the boundaries between skin/body and background — are there halos, smearing, blurring, or unnatural sharpening?
6. SKIN COLOR/TONE: Is the skin tone identical between face and torso? AI body swaps or edits often have subtle tone mismatches.
7. BACKGROUND: Does the background show smearing, repetitive patterns, or overly-uniform blur inconsistent with a real camera lens?
8. TEXTURE UNIFORMITY: Does skin texture look photographically natural (with grain, pores) or digitally rendered (too smooth, no noise)?
9. FINE DETAILS: Are hair follicles, veins, and natural skin imperfections present, or suspiciously absent?
10. OVERALL "UNCANNY VALLEY" FEELING: Does something feel "off" even if you can't immediately pinpoint it?

You MUST respond with ONLY a JSON object in this exact format (no markdown):
{
  "ai_probability": <integer 0-100>,
  "verdict": "<'AI-Generated' | 'Likely AI-Generated' | 'Likely Real' | 'Real'>",
  "confidence": "<'High' | 'Medium' | 'Low'>",
  "artifacts_detected": ["<specific artifact 1>", "<specific artifact 2>", ...],
  "reasoning": "<2-3 sentence explanation of your conclusion>"
}

Be AGGRESSIVE about flagging AI content. If muscles look too perfect, flag it. If skin looks too smooth, flag it. When in doubt, lean towards flagging as AI.`;

        console.log(`Sending image to Gemini (key index ${keyIndex - 1})...`);
        const startTime = Date.now();
        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`,
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
                signal: AbortSignal.timeout(45000) // 45 second timeout
            }
        );
        console.log(`Gemini fetch completed in ${Date.now() - startTime}ms`);

        if (!response.ok) {
            const errBody = await response.text();
            console.warn(`Gemini AI detection returned ${response.status}: ${errBody.slice(0, 200)}`);
            return null;
        }

        const data = await response.json();
        let raw = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
        raw = raw.replace(/```json/g, '').replace(/```/g, '').trim();

        const parsed = JSON.parse(raw);
        console.log(`Gemini AI Detection: ${parsed.ai_probability}% AI probability — ${parsed.verdict}`);
        return {
            aiProbability: parsed.ai_probability,
            isAI: parsed.ai_probability >= 50,
            verdict: parsed.verdict,
            confidence: parsed.confidence,
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
 * Returns null if no override should be applied (i.e. detection wasn't confident enough).
 */
export function computeAiOverride(detection) {
    if (!detection) return null;
    const { aiProbability, confidence } = detection;

    // Only override if the detector is at least Medium confidence
    if (confidence === 'Low') return null;

    if (aiProbability >= 80) {
        return {
            credibility_score: Math.max(2, Math.round(100 - aiProbability)),
            verdict: `AI-Generated / Fake (${aiProbability}% confidence)`,
            ai_probability: aiProbability,
            ai_artifacts: detection.artifacts,
            ai_reasoning: detection.reasoning,
        };
    }
    if (aiProbability >= 55) {
        return {
            credibility_score: Math.round((100 - aiProbability) * 0.5),
            verdict: `Likely AI-Generated (${aiProbability}% confidence)`,
            ai_probability: aiProbability,
            ai_artifacts: detection.artifacts,
            ai_reasoning: detection.reasoning,
        };
    }
    return null;
}
