export function detectExtremePhysiqueCasualSetting(description) {
    if (!description) return null;
    const d = description.toLowerCase();

    const EXTREME_PHYSIQUE_WORDS = [
        'shredded', 'ripped', 'jacked', 'vascular', 'vasc',
        'six.?pack', 'six pack',
        'bodybuilder', 'bodybuilding',
        'competition physique', 'stage physique', 'contest physique',
        'competition-ready', 'stage-ready',
        'hypertrophy', 'jacked', 'swole',
        'massive muscles', 'huge muscles', 'enormous muscles',
        'deeply defined', 'ultra-defined', 'extreme definition',
    ];

    const CASUAL_SETTING_WORDS = [
        'mirror', 'selfie', 'bathroom', 'bedroom', 'living room',
        'home', 'house', 'apartment', 'hallway', 'hotel room',
    ];

    const hasExtremePhysique = EXTREME_PHYSIQUE_WORDS.some(w => new RegExp(w).test(d));
    const hasCasualSetting = CASUAL_SETTING_WORDS.some(w => new RegExp(w).test(d));

    if (hasExtremePhysique && hasCasualSetting) {
        return 'Image description indicates an EXTREME/COMPETITION-LEVEL physique (e.g. shredded, vascular, six-pack) in a CASUAL everyday setting (mirror, bathroom, bedroom). This specific combination is statistically very unlikely to be a natural unedited photo. AI muscle enhancement apps commonly produce exactly this result.';
    }

    return null;
}

export function detectAiPortrait(description) {
    if (!description) return null;
    const d = description.toLowerCase();

    const BG_COLOR_WORDS = ['gray', 'grey', 'white', 'beige', 'neutral', 'plain', 'solid', 'uniform', 'seamless', 'simple', 'clean', 'blank', 'muted'];
    const BG_CONTEXT_WORDS = ['background', 'backdrop', 'wall', 'surface'];
    const hasBgColor = BG_COLOR_WORDS.some(w => d.includes(w));
    const hasBgContext = BG_CONTEXT_WORDS.some(w => d.includes(w));
    const hasUniformBg = hasBgColor && hasBgContext;

    const PORTRAIT_SIGNALS = [
        'portrait', 'headshot', 'head shot', 'close-up', 'close up',
        'facing', 'looking at', 'looking directly', 'direct gaze',
        'front-facing', 'face forward', 'centered', 'centered in',
        'passport', 'id photo', 'profile photo', 'professional photo',
        'studio photo', 'head and shoulders',
    ];
    const hasPortraitFraming = PORTRAIT_SIGNALS.some(w => d.includes(w));

    const PERFECT_SIGNALS = [
        'smooth skin', 'clear skin', 'perfect skin', 'flawless', 'blemish',
        'studio lighting', 'professional lighting', 'even lighting', 'well-lit', 'well lit',
        'sharp', 'high quality', 'photorealistic', 'realistic', 'detailed',
        'no visible imperfection', 'no obvious',
    ];
    const hasPerfectQuality = PERFECT_SIGNALS.some(w => d.includes(w));

    const REAL_CONTEXT = [
        'mirror', 'selfie', 'phone', 'holding', 'friends', 'family',
        'outdoors', 'outside', 'street', 'crowd', 'event', 'concert',
        'sports', 'jersey', 'gym', 'office', 'classroom', 'restaurant',
        'car', 'park', 'beach', 'nature',
    ];
    const hasRealContext = REAL_CONTEXT.some(w => d.includes(w));

    const FACE_SIGNALS = ['man', 'woman', 'person', 'individual', 'face', 'young', 'male', 'female', 'subject'];
    const hasFace = FACE_SIGNALS.some(w => d.includes(w));

    if (hasUniformBg && hasPortraitFraming && hasFace && !hasRealContext) {
        return 'Image matches the classic AI-generated portrait signature: uniform/neutral background, centered forward-facing portrait, no real-world environmental context. AI image generators (Gemini, DALL-E, Midjourney) consistently produce this exact pattern.';
    }

    if (hasUniformBg && hasFace && hasPerfectQuality && !hasRealContext) {
        return 'Image has a uniform neutral background with studio-quality rendering, no real-world context — a common signature of AI portrait generators.';
    }

    return null;
}

export function detectAiFilename(filename) {
    if (!filename) return null;
    const f = filename.toLowerCase();

    const AI_KEYWORDS = [
        'midjourney', 'dall-e', 'dalle', 'stable-diffusion', 'stablediffusion',
        'flux', 'comfyui', 'chatgpt', 'ai-generated', 'ai_generated',
        'generated_by', 'deepfake', 'faceapp', 'bing image creator', 'nightcafe'
    ];

    for (const keyword of AI_KEYWORDS) {
        if (f.includes(keyword)) {
            return `The original filename ("${filename}") explicitly contains the name of an AI image generator or AI tool ("${keyword}"). This is extremely strong evidence that the image is AI-generated.`;
        }
    }

    return null;
}
