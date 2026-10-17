import crypto from 'crypto';

// ── API ACCESS CONTROL ──
// The analysis endpoints call paid LLMs (Groq/Gemini), so an unauthenticated
// port can be drained by anyone who can reach it. We support an optional shared
// token (`API_AUTH_TOKEN`): when set, every /api request must present it as
// `Authorization: Bearer <token>` or `X-API-Key: <token>`.
//
// When the token is NOT set, auth is disabled so the app still works out of the
// box on localhost — but we always enforce a same-origin check, which stops a
// random web page from making a visitor's browser hammer the API (CSRF).
export function timingSafeEqual(a, b) {
    const bufA = Buffer.from(String(a));
    const bufB = Buffer.from(String(b));
    if (bufA.length !== bufB.length) return false;
    return crypto.timingSafeEqual(bufA, bufB);
}

export function extractToken(req) {
    const auth = req.headers?.authorization || '';
    const match = /^Bearer\s+(.+)$/i.exec(auth);
    if (match) return match[1].trim();
    const apiKey = req.headers?.['x-api-key'];
    if (typeof apiKey === 'string' && apiKey.trim()) return apiKey.trim();
    return null;
}

// Normalise an origin string to its `scheme://host` form; return it unchanged
// if it does not parse (so a typo matches nothing instead of matching all).
function normalizeOrigin(value) {
    try {
        return new URL(value).origin;
    } catch {
        return String(value);
    }
}

// Normalise a comma-separated CORS allowlist into a list of `scheme://host`
// origins.
export function parseOrigins(raw) {
    return String(raw || '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .map(normalizeOrigin);
}

// Allow requests with no Origin header (non-browser clients, curl, same-origin
// navigations) and requests whose Origin matches the request host. An explicit
// cross-origin request is allowed only when its origin is on the allowlist.
export function isSameOrigin(req, allowedOrigins = []) {
    const origin = req.headers?.origin;
    if (!origin || origin === 'null') return true;
    try {
        const parsed = new URL(origin);
        if (parsed.host === req.headers?.host) return true;
        return allowedOrigins.some((o) => normalizeOrigin(o) === parsed.origin);
    } catch {
        return false;
    }
}

export function createApiAuth(
    getToken = () => process.env.API_AUTH_TOKEN,
    getAllowedOrigins = () => parseOrigins(process.env.CORS_ORIGINS),
) {
    return function requireApiAuth(req, res, next) {
        if (!isSameOrigin(req, getAllowedOrigins())) {
            return res.status(403).json({ error: 'Cross-origin requests are not allowed.' });
        }
        const expected = getToken();
        if (!expected) return next();
        const provided = extractToken(req);
        if (provided && timingSafeEqual(provided, expected)) return next();
        res.set('WWW-Authenticate', 'Bearer');
        return res.status(401).json({ error: 'Unauthorized. Provide a valid API token.' });
    };
}

// Fail-fast guard for production: an unauthenticated public API can be drained
// by anyone who can reach it. Returns an error message, or null when the
// configuration is safe.
export function productionAuthError(env = process.env) {
    if (env.NODE_ENV === 'production' && !env.API_AUTH_TOKEN) {
        return 'API_AUTH_TOKEN must be set when NODE_ENV=production — refusing to run an unauthenticated API.';
    }
    return null;
}
