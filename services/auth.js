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

// Allow requests with no Origin header (non-browser clients, curl, same-origin
// navigations). Reject any explicit Origin that does not match the request host.
export function isSameOrigin(req) {
    const origin = req.headers?.origin;
    if (!origin || origin === 'null') return true;
    try {
        return new URL(origin).host === req.headers.host;
    } catch {
        return false;
    }
}

export function createApiAuth(getToken = () => process.env.API_AUTH_TOKEN) {
    return function requireApiAuth(req, res, next) {
        if (!isSameOrigin(req)) {
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
