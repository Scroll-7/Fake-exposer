/**
 * A failover-only pool of API keys.
 *
 * Unlike a round-robin ring, a KeyRing keeps using the *current* key until it
 * fails. When a key fails, call `markBad(key)` and then `rotate()` to fail over
 * to the next usable key. A marked key is skipped until its cooldown elapses,
 * after which it returns to the pool automatically (useful for rate limits that
 * reset). Pass a long cooldown for keys that are outright denied.
 *
 * @param {string} rawValue — Comma-separated keys, e.g. process.env.SOME_KEYS
 * @param {Object} [options]
 * @param {number} [options.defaultCooldownMs=60000] — Cooldown used by markBad
 * @returns {{
 *   readonly size: number,
 *   current: () => (string|null),
 *   markBad: (key: string, cooldownMs?: number) => void,
 *   rotate: () => (string|null),
 *   reset: () => void
 * }}
 */
export function createKeyRing(rawValue, { defaultCooldownMs = 60_000 } = {}) {
    const keys = String(rawValue || '')
        .split(',')
        .map(k => k.trim())
        .filter(Boolean);
    const cooldowns = new Map();
    let cursor = 0;

    function usable(key) {
        const until = cooldowns.get(key);
        return until === undefined || Date.now() >= until;
    }

    function current() {
        if (keys.length === 0) return null;
        for (let i = 0; i < keys.length; i++) {
            const idx = (cursor + i) % keys.length;
            if (usable(keys[idx])) {
                cursor = idx;
                return keys[idx];
            }
        }
        return null;
    }

    function markBad(key, cooldownMs = defaultCooldownMs) {
        if (key) cooldowns.set(key, Date.now() + cooldownMs);
    }

    function rotate() {
        if (keys.length === 0) return null;
        cursor = (cursor + 1) % keys.length;
        return current();
    }

    function reset() {
        cooldowns.clear();
        cursor = 0;
    }

    return {
        get size() { return keys.length; },
        current,
        markBad,
        rotate,
        reset
    };
}
