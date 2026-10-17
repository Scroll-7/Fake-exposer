# Honest App Rating (Post-Fix)

## Overall: 8.5/10 — much better, production-hardened

All requested fixes are implemented (auth on /api with same-origin guard + optional API_AUTH_TOKEN, SSRF hardened with connect-time DNS validation, full IPv4/IPv6 range coverage, TOCTOU/size caps).
Tests: 215/0. Eval harness: 33/0. Coverage ~85.8% lines / 79.8% branches.

## ML — 6/10
Text (TF-IDF+LogReg+heuristics) reasonable, not overfit. Image: realistic 512px ROC-AUC ~0.86–0.89 (macro-F1~0.80), CIFAKE 32px ROC-AUC ~0.63 (domain shift). Dedup check: eval stems overlap with train pool = 0%. v0 remains default (more general) vs v1 probe (better same-domain, loses cross-domain).

## Bugs/correctness — 9/10
215 passing, added auth tests + extended scraper/server SSRF tests. Weak spots: server.js routes (69.6%), groq.js tail.

## Security — 8.5/10
Auth middleware (services/auth.js, 100%): timing-safe token check (Bearer/X-API-Key), same-origin enforced, backwards compatible (no token = local mode with warning). SSRF (services/scraper.js): full IPv4/IPv6 coverage, custom safeLookup (connect-time validation), manual redirects (5 hops), 2MB cap, up-front guard before Jina reader.

## Design — 8/10
Paper theme, token UI, auth headers from frontend, .glass-panel→.panel cleanup.

## Architecture/docs — 9/10
New auth module, hardened scraper, .env.example updated, comprehensive tests.

## Scorecard
| Category | Before | After |
|---|---|---|
| ML/generalization | 5.5 | 6.0 |
| Bugs/correctness | 8.5 | 9.0 |
| Testing | 8.5 | 9.0 |
| Security | 4.0 | 8.5 |
| Design | 8.0 | 8.0 |
| Architecture/docs | 8.5 | 9.0 |
| **Overall** | **7.0** | **8.5** |

