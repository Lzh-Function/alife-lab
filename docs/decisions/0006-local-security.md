# ADR 0006 — Local-only access policy

- Status: accepted (Phase 1)
- Date: 2026-11-18

- `alife serve` refuses non-loopback bind addresses until authentication exists.
- Host header allow-list (`127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>`)
  blocks DNS rebinding (HTTP 421).
- Non-GET requests require `X-ALife-Token`; the random token is generated per server
  start, stored in `.alife/session_token` (mode 0600, git-ignored) and injected into
  the served `index.html`. Cross-origin pages can neither read it nor send custom
  headers without a CORS preflight, which is never granted. The GUI re-reads the
  token from `/` after a 403, so open tabs survive a server restart.
- If `Origin` is present it must be one of the server's own origins. WebSocket
  upgrades require an allowed Origin and the token.
- Bodies are limited to 2 MiB; brush payloads to 65 536 cells; artifact paths are
  derived from validated IDs only (no user-supplied paths are opened).
- No `eval`, no shell construction from user input; `nvidia-smi`/`git` are invoked
  with fixed argv.
