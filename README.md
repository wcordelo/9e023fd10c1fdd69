# NEON Co-Pilot (Python)

AI co-pilot that connects to NEON, reconstructs fragmented challenges, routes each checkpoint, and responds using the required JSON protocol.

## What This Solves

- Connects to `wss://neonhealth.software/agent-puzzle/challenge`
- Reconstructs timestamped signal fragments into prompts
- Routes prompts to the correct handler:
  - Handshake / vessel authorization
  - Computation (`Math.floor`, arithmetic, `%`, `#` suffix handling)
  - Knowledge archive (Wikipedia summary + Nth word extraction)
  - Crew manifest (Gemini-based answer with strict char constraints)
  - Transmission verification (session-memory recall + fuzzy fallback)
- Applies response validation before sending

## Project Files

- `copilot.py` - main runtime + handlers (read the **module docstring at the top** for end-to-end data flow)
- `tests/test_copilot.py` - regression suite
- `tests/fixtures.py` - captured real challenge fixtures
- `resume.txt` - crew manifest source
- `requirements.txt` - dependencies
- `.env` - local secrets/config (not for commit)

## Prerequisites

- Python 3.11+ (3.12 recommended)
- A valid Gemini API key in `GOOGLE_API_KEY`
- Your NEON code in `NEON_CODE`

## Setup

```bash
cd neon-health
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example` for keys):

```dotenv
GOOGLE_API_KEY=your_key_here
NEON_CODE=your_neon_code_here
```

## Run

```bash
source venv/bin/activate
python copilot.py
```

## Optional Debug Logging

Verbose trace logs are off by default.

```bash
NEON_DEBUG=1 python copilot.py
```

## Tests

Run full regression suite:

```bash
source venv/bin/activate
python -m pytest tests/test_copilot.py -v
```

Quick run:

```bash
python -m pytest tests/test_copilot.py -q
```

## Live-Run Safety Workflow (Important)

Use this process before each new NEON submission:

1. Run tests locally and ensure all pass.
2. Make one live submission.
3. If any checkpoint fails:
   - Capture the exact prompt and expected behavior.
   - Add a new fixture in `tests/fixtures.py`.
   - Add/update a test in `tests/test_copilot.py`.
   - Fix code until the new test and full suite pass.
4. Only then submit again.

This keeps attempts low and converts every failure into permanent regression coverage.

## Notes on External Calls in Tests

- Knowledge tests hit live Wikipedia endpoints.
- Manifest tests hit Gemini (requires `GOOGLE_API_KEY`).
- Computation expected values are validated against JavaScript behavior and stored in fixtures.

## Cache Cleanup (Optional)

If you want a clean tree before pushing:

```bash
rm -rf .pytest_cache
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
```

## GitHub Push Checklist

- Ensure `.env` is excluded by `.gitignore`
- Run tests: `python -m pytest tests/test_copilot.py -q`
- Confirm no local secrets are tracked
- Commit source + tests + docs
