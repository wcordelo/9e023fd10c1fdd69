"""
NEON co-pilot: WebSocket client + checkpoint handlers for the agent puzzle.

Data flow (single session):
  1) Receive JSON from NEON; `message` may be fragmented `{word, timestamp}` bursts.
  2) `reconstruct_message` sorts by timestamp and joins into one prompt string.
  3) `route_and_respond` classifies the prompt, dispatches to a handler, validates
     the JSON response shape, then appends to `session_memory` for verification.
  4) Send exactly one JSON object per challenge (`enter_digits` or `speak_text`).

Handlers: handshake (keypad), computation (JS-style math), knowledge (Wikipedia
word N), manifest (Gemini + resume + length rules), verification (recall from
session memory). See README.md for run/test notes.
"""

import asyncio
import json
import math
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import websockets
from dotenv import load_dotenv
from google import genai

WS_URL = "wss://neonhealth.software/agent-puzzle/challenge"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
# Protocol caps `speak_text.text` at 256 chars; used as default max for open-ended constraints.
MAX_SPEAK_TEXT = 256
WIKI_HEADERS = {"User-Agent": "NEON-CoPilot/1.0 (https://neonhealth.software)"}
VALID_CHECKPOINTS = {"handshake", "computation", "knowledge", "manifest", "verification"}
# After sanitization, only these characters may reach eval(); blocks code injection.
SAFE_EXPR_PATTERN = r"^[0-9A-Za-z+\-*/%().,_\s]+$"

# Keyword hints for fast routing (see classify_checkpoint order).
COMPUTATION_HINTS = {
    "calculate", "compute", "evaluate",
    "math.floor", "math.ceil", "math.round", "math.abs", "math.pow",
}
COMPUTATION_OPS = {"+", "-", "*", "/", "%", "math."}
HANDSHAKE_HINTS = {"authorization code", "neon code", "comm panel keypad", "respond on frequency"}
KNOWLEDGE_HINTS = {"knowledge archive", "entry summary for"}
VERIFICATION_HINTS = {
    "transmission verification",
    "earlier you transmitted",
    "earlier, you transmitted",
    "earlier you spoke",
    "recall a specific word",
    "recall the",
}
MANIFEST_HINTS = {
    "crew member", "crew manifest", "resume",
    "background", "education", "work experience",
    "skills", "best project", "granted access",
}


ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}


@dataclass
class LengthConstraint:
    min_chars: int
    max_chars: int


class NeonCopilot:
    """Orchestrates classification, external APIs (Wikipedia, Gemini), and session memory.

    `session_memory` entries: {checkpoint, prompt, response} where `response` is the
    validated text/digits we actually sent (used for verification word recall).
    """

    def __init__(self, neon_code: str, resume_text: str, gemini_api_key: str):
        self.neon_code = neon_code.strip()
        self.resume_text = resume_text.strip()
        self.gemini = genai.Client(api_key=gemini_api_key)
        self.session_memory: List[Dict[str, str]] = []
        # Enable verbose runtime logs only when explicitly requested.
        self.debug = os.getenv("NEON_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    def _log(self, message: str) -> None:
        if self.debug:
            print(message)

    def _sanitize_expression(self, expr: str) -> str:
        # Strip narrative/punctuation so eval() only sees JS-like math tokens.
        expr = expr.replace("`", "").strip()
        expr = re.sub(r"[^0-9A-Za-z+\-*/%().,_\s]", " ", expr)
        return re.sub(r"\s+", " ", expr).strip()

    def _with_optional_pound(self, digits: str, prompt: str) -> str:
        # Comm panel expects a trailing # when the prompt says "pound key" / "followed by #".
        return f"{digits}#" if self.needs_pound_key(prompt) else digits

    def _wiki_extract_for_title(self, title: str) -> Optional[str]:
        # REST summary; WIKI_HEADERS required (Wikipedia blocks anonymous clients without User-Agent).
        encoded = urllib.parse.quote(title, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        r = requests.get(url, timeout=8, headers=WIKI_HEADERS)
        if r.status_code != 200:
            return None
        return r.json().get("extract")

    # ── Message reconstruction ────────────────────────────────────────

    def reconstruct_message(self, fragments: Any) -> str:
        # Server may send a plain string in edge cases; otherwise sort bursts by time.
        if not isinstance(fragments, list):
            return str(fragments)
        clean = [f for f in fragments if isinstance(f, dict) and "word" in f and "timestamp" in f]
        ordered = sorted(clean, key=lambda x: x["timestamp"])
        return " ".join(str(item["word"]) for item in ordered).strip()

    # ── Checkpoint classification ─────────────────────────────────────

    def classify_checkpoint(self, prompt: str) -> str:
        # Order matters: computation before handshake so narratives that say "frequency"
        # but embed Math.* still route to math; verification before manifest so "skills"
        # in a recall prompt does not open a new manifest answer.
        p = prompt.lower()

        has_math_expr = any(k in p for k in COMPUTATION_HINTS) and any(op in p for op in COMPUTATION_OPS)
        if has_math_expr:
            return "computation"

        if any(k in p for k in HANDSHAKE_HINTS):
            return "handshake"

        if any(k in p for k in KNOWLEDGE_HINTS):
            return "knowledge"

        if any(k in p for k in VERIFICATION_HINTS):
            return "verification"

        if any(k in p for k in MANIFEST_HINTS):
            return "manifest"

        return self.classify_with_gemini(prompt)

    def classify_with_gemini(self, prompt: str) -> str:
        system_instruction = (
            "Classify the prompt into exactly one category: handshake, computation, knowledge, "
            "manifest, verification. Return ONLY the category name, nothing else."
        )
        text = self.gemini_text(system_instruction=system_instruction, user_prompt=prompt, max_tokens=16)
        label = (text or "").strip().lower()
        if label in VALID_CHECKPOINTS:
            return label
        # Conservative default: manifest is the safest bucket when the model output is unusable.
        return "manifest"

    # ── Constraint / index parsing ────────────────────────────────────

    def parse_length_constraints(self, prompt: str) -> LengthConstraint:
        # "less than N" / "under N" => at most N-1 characters (strict inequality).
        p = prompt.lower()

        between = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s+(?:total\s+)?characters", p)
        if between:
            a, b = int(between.group(1)), int(between.group(2))
            return LengthConstraint(min(a, b), max(a, b))

        exact = re.search(r"exactly\s+(\d+)\s+(?:total\s+)?characters", p)
        if exact:
            n = int(exact.group(1))
            return LengthConstraint(n, n)

        at_most = re.search(r"(?:at most|no more than)\s+(\d+)\s+(?:total\s+)?characters", p)
        if at_most:
            return LengthConstraint(1, int(at_most.group(1)))

        less_than = re.search(r"(?:less than|under|fewer than|in less than)\s+(\d+)\s+(?:total\s+)?characters", p)
        if less_than:
            return LengthConstraint(1, int(less_than.group(1)) - 1)

        at_least = re.search(r"at least\s+(\d+)\s+(?:total\s+)?characters", p)
        if at_least:
            # Upper bound still capped by protocol max for speak_text.
            return LengthConstraint(int(at_least.group(1)), MAX_SPEAK_TEXT)

        return LengthConstraint(1, MAX_SPEAK_TEXT)

    def parse_ordinal_index(self, prompt: str) -> Optional[int]:
        # "Nth word" in the prompt is 1-based English; return value is 0-based for list indexing.
        p = prompt.lower()
        m = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", p)
        if m:
            return int(m.group(1)) - 1
        m2 = re.search(r"\b(\d+)\s+word\b", p)
        if m2:
            return int(m2.group(1)) - 1
        for word, num in ORDINAL_WORDS.items():
            if re.search(rf"\b{word}\s+word\b", p):
                return num - 1
        # Loose fallback: "first" ... without "word" (prefer tighter matches above).
        for word, num in ORDINAL_WORDS.items():
            if re.search(rf"\b{word}\b", p):
                return num - 1
        return None

    # ── Knowledge archive helpers ─────────────────────────────────────

    def parse_title(self, prompt: str) -> Optional[str]:
        # Wikipedia titles appear as 'Title', "Title", or curly quotes; last resort: unquoted after "for".
        quote = re.search(r"for\s+['\u2018\u2019\u201c\u201d\"]+([^'\u2018\u2019\u201c\u201d\"]+)['\u2018\u2019\u201c\u201d\"]+", prompt)
        if quote:
            return quote.group(1).strip()
        quote2 = re.search(r"for\s+'([^']+)'", prompt, re.IGNORECASE)
        if quote2:
            return quote2.group(1).strip()
        quote3 = re.search(r'for\s+"([^"]+)"', prompt, re.IGNORECASE)
        if quote3:
            return quote3.group(1).strip()
        plain = re.search(r"(?:entry|summary)\s+for\s+([A-Za-z0-9 _\-/\(\),.]+?)(?:,\s*which|\.\s|$)", prompt, re.IGNORECASE)
        if plain:
            return plain.group(1).strip(" .?\"',")
        return None

    def wiki_summary(self, title: str) -> Optional[str]:
        t = title.strip()
        # Try common title shapes (underscores vs spaces, capitalization) before search API.
        variants = list(dict.fromkeys([
            t, t.replace(" ", "_"), t.replace("_", " "),
            t.title(), t.title().replace(" ", "_"),
            t.capitalize(),
        ]))
        for v in variants:
            if not v:
                continue
            try:
                extract = self._wiki_extract_for_title(v)
                if extract:
                    return extract
            except requests.RequestException:
                continue

        try:
            # If title variants miss (underscores, redirects), resolve via search API then re-fetch summary.
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "query", "list": "search", "srsearch": title, "format": "json", "srlimit": 1}
            sr = requests.get(search_url, params=params, timeout=8, headers=WIKI_HEADERS)
            if sr.status_code == 200:
                results = sr.json().get("query", {}).get("search", [])
                if results:
                    best_title = results[0].get("title")
                    if best_title:
                        extract = self._wiki_extract_for_title(best_title)
                        if extract:
                            return extract
        except requests.RequestException:
            pass
        return None

    def safe_words(self, text: str) -> List[str]:
        # Whitespace split matches NEON-style tokenization; strip punctuation on pick, not here.
        return text.split()

    def strip_word_punctuation(self, word: str) -> str:
        return word.strip(".,;:!?\"'`()[]{}*")

    # ── Computation helpers ───────────────────────────────────────────

    def extract_expression(self, prompt: str) -> str:
        # Prompts often have narrative before "Evaluate ... : <expr>"; use the last ":" so
        # earlier colons in the story do not truncate the expression.
        p = prompt.strip()
        if ":" in p:
            rhs = p.rsplit(":", 1)[1].strip()
            if any(op in rhs for op in ["Math.", "+", "*", "/", "%", "(", ")"]):
                return self._sanitize_expression(rhs)

        math_expr = re.search(r"(Math\.[A-Za-z]+\s*\(.*\)(?:\s*[%/*+\-]\s*[\d(][\d\s%/*+\-()]*)*)", p, re.IGNORECASE)
        if math_expr:
            expr = math_expr.group(1)
        else:
            paren_expr = re.search(r"(\([^:]*\)(?:\s*[%/*+\-]\s*[\d(][\d\s%/*+\-()]*)*)", p)
            if paren_expr:
                expr = paren_expr.group(1)
            else:
                calc2 = re.search(r"(?:calculate|compute|determine|evaluate)\s+(.+?)(?:\s+and\s+transmit|\s+and\s+enter|\s+then|\s+followed by|$)", p, re.IGNORECASE)
                expr = calc2.group(1) if calc2 else p
        return self._sanitize_expression(expr)

    def eval_expression_python(self, expr: str) -> int:
        converted = expr
        converted = converted.replace("Math.floor", "math.floor")
        converted = converted.replace("Math.ceil", "math.ceil")
        converted = converted.replace("Math.round", "round")
        converted = converted.replace("Math.abs", "abs")
        converted = re.sub(r"Math\.pow\s*\(", "pow(", converted)
        if not re.match(SAFE_EXPR_PATTERN, converted):
            raise ValueError(f"Unsafe expression: {converted}")
        # Restricted env: approximate JavaScript Math.* + arithmetic (see tests for JS parity).
        allowed = {"math": math, "round": round, "abs": abs, "pow": pow}
        value = eval(converted, {"__builtins__": {}}, allowed)  # noqa: S307
        # Checkpoint expects integer digits; int() truncates toward zero (matches non-negative JS results here).
        return int(value)

    def eval_expression_with_gemini(self, expr: str) -> int:
        system_instruction = (
            "Evaluate the following JavaScript math expression. "
            "Return ONLY the final integer result. No words, no explanation, just the number."
        )
        text = self.gemini_text(system_instruction=system_instruction, user_prompt=expr, max_tokens=32)
        m = re.search(r"-?\d+", text or "")
        if not m:
            raise ValueError(f"No integer found in Gemini output: {text}")
        return int(m.group(0))

    def needs_pound_key(self, prompt: str) -> bool:
        p = prompt.lower()
        return "pound key" in p or "followed by #" in p or "followed by the #" in p

    # ── Gemini wrapper ────────────────────────────────────────────────

    def gemini_text(self, system_instruction: str, user_prompt: str, max_tokens: int = 256) -> str:
        # Low temperature for deterministic routing, manifest answers, and math fallback.
        response = self.gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "max_output_tokens": max_tokens,
                "temperature": 0.2,
            },
        )
        return (response.text or "").strip()

    # ── Session memory ────────────────────────────────────────────────

    def fuzzy_best_memory(self, query: str) -> Optional[Dict[str, str]]:
        # Fallback when topic extraction does not match: token overlap over full history.
        if not self.session_memory:
            return None
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        best_item = None
        best_score = -1
        for item in self.session_memory:
            blob = f"{item.get('prompt', '')} {item.get('response', '')}".lower()
            item_tokens = set(re.findall(r"[a-z0-9]+", blob))
            overlap = len(query_tokens.intersection(item_tokens))
            if item.get("checkpoint") == "manifest":
                overlap += 2  # Prefer manifest rows when verification asks about prior speech.
            if overlap > best_score:
                best_score = overlap
                best_item = item
        return best_item

    # ── Text clamping ─────────────────────────────────────────────────

    def clamp_text(self, text: str, min_chars: int, max_chars: int) -> str:
        # If still short after Gemini retries, pad from resume (already validated content).
        out = text.strip()
        if len(out) > max_chars:
            cut = out[:max_chars]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            out = cut.strip() or out[:max_chars]
        if len(out) < min_chars:
            filler = " " + self.resume_text.replace("\n", " ")
            out = (out + filler[: min_chars - len(out) + 10]).strip()
        return out[:max_chars]

    # ── Response format guard ─────────────────────────────────────────

    def validate_response(self, obj: Dict[str, Any]) -> Dict[str, str]:
        t = obj.get("type")
        if t not in {"enter_digits", "speak_text"}:
            raise ValueError(f"Invalid response type: {t}")
        if t == "enter_digits":
            digits = str(obj.get("digits", "")).strip()
            # Allow a-f so NEON_CODE hex survives sanitization; keep optional trailing #.
            digits = re.sub(r"[^0-9a-fA-F#]", "", digits)
            if not digits:
                raise ValueError("enter_digits requires non-empty digits")
            return {"type": "enter_digits", "digits": digits}
        text = str(obj.get("text", "")).strip()
        if not text:
            raise ValueError("speak_text requires non-empty text")
        if len(text) > MAX_SPEAK_TEXT:
            # Prefer a word boundary so we do not send a chopped token.
            if " " in text[:MAX_SPEAK_TEXT]:
                text = text[:MAX_SPEAK_TEXT].rsplit(" ", 1)[0]
            else:
                text = text[:MAX_SPEAK_TEXT]
        return {"type": "speak_text", "text": text}

    # ── Handler: Handshake ────────────────────────────────────────────

    def make_handshake_response(self, prompt: str) -> Dict[str, str]:
        # Priority: vessel auth code -> frequency next to "AI co-pilot" clause -> any frequency -> neon code.
        p = prompt.lower()

        if any(k in p for k in ["authorization code", "neon code", "vessel code"]):
            digits = self._with_optional_pound(self.neon_code, prompt)
            return {"type": "enter_digits", "digits": digits}

        ai_clause = re.search(
            r"(?:ai co-?pilot|excellent software engineer).*?frequency\s+(\d+)", p
        )
        if ai_clause:
            digits = self._with_optional_pound(ai_clause.group(1), prompt)
            return {"type": "enter_digits", "digits": digits}

        freq = re.search(r"frequency\s+(\d+)", p)
        if freq:
            digits = self._with_optional_pound(freq.group(1), prompt)
            return {"type": "enter_digits", "digits": digits}

        digits = self._with_optional_pound(self.neon_code, prompt)
        return {"type": "enter_digits", "digits": digits}

    # ── Handler: Computation ──────────────────────────────────────────

    def make_computation_response(self, prompt: str) -> Dict[str, str]:
        expr = self.extract_expression(prompt)
        self._log(f"  [computation] extracted expr: {expr}")
        try:
            result = self.eval_expression_python(expr)
            self._log(f"  [computation] python eval: {result}")
        except Exception as e:
            self._log(f"  [computation] python eval failed ({e}), falling back to Gemini")
            result = self.eval_expression_with_gemini(expr)
            self._log(f"  [computation] gemini eval: {result}")
        digits = self._with_optional_pound(str(result), prompt)
        return {"type": "enter_digits", "digits": digits}

    # ── Handler: Knowledge Archive ────────────────────────────────────

    def make_knowledge_response(self, prompt: str) -> Dict[str, str]:
        title = self.parse_title(prompt)
        idx = self.parse_ordinal_index(prompt)
        self._log(f"  [knowledge] title={title!r} word_index={idx}")
        if not title:
            raise ValueError("Could not parse knowledge archive title")
        if idx is None:
            raise ValueError("Could not parse word index")
        summary = self.wiki_summary(title)
        if not summary:
            raise ValueError(f"Failed to fetch summary for title: {title}")
        words = self.safe_words(summary)
        self._log(f"  [knowledge] summary has {len(words)} words, picking index {idx}")
        if not words:
            raise ValueError("Summary had no words")
        # If Wikipedia extract is shorter than N words, do not crash; clamp to last word.
        idx = max(0, min(idx, len(words) - 1))
        picked = self.strip_word_punctuation(words[idx])
        self._log(f"  [knowledge] picked word: {picked!r}")
        return {"type": "speak_text", "text": picked}

    # ── Handler: Manifest ─────────────────────────────────────────────

    def make_manifest_response(self, prompt: str) -> Dict[str, str]:
        constraints = self.parse_length_constraints(prompt)
        target = (constraints.min_chars + constraints.max_chars) // 2
        self._log(f"  [manifest] constraints: {constraints.min_chars}-{constraints.max_chars} chars, target={target}")

        system = (
            "You are answering a resume question for an authentication protocol. "
            "Use ONLY the resume content provided below. "
            f"Your answer MUST be between {constraints.min_chars} and {constraints.max_chars} characters long. "
            f"Aim for approximately {target} characters. "
            "Return ONLY the final answer text. No quotes, no prefixes, no explanation."
        )
        user = f"RESUME:\n{self.resume_text}\n\nQUESTION:\n{prompt}"

        answer = self.gemini_text(system_instruction=system, user_prompt=user, max_tokens=200)
        self._log(f"  [manifest] initial response ({len(answer)} chars): {answer[:80]}...")

        # Retry when Gemini misses length bounds; clamp_text is last resort.
        for retry in range(3):
            if constraints.min_chars <= len(answer) <= constraints.max_chars:
                break
            direction = "shorter" if len(answer) > constraints.max_chars else "longer with more detail"
            fix = (
                f"Your previous answer was {len(answer)} characters. "
                f"It must be between {constraints.min_chars} and {constraints.max_chars} characters. "
                f"Rewrite it {direction}. Aim for exactly {target} characters. "
                f"Return ONLY the answer text.\n\nPrevious answer:\n{answer}"
            )
            answer = self.gemini_text(system_instruction=system, user_prompt=fix, max_tokens=220)
            self._log(f"  [manifest] retry {retry+1} ({len(answer)} chars): {answer[:80]}...")

        final = self.clamp_text(answer, constraints.min_chars, constraints.max_chars)
        self._log(f"  [manifest] final ({len(final)} chars): {final[:80]}...")
        return {"type": "speak_text", "text": final}

    # ── Handler: Verification ─────────────────────────────────────────

    def make_verification_response(self, prompt: str) -> Dict[str, str]:
        p = prompt.lower()

        # Topic = what we spoke about (e.g. "best project", "skills"); used to pick the right manifest row.
        topic_patterns = [
            r"earlier you transmitted (?:your )?(?:crew member'?s? )?(.+?)(?:\.\s|\.$|$)",
            r"earlier,? you (?:transmitted|spoke about) (?:your )?(?:crew member'?s? )?(.+?)(?:\.\s|\.$|$)",
            r"you transmitted (?:your )?(?:crew member'?s? )?(.+?)(?:\.\s|\.$|$)",
        ]
        topic = None
        for pat in topic_patterns:
            m = re.search(pat, p)
            if m:
                topic = m.group(1).strip()
                break

        self._log(f"  [verification] extracted topic: {topic!r}")

        item = None
        if topic:
            topic_tokens = set(re.findall(r"[a-z0-9]+", topic))
            manifest_items = [m for m in self.session_memory if m.get("checkpoint") == "manifest"]
            scored: List[Tuple[int, Dict[str, str]]] = []
            for mem in manifest_items:
                prompt_lower = mem.get("prompt", "").lower()
                prompt_tokens = set(re.findall(r"[a-z0-9]+", prompt_lower))
                # Weighted matching helps map "best project"/"skills" prompts
                # to the exact manifest response previously sent.
                score = len(topic_tokens.intersection(prompt_tokens))
                for tok in topic_tokens:
                    if tok in prompt_lower:
                        score += 3
                scored.append((score, mem))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0:
                item = scored[0][1]
                self._log(f"  [verification] matched manifest prompt: {item['prompt'][:60]}...")

        if item is None:
            item = self.fuzzy_best_memory(prompt)
            if item:
                self._log(f"  [verification] fuzzy fallback matched: {item.get('prompt', '')[:60]}...")

        if not item:
            self._log("  [verification] no matching memory found!")
            return {"type": "speak_text", "text": "unknown"}

        idx = self.parse_ordinal_index(prompt)
        base = item.get("response", "").strip()
        self._log(f"  [verification] response to search: {base[:80]}...")
        self._log(f"  [verification] word index: {idx}")

        if idx is None:
            # Rare: could not parse ordinal; return whole prior reply (truncated) instead of failing closed.
            return {"type": "speak_text", "text": base[:MAX_SPEAK_TEXT]}

        words = self.safe_words(base)
        if not words:
            return {"type": "speak_text", "text": "unknown"}
        idx = max(0, min(idx, len(words) - 1))
        picked = self.strip_word_punctuation(words[idx])
        self._log(f"  [verification] words: {words[:10]}... picked [{idx}] = {picked!r}")
        return {"type": "speak_text", "text": picked}

    # ── Main routing ──────────────────────────────────────────────────

    def route_and_respond(self, prompt: str) -> Dict[str, str]:
        checkpoint = self.classify_checkpoint(prompt)
        self._log(f"  [router] classified as: {checkpoint}")

        if checkpoint == "handshake":
            response = self.make_handshake_response(prompt)
        elif checkpoint == "computation":
            response = self.make_computation_response(prompt)
        elif checkpoint == "knowledge":
            response = self.make_knowledge_response(prompt)
        elif checkpoint == "verification":
            response = self.make_verification_response(prompt)
        else:
            response = self.make_manifest_response(prompt)
            checkpoint = "manifest"

        safe = self.validate_response(response)
        # Store post-validation payload so verification recalls exactly what NEON received.
        self.session_memory.append({
            "checkpoint": checkpoint,
            "prompt": prompt,
            "response": safe.get("text", safe.get("digits", "")),
        })
        return safe

    # ── WebSocket main loop ───────────────────────────────────────────

    async def run(self) -> None:
        backoff_seconds = [1, 2, 4]
        for attempt in range(1, 4):
            try:
                # Fresh memory each connection attempt so verification cannot use stale prior runs.
                self.session_memory.clear()
                print(f"\n[info] connecting to NEON (attempt {attempt}/3)")
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    print("[info] connected\n")
                    while True:
                        raw = await ws.recv()
                        payload = json.loads(raw)
                        msg_type = payload.get("type")
                        if msg_type == "challenge":
                            reconstructed = self.reconstruct_message(payload.get("message", []))
                            self._log(f"[challenge] {reconstructed}")
                            response = self.route_and_respond(reconstructed)
                            self._log(f"[response] {json.dumps(response)}\n")
                            await ws.send(json.dumps(response))
                        elif msg_type == "success":
                            print("[SUCCESS] Authentication complete!")
                            return
                        elif msg_type == "error":
                            print(f"[ERROR] {payload.get('message', 'unknown error')}")
                            return
                        else:
                            # Helps diagnose protocol changes without failing the loop silently.
                            print(f"[debug] unhandled payload type={msg_type}: {payload}")
            except websockets.exceptions.ConnectionClosed as exc:
                print(f"[warn] connection closed: {exc}")
                if attempt == 3:
                    raise
                wait = backoff_seconds[attempt - 1]
                print(f"[info] reconnecting in {wait}s...")
                await asyncio.sleep(wait)
            except Exception as exc:
                print(f"[warn] unexpected error: {exc}")
                if attempt == 3:
                    raise
                wait = backoff_seconds[attempt - 1]
                print(f"[info] reconnecting in {wait}s...")
                await asyncio.sleep(wait)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    # Loads `.env` from cwd (or parents, per python-dotenv), then reads resume next to this file.
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    neon_code = os.getenv("NEON_CODE", "").strip()
    if not google_api_key:
        raise SystemExit("Missing GOOGLE_API_KEY in .env")
    if not neon_code:
        raise SystemExit("Missing NEON_CODE in .env")

    resume_path = os.path.join(os.path.dirname(__file__), "resume.txt")
    if not os.path.exists(resume_path):
        raise SystemExit("Missing resume.txt")
    resume_text = load_text_file(resume_path)
    if not resume_text.strip():
        raise SystemExit("resume.txt is empty")

    copilot = NeonCopilot(neon_code=neon_code, resume_text=resume_text, gemini_api_key=google_api_key)
    asyncio.run(copilot.run())


if __name__ == "__main__":
    main()
