"""
Offline test suite for the NEON copilot.
Run with: python -m pytest tests/test_copilot.py -v
All tests run without network calls to NEON (some hit Wikipedia).

IMPORTANT: All computation expected values are verified against actual
JavaScript execution via Node.js to ensure parity with NEON's evaluator.
"""
import math
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from copilot import NeonCopilot
from tests.fixtures import (
    HANDSHAKE_FREQUENCY_CHALLENGES,
    HANDSHAKE_AUTH_CHALLENGES,
    COMPUTATION_CHALLENGES,
    KNOWLEDGE_CHALLENGES,
    MANIFEST_CHALLENGES,
    VERIFICATION_CHALLENGES,
    ROUTING_EDGE_CASES,
)

NEON_CODE = "9e023fd10c1fdd69"
RESUME = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resume.txt")).read()


@pytest.fixture
def copilot():
    return NeonCopilot(neon_code=NEON_CODE, resume_text=RESUME, gemini_api_key="dummy")


@pytest.fixture
def copilot_with_gemini():
    """Real Gemini client for manifest tests."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return NeonCopilot(neon_code=NEON_CODE, resume_text=RESUME, gemini_api_key=key)


# ── Routing ───────────────────────────────────────────────────────────

class TestRouting:
    @pytest.mark.parametrize("case", ROUTING_EDGE_CASES, ids=lambda c: c.get("note", c["prompt"][:50]))
    def test_routing_edge_cases(self, copilot, case):
        result = copilot.classify_checkpoint(case["prompt"])
        assert result == case["expected_route"], (
            f"Routed to '{result}' but expected '{case['expected_route']}'\n"
            f"Prompt: {case['prompt'][:80]}..."
        )

    def test_computation_not_hijacked_by_handshake(self, copilot):
        """'frequency' in narrative must not trigger handshake."""
        prompt = (
            "Shield frequency calibration needed. Evaluate the following harmonic "
            "sequence and transmit the result, followed by the pound key: "
            "(Math.floor(1308931 / 664) * 54 + 8752) % 9952"
        )
        assert copilot.classify_checkpoint(prompt) == "computation"

    def test_verification_not_hijacked_by_manifest(self, copilot):
        """'skills' / 'crew member' in verification prompt must not trigger manifest."""
        prompt = (
            "Transmission verification. Earlier you transmitted your crew member's "
            "skills. Speak the 4th word of that transmission."
        )
        assert copilot.classify_checkpoint(prompt) == "verification"


# ── JS-Python Computation Parity ─────────────────────────────────────
# These tests verify our Python Math converter produces the EXACT same
# integer result as JavaScript's eval(). Expected values were generated
# by running: node -e "console.log(eval('<js_expression>'))"

class TestJSPythonParity:
    """Directly evaluate the raw JS expression through our Python converter
    and compare against the Node.js-verified expected result."""

    @pytest.mark.parametrize(
        "case",
        COMPUTATION_CHALLENGES,
        ids=lambda c: c["js_expression"][:50],
    )
    def test_js_expression_parity(self, copilot, case):
        js_expr = case["js_expression"]
        expected = case["expected_result"]
        result = copilot.eval_expression_python(js_expr)
        assert result == expected, (
            f"Python eval gave {result} but JavaScript gives {expected}\n"
            f"Expression: {js_expr}"
        )

    def test_math_floor_positive(self, copilot):
        # JS: Math.floor(7.9) => 7
        assert copilot.eval_expression_python("Math.floor(7.9)") == 7

    def test_math_floor_negative(self, copilot):
        # JS: Math.floor(-4.2) => -5 (floor, not truncate)
        assert copilot.eval_expression_python("Math.floor(-4.2)") == -5

    def test_math_floor_exact_int(self, copilot):
        # JS: Math.floor(10.0) => 10
        assert copilot.eval_expression_python("Math.floor(10.0)") == 10

    def test_modulo_positive(self, copilot):
        # JS: 17 % 5 => 2 (same as Python for positive)
        assert copilot.eval_expression_python("17 % 5") == 2

    def test_integer_division_floored(self, copilot):
        # JS: Math.floor(10 / 3) => 3
        assert copilot.eval_expression_python("Math.floor(10 / 3)") == 3

    def test_combined_floor_modulo(self, copilot):
        # JS: Math.floor(100 / 7) % 9 => Math.floor(14.28) % 9 => 14 % 9 => 5
        assert copilot.eval_expression_python("Math.floor(100 / 7) % 9") == 5

    def test_nested_parentheses(self, copilot):
        # JS: Math.floor(((100 + 200) * 3) / 7) => Math.floor(900/7) => 128
        assert copilot.eval_expression_python("Math.floor(((100 + 200) * 3) / 7)") == 128

    def test_large_product_within_double(self, copilot):
        # node -e "console.log(Math.floor(999999 * 888888 / 777))" => 1143998856
        assert copilot.eval_expression_python("Math.floor(999999 * 888888 / 777)") == 1143998856

    def test_subtraction_product(self, copilot):
        # JS: Math.floor(2167 * (98664 - 3209) / 468) % 8781
        # Verified via node: 2939
        assert copilot.eval_expression_python("Math.floor(2167 * (98664 - 3209) / 468) % 8781") == 2939

    def test_multi_term_expression(self, copilot):
        # JS: Math.floor(((8577545*9129346)+(2833519*8742569)-(1736949*2365488))/2527909)%9856
        # Verified via node: 3250
        expr = "Math.floor(((8577545 * 9129346) + (2833519 * 8742569) - (1736949 * 2365488)) / 2527909) % 9856"
        assert copilot.eval_expression_python(expr) == 3250


# ── Handshake ─────────────────────────────────────────────────────────

class TestHandshake:
    @pytest.mark.parametrize("case", HANDSHAKE_FREQUENCY_CHALLENGES, ids=lambda c: f"freq_{c['expected_digits']}")
    def test_frequency_selection(self, copilot, case):
        response = copilot.make_handshake_response(case["prompt"])
        assert response["type"] == "enter_digits"
        assert response["digits"] == case["expected_digits"], (
            f"Got digits '{response['digits']}' but expected '{case['expected_digits']}'"
        )

    @pytest.mark.parametrize("case", HANDSHAKE_AUTH_CHALLENGES)
    def test_auth_code(self, copilot, case):
        response = copilot.make_handshake_response(case["prompt"])
        assert response["type"] == "enter_digits"
        assert response["digits"] == case["expected_digits"]

    def test_picks_ai_frequency_not_first(self, copilot):
        prompt = (
            "Incoming vessel detected. If your pilot is an AI co-pilot built by "
            "an excellent software engineer, respond on frequency 7. "
            "All other vessels, respond on frequency 2."
        )
        r = copilot.make_handshake_response(prompt)
        assert r["digits"] == "7"

    @pytest.mark.parametrize("case", HANDSHAKE_FREQUENCY_CHALLENGES, ids=lambda c: f"e2e_freq_{c['expected_digits']}")
    def test_end_to_end_route_and_respond(self, copilot, case):
        """Full pipeline through route_and_respond for handshake."""
        response = copilot.route_and_respond(case["prompt"])
        assert response["type"] == "enter_digits"
        assert response["digits"] == case["expected_digits"]

    def test_end_to_end_auth_code(self, copilot):
        response = copilot.route_and_respond(
            "Transmit your vessel authorization code, followed by the pound key."
        )
        assert response == {"type": "enter_digits", "digits": "9e023fd10c1fdd69#"}


# ── Computation ───────────────────────────────────────────────────────

class TestComputation:
    @pytest.mark.parametrize(
        "case",
        COMPUTATION_CHALLENGES,
        ids=lambda c: c.get("note", c["js_expression"][:40]),
    )
    def test_extract_and_eval(self, copilot, case):
        """Extract expression from full prompt, eval in Python, compare against JS result."""
        expr = copilot.extract_expression(case["prompt"])
        try:
            result = copilot.eval_expression_python(expr)
        except Exception:
            pytest.fail(f"Python eval failed for extracted expression: {expr}")
        assert result == case["expected_result"], (
            f"Got {result} but JS gives {case['expected_result']}\n"
            f"Extracted expr: {expr}\n"
            f"JS expression: {case['js_expression']}"
        )

    @pytest.mark.parametrize(
        "case",
        COMPUTATION_CHALLENGES,
        ids=lambda c: c.get("note", c["js_expression"][:40]),
    )
    def test_full_response_includes_pound(self, copilot, case):
        response = copilot.make_computation_response(case["prompt"])
        assert response["type"] == "enter_digits"
        if "pound key" in case["prompt"].lower():
            assert response["digits"].endswith("#"), f"Missing # suffix: {response['digits']}"
            numeric = response["digits"].rstrip("#")
        else:
            numeric = response["digits"]
        assert int(numeric) == case["expected_result"]

    def test_expression_with_no_colon(self, copilot):
        # JS: Math.floor((7 * 3 + 2) / 5) => 4 (verified via node)
        prompt = "Calculate Math.floor((7 * 3 + 2) / 5) and transmit the result followed by the pound key."
        response = copilot.make_computation_response(prompt)
        assert response["digits"].rstrip("#") == "4"

    @pytest.mark.parametrize(
        "case",
        COMPUTATION_CHALLENGES,
        ids=lambda c: c.get("note", c["js_expression"][:40]),
    )
    def test_end_to_end_route_and_respond(self, copilot, case):
        """Full pipeline: route_and_respond classifies, extracts, evals,
        formats, and validates -- compare against JS-verified result."""
        response = copilot.route_and_respond(case["prompt"])
        assert response["type"] == "enter_digits"
        numeric = response["digits"].rstrip("#")
        assert int(numeric) == case["expected_result"], (
            f"route_and_respond gave {numeric} but JS gives {case['expected_result']}"
        )


# ── Expression Extraction ────────────────────────────────────────────

class TestExpressionExtraction:
    """Verify that extract_expression pulls the correct JS expression
    from various prompt formats."""

    @pytest.mark.parametrize(
        "case",
        COMPUTATION_CHALLENGES,
        ids=lambda c: c["js_expression"][:40],
    )
    def test_extracted_matches_js(self, copilot, case):
        extracted = copilot.extract_expression(case["prompt"])
        extracted_eval = copilot.eval_expression_python(extracted)
        js_eval = case["expected_result"]
        assert extracted_eval == js_eval, (
            f"Extracted '{extracted}' evaluates to {extracted_eval} "
            f"but JS expression '{case['js_expression']}' = {js_eval}"
        )

    def test_after_last_colon(self, copilot):
        prompt = "Some narrative: more narrative: Math.floor(10 / 3)"
        expr = copilot.extract_expression(prompt)
        assert copilot.eval_expression_python(expr) == 3

    def test_with_backticks(self, copilot):
        prompt = "Evaluate: `Math.floor(100 / 7) % 9`"
        expr = copilot.extract_expression(prompt)
        assert copilot.eval_expression_python(expr) == 5


# ── Length Constraints ────────────────────────────────────────────────

class TestLengthConstraints:
    def test_between(self, copilot):
        c = copilot.parse_length_constraints("between 64 and 256 total characters")
        assert c.min_chars == 64 and c.max_chars == 256

    def test_less_than(self, copilot):
        c = copilot.parse_length_constraints("in less than 256 total characters")
        assert c.min_chars == 1 and c.max_chars == 255

    def test_under(self, copilot):
        c = copilot.parse_length_constraints("under 200 characters")
        assert c.max_chars == 199

    def test_fewer_than(self, copilot):
        c = copilot.parse_length_constraints("fewer than 128 characters")
        assert c.max_chars == 127

    def test_at_most(self, copilot):
        c = copilot.parse_length_constraints("at most 100 characters")
        assert c.min_chars == 1 and c.max_chars == 100

    def test_exactly(self, copilot):
        c = copilot.parse_length_constraints("exactly 50 characters")
        assert c.min_chars == 50 and c.max_chars == 50

    def test_at_least(self, copilot):
        c = copilot.parse_length_constraints("at least 30 characters")
        assert c.min_chars == 30 and c.max_chars == 256

    def test_no_constraint(self, copilot):
        c = copilot.parse_length_constraints("Speak something about the crew.")
        assert c.min_chars == 1 and c.max_chars == 256

    @pytest.mark.parametrize("case", MANIFEST_CHALLENGES, ids=lambda c: c["topic_keyword"])
    def test_manifest_constraint_parsing(self, copilot, case):
        c = copilot.parse_length_constraints(case["prompt"])
        assert c.min_chars == case["min_chars"], f"min_chars: got {c.min_chars}, expected {case['min_chars']}"
        assert c.max_chars == case["max_chars"], f"max_chars: got {c.max_chars}, expected {case['max_chars']}"


# ── Knowledge Archive ─────────────────────────────────────────────────

class TestKnowledge:
    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_title_parsing(self, copilot, case):
        title = copilot.parse_title(case["prompt"])
        assert title is not None, f"Failed to parse title from: {case['prompt'][:60]}"
        assert title.replace("_", " ").lower() == case["title"].replace("_", " ").lower()

    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_word_index_parsing(self, copilot, case):
        idx = copilot.parse_ordinal_index(case["prompt"])
        assert idx == case["word_index"], f"Got index {idx} but expected {case['word_index']}"

    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_wiki_fetch(self, copilot, case):
        summary = copilot.wiki_summary(case["title"])
        assert summary is not None, f"Failed to fetch summary for {case['title']}"
        words = copilot.safe_words(summary)
        assert len(words) > case["word_index"], (
            f"Summary for {case['title']} has {len(words)} words but need index {case['word_index']}"
        )

    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_full_knowledge_response(self, copilot, case):
        response = copilot.make_knowledge_response(case["prompt"])
        assert response["type"] == "speak_text"
        assert len(response["text"]) > 0
        assert len(response["text"]) <= 256

    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_exact_word_from_wikipedia(self, copilot, case):
        """End-to-end: fetch from live Wikipedia, extract Nth word,
        verify it matches the expected word (verified manually)."""
        response = copilot.make_knowledge_response(case["prompt"])
        assert response["text"] == case["expected_word"], (
            f"For '{case['title']}' word {case['word_index']}: "
            f"got '{response['text']}' but expected '{case['expected_word']}'"
        )

    @pytest.mark.parametrize("case", KNOWLEDGE_CHALLENGES, ids=lambda c: c["title"])
    def test_end_to_end_route_and_respond(self, copilot, case):
        """Full pipeline through route_and_respond for knowledge."""
        response = copilot.route_and_respond(case["prompt"])
        assert response["type"] == "speak_text"
        assert response["text"] == case["expected_word"]


# ── Ordinal Parsing ───────────────────────────────────────────────────

class TestOrdinalParsing:
    @pytest.mark.parametrize("text,expected", [
        ("Speak the 1st word", 0),
        ("Speak the 2nd word", 1),
        ("Speak the 3rd word", 2),
        ("Speak the 4th word", 3),
        ("Speak the 10th word", 9),
        ("Speak the 18th word", 17),
        ("Speak the 21st word", 20),
    ])
    def test_numeric_ordinals(self, copilot, text, expected):
        assert copilot.parse_ordinal_index(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("Speak the first word", 0),
        ("Speak the second word", 1),
        ("Speak the fifth word", 4),
        ("Speak the twentieth word", 19),
    ])
    def test_word_ordinals(self, copilot, text, expected):
        assert copilot.parse_ordinal_index(text) == expected

    def test_does_not_grab_random_numbers(self, copilot):
        prompt = "speak the 5th word in the entry summary for '2001 A Space Odyssey'"
        assert copilot.parse_ordinal_index(prompt) == 4


# ── Verification ──────────────────────────────────────────────────────

class TestVerification:
    def _setup_memory(self, copilot):
        copilot.session_memory = [
            {
                "checkpoint": "manifest",
                "prompt": "Speak a summary of your crew member's best project (work or personal) based on the information in their resume, between 64 and 256 total characters.",
                "response": "At Bello, William built a full-stack platform serving 60K+ users, led AWS/GCP infrastructure, and drove $2M+ in revenue through the launch of the Blueprint Protocol.",
            },
            {
                "checkpoint": "manifest",
                "prompt": "Speak a summary of your crew member's skills based on the information in their resume, between 64 and 256 total characters.",
                "response": "William Lopez-Cordero is a Full Stack Engineer skilled in TypeScript, Go, Python, and SQL with React, Next.js, Node.js, AWS, GCP, Azure, Docker, and Pulumi expertise.",
            },
            {
                "checkpoint": "manifest",
                "prompt": "Speak a summary of your crew member's education based on the information in their resume, between 64 and 256 total characters.",
                "response": "William Lopez-Cordero earned a Bachelor of Science in Aerospace Engineering from the Massachusetts Institute of Technology (MIT).",
            },
            {
                "checkpoint": "manifest",
                "prompt": "Speak a summary of your crew member's work experience based on the information in their resume, between 64 and 256 total characters.",
                "response": "William has worked at Bello, Elphi, Facebook, and Google building scalable platforms and cloud infrastructure serving millions of users.",
            },
        ]

    def test_best_project_2nd_word(self, copilot):
        self._setup_memory(copilot)
        prompt = "Transmission verification. Earlier you transmitted your crew member's best project. Speak the 2nd word of that transmission."
        r = copilot.make_verification_response(prompt)
        words = copilot.safe_words(copilot.session_memory[0]["response"])
        expected = copilot.strip_word_punctuation(words[1])
        assert r["text"] == expected, f"Got '{r['text']}' but expected '{expected}'"

    def test_skills_4th_word(self, copilot):
        self._setup_memory(copilot)
        prompt = "Transmission verification. Earlier you transmitted your crew member's skills. Speak the 4th word of that transmission."
        r = copilot.make_verification_response(prompt)
        words = copilot.safe_words(copilot.session_memory[1]["response"])
        expected = copilot.strip_word_punctuation(words[3])
        assert r["text"] == expected, f"Got '{r['text']}' but expected '{expected}'"

    def test_education_1st_word(self, copilot):
        self._setup_memory(copilot)
        prompt = "Transmission verification. Earlier you transmitted your crew member's education. Speak the 1st word of that transmission."
        r = copilot.make_verification_response(prompt)
        words = copilot.safe_words(copilot.session_memory[2]["response"])
        expected = copilot.strip_word_punctuation(words[0])
        assert r["text"] == expected, f"Got '{r['text']}' but expected '{expected}'"

    def test_work_experience_3rd_word(self, copilot):
        self._setup_memory(copilot)
        prompt = "Transmission verification. Earlier you transmitted your crew member's work experience. Speak the 3rd word of that transmission."
        r = copilot.make_verification_response(prompt)
        words = copilot.safe_words(copilot.session_memory[3]["response"])
        expected = copilot.strip_word_punctuation(words[2])
        assert r["text"] == expected, f"Got '{r['text']}' but expected '{expected}'"

    def test_topic_matching_finds_correct_memory(self, copilot):
        self._setup_memory(copilot)
        prompt = "Transmission verification. Earlier you transmitted your crew member's skills. Speak the 1st word of that transmission."
        r = copilot.make_verification_response(prompt)
        skills_first = copilot.strip_word_punctuation(copilot.safe_words(copilot.session_memory[1]["response"])[0])
        assert r["text"] == skills_first


# ── Response Format Guard ─────────────────────────────────────────────

class TestResponseGuard:
    def test_valid_enter_digits(self, copilot):
        r = copilot.validate_response({"type": "enter_digits", "digits": "1234#"})
        assert r == {"type": "enter_digits", "digits": "1234#"}

    def test_strips_invalid_chars(self, copilot):
        r = copilot.validate_response({"type": "enter_digits", "digits": "12!@34#"})
        assert r == {"type": "enter_digits", "digits": "1234#"}

    def test_valid_speak_text(self, copilot):
        r = copilot.validate_response({"type": "speak_text", "text": "hello"})
        assert r == {"type": "speak_text", "text": "hello"}

    def test_truncates_long_text(self, copilot):
        long = "x" * 300
        r = copilot.validate_response({"type": "speak_text", "text": long})
        assert len(r["text"]) <= 256

    def test_rejects_invalid_type(self, copilot):
        with pytest.raises(ValueError):
            copilot.validate_response({"type": "bad_type", "text": "hello"})

    def test_rejects_empty_digits(self, copilot):
        with pytest.raises(ValueError):
            copilot.validate_response({"type": "enter_digits", "digits": "!!!"})

    def test_rejects_empty_text(self, copilot):
        with pytest.raises(ValueError):
            copilot.validate_response({"type": "speak_text", "text": ""})


# ── Manifest (requires Gemini API key) ────────────────────────────────

class TestManifest:
    @pytest.mark.parametrize("case", MANIFEST_CHALLENGES, ids=lambda c: c["topic_keyword"])
    def test_manifest_response_length(self, copilot_with_gemini, case):
        response = copilot_with_gemini.make_manifest_response(case["prompt"])
        assert response["type"] == "speak_text"
        text = response["text"]
        assert case["min_chars"] <= len(text) <= case["max_chars"], (
            f"Length {len(text)} out of range [{case['min_chars']}, {case['max_chars']}]\n"
            f"Response: {text}"
        )


# ── Message Reconstruction ───────────────────────────────────────────

class TestReconstruction:
    def test_basic(self, copilot):
        fragments = [
            {"word": "2", "timestamp": 1},
            {"word": "plus", "timestamp": 2},
            {"word": "What's", "timestamp": 0},
            {"word": "3?", "timestamp": 3},
        ]
        assert copilot.reconstruct_message(fragments) == "What's 2 plus 3?"

    def test_string_fallback(self, copilot):
        assert copilot.reconstruct_message("plain string") == "plain string"

    def test_empty(self, copilot):
        assert copilot.reconstruct_message([]) == ""


# ── Text Clamping ─────────────────────────────────────────────────────

class TestClamping:
    def test_truncates_at_word_boundary(self, copilot):
        text = "This is a test sentence that is too long"
        result = copilot.clamp_text(text, 1, 20)
        assert len(result) <= 20
        assert not result.endswith(" ")

    def test_pads_when_too_short(self, copilot):
        text = "Short"
        result = copilot.clamp_text(text, 20, 100)
        assert len(result) >= 20
