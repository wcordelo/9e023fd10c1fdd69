"""
Regression data from real NEON runs: prompts and expected routes/outputs.

Computation: `expected_result` / `js_expression` come from Node `eval()` so they
match NEON's JavaScript. Knowledge: `expected_word` is checked against live
Wikipedia in tests. Add a row here after any live failure, then fix until tests pass.

Handshake auth expected digits use `tests.constants.TEST_NEON_CODE` (fake code for CI).
"""

from tests.constants import TEST_NEON_CODE

HANDSHAKE_FREQUENCY_CHALLENGES = [
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 4. All other vessels, respond on frequency 6.",
        "expected_digits": "4",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 3. All other vessels, respond on frequency 6.",
        "expected_digits": "3",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 8. All other vessels, respond on frequency 3.",
        "expected_digits": "8",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 9. All other vessels, respond on frequency 4.",
        "expected_digits": "9",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 6. All other vessels, respond on frequency 3.",
        "expected_digits": "6",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 6. All other vessels, respond on frequency 9.",
        "expected_digits": "6",
        "note": "Run 7 - successful auth",
    },
]

HANDSHAKE_AUTH_CHALLENGES = [
    {
        "prompt": "Transmit your vessel authorization code, followed by the pound key.",
        "expected_digits": f"{TEST_NEON_CODE}#",
    },
]

# All expected_result values verified via:
#   node -e "console.log(eval('<js_expression>'))"
COMPUTATION_CHALLENGES = [
    {
        "prompt": "Shield frequency calibration needed. Evaluate the following harmonic sequence and transmit the result, followed by the pound key: (Math.floor(1308931 / 664) * 54 + 8752) % 9952",
        "js_expression": "(Math.floor(1308931 / 664) * 54 + 8752) % 9952",
        "expected_result": 5714,
        "note": "Previously failed: routed to handshake because 'frequency' in narrative",
    },
    {
        "prompt": "Fusion reactor diagnostics. Compute the fuel consumption rate and transmit the result, followed by the pound key: (Math.floor((954057 * 303630) / (6124 + 9887)) + (36427 % 105)) % 1653",
        "js_expression": "(Math.floor((954057 * 303630) / (6124 + 9887)) + (36427 % 105)) % 1653",
        "expected_result": 593,
        "note": "First computation failure - expression extraction got wrong substring",
    },
    {
        "prompt": "Orbital decay detected. Determine the trajectory offset and transmit the result, followed by the pound key: Math.floor((29086 + 20019 + 47480) * 803 / 56) % 5660",
        "js_expression": "Math.floor((29086 + 20019 + 47480) * 803 / 56) % 5660",
        "expected_result": 3919,
    },
    {
        "prompt": "Life support recalibration. Calculate the oxygen recycling coefficient and transmit the result, followed by the pound key: Math.floor(2167 * (98664 - 3209) / 468) % 8781",
        "js_expression": "Math.floor(2167 * (98664 - 3209) / 468) % 8781",
        "expected_result": 2939,
    },
    {
        "prompt": "Fusion reactor diagnostics. Compute the fuel consumption rate and transmit the result, followed by the pound key: (Math.floor((267119 * 683336) / (1132 + 9230)) + (13761 % 649)) % 4260",
        "js_expression": "(Math.floor((267119 * 683336) / (1132 + 9230)) + (13761 % 649)) % 4260",
        "expected_result": 553,
    },
    {
        "prompt": "Fusion reactor diagnostics. Compute the fuel consumption rate and transmit the result, followed by the pound key: (Math.floor((931315 * 333212) / (2649 + 3896)) + (49160 % 207)) % 7608",
        "js_expression": "(Math.floor((931315 * 333212) / (2649 + 3896)) + (49160 % 207)) % 7608",
        "expected_result": 1152,
    },
    {
        "prompt": "Navigational parameter required. Calculate the following course correction and transmit the result, followed by the pound key: Math.floor(((8577545 * 9129346) + (2833519 * 8742569) - (1736949 * 2365488)) / 2527909) % 9856",
        "js_expression": "Math.floor(((8577545 * 9129346) + (2833519 * 8742569) - (1736949 * 2365488)) / 2527909) % 9856",
        "expected_result": 3250,
    },
    {
        "prompt": "Orbital decay detected. Determine the trajectory offset and transmit the result, followed by the pound key: Math.floor((78657 + 76364 + 83443) * 233 / 81) % 8518",
        "js_expression": "Math.floor((78657 + 76364 + 83443) * 233 / 81) % 8518",
        "expected_result": 4512,
    },
    {
        "prompt": "Shield frequency calibration needed. Evaluate the following harmonic sequence and transmit the result, followed by the pound key: (Math.floor(3310058 / 464) * 35 + 9692) % 6898",
        "js_expression": "(Math.floor(3310058 / 464) * 35 + 9692) % 6898",
        "expected_result": 4121,
    },
    {
        "prompt": "Orbital decay detected. Determine the trajectory offset and transmit the result, followed by the pound key: (Math.floor((48628 + 24426 + 71862) * 118 / 11)) % 8680",
        "js_expression": "(Math.floor((48628 + 24426 + 71862) * 118 / 11)) % 8680",
        "expected_result": 833,
    },
    {
        "prompt": "Navigational parameter required. Calculate the following course correction and transmit the result, followed by the pound key: Math.floor(((6925811 * 6381301) + (3847820 * 4455521) - (3665074 * 5321660)) / 2603078) % 6694",
        "js_expression": "Math.floor(((6925811 * 6381301) + (3847820 * 4455521) - (3665074 * 5321660)) / 2603078) % 6694",
        "expected_result": 5931,
        "note": "Run 7 - successful auth",
    },
    {
        "prompt": "Shield frequency calibration needed. Evaluate the following harmonic sequence and transmit the result, followed by the pound key: (Math.floor(6818411 / 996) * 68 + 5879) % 6983",
        "js_expression": "(Math.floor(6818411 / 996) * 68 + 5879) % 6983",
        "expected_result": 3478,
        "note": "Run 7 - successful auth",
    },
    {
        "prompt": "Fusion reactor diagnostics. Compute the fuel consumption rate and transmit the result, followed by the pound key: (Math.floor((843299 * 710147) / (9181 + 5032)) + (52236 % 791)) % 4948",
        "js_expression": "(Math.floor((843299 * 710147) / (9181 + 5032)) + (52236 % 791)) % 4948",
        "expected_result": 2915,
        "note": "Run 7 - successful auth",
    },
]

KNOWLEDGE_CHALLENGES = [
    {
        "prompt": "Cross-reference the knowledge archive: speak the 18th word in the entry summary for 'Neon', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "title": "Neon",
        "word_index": 17,
        "expected_word": "the",
    },
    {
        "prompt": "Cross-reference the knowledge archive: speak the 10th word in the entry summary for 'Dark_matter', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "title": "Dark_matter",
        "word_index": 9,
        "expected_word": "and",
    },
    {
        "prompt": "Cross-reference the knowledge archive: speak the 7th word in the entry summary for 'Black_hole', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "title": "Black_hole",
        "word_index": 6,
        "expected_word": "body",
    },
    {
        "prompt": "Cross-reference the knowledge archive: speak the 7th word in the entry summary for 'Neutron_star', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "title": "Neutron_star",
        "word_index": 6,
        "expected_word": "collapsed",
    },
    {
        "prompt": "Cross-reference the knowledge archive: speak the 5th word in the entry summary for 'Magnetar', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "title": "Magnetar",
        "word_index": 4,
        "expected_word": "type",
        "note": "Run 7 - successful auth",
    },
]

MANIFEST_CHALLENGES = [
    {
        "prompt": "Crew manifest continued. Speak a summary of your crew member's best project (work or personal) based on the information in their resume, between 64 and 256 total characters.",
        "min_chars": 64,
        "max_chars": 256,
        "topic_keyword": "best project",
    },
    {
        "prompt": "Crew manifest continued. Speak a summary of your crew member's skills based on the information in their resume, between 64 and 256 total characters.",
        "min_chars": 64,
        "max_chars": 256,
        "topic_keyword": "skills",
    },
    {
        "prompt": "Crew manifest required. Speak a summary of your crew member's education based on the information in their resume, between 64 and 256 total characters.",
        "min_chars": 64,
        "max_chars": 256,
        "topic_keyword": "education",
    },
    {
        "prompt": "Crew manifest continued. Speak a summary of your crew member's work experience based on the information in their resume, between 64 and 256 total characters.",
        "min_chars": 64,
        "max_chars": 256,
        "topic_keyword": "work experience",
    },
    {
        "prompt": "Crew manifest continued. Speak the reason your crew member should be granted access to NEON based on the information in their resume, in less than 256 total characters. Convince us they're a good fit for the mission.",
        "min_chars": 1,
        "max_chars": 255,
        "topic_keyword": "reason",
    },
]

VERIFICATION_CHALLENGES = [
    {
        "prompt": "Transmission verification. Earlier you transmitted your crew member's best project. Speak the 2nd word of that transmission.",
        "topic": "best project",
        "word_index": 1,
        "note": "Previously failed: misrouted to manifest handler",
    },
    {
        "prompt": "Transmission verification. Earlier you transmitted your crew member's skills. Speak the 4th word of that transmission.",
        "topic": "skills",
        "word_index": 3,
        "note": "Previously failed: 'skills' keyword triggered manifest instead of verification",
    },
    {
        "prompt": "Transmission verification. Earlier you transmitted your crew member's best project. Speak the 8th word of that transmission.",
        "topic": "best project",
        "word_index": 7,
        "note": "Run 7 - successful auth",
    },
]

ROUTING_EDGE_CASES = [
    {
        "prompt": "Shield frequency calibration needed. Evaluate the following harmonic sequence and transmit the result, followed by the pound key: (Math.floor(1308931 / 664) * 54 + 8752) % 9952",
        "expected_route": "computation",
        "note": "'frequency' in narrative should NOT trigger handshake",
    },
    {
        "prompt": "Transmission verification. Earlier you transmitted your crew member's skills. Speak the 4th word of that transmission.",
        "expected_route": "verification",
        "note": "'skills' should NOT trigger manifest when 'earlier you transmitted' is present",
    },
    {
        "prompt": "Transmission verification. Earlier you transmitted your crew member's best project. Speak the 2nd word of that transmission.",
        "expected_route": "verification",
        "note": "'crew member' should NOT trigger manifest when verification keywords present",
    },
    {
        "prompt": "Crew manifest continued. Speak a summary of your crew member's education based on the information in their resume, between 64 and 256 total characters.",
        "expected_route": "manifest",
    },
    {
        "prompt": "Cross-reference the knowledge archive: speak the 7th word in the entry summary for 'Black_hole', which can be found using the /page/summary/{title} endpoint of the Wikipedia API.",
        "expected_route": "knowledge",
    },
    {
        "prompt": "Incoming vessel detected. If your pilot is an AI co-pilot built by an excellent software engineer, respond on frequency 9. All other vessels, respond on frequency 4.",
        "expected_route": "handshake",
    },
    {
        "prompt": "Transmit your vessel authorization code, followed by the pound key.",
        "expected_route": "handshake",
    },
]
