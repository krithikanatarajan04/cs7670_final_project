import pytest

from pipeline.cfg import record_transition, reset_trace


# ------------------------------------------------------------
# Test 1 — Correct sequence passes
# ------------------------------------------------------------

def test_correct_sequence_passes():
    reset_trace()

    record_transition("Researcher")
    record_transition("Analyzer")
    record_transition("Verifier")
    record_transition("RecommendationAgent")

    # If no exception → test passes


# ------------------------------------------------------------
# Test 2 — Skipped agent is caught
# ------------------------------------------------------------

def test_skipped_agent_detected():
    reset_trace()

    record_transition("Researcher")

    with pytest.raises(RuntimeError):
        record_transition("Verifier")   # Skips Analyzer


# ------------------------------------------------------------
# Test 3 — Repeated agent is caught
# ------------------------------------------------------------

def test_repeated_agent_detected():
    reset_trace()

    record_transition("Researcher")

    with pytest.raises(RuntimeError):
        record_transition("Researcher")   # Repeat


# ------------------------------------------------------------
# Test 4 — Reversed sequence is caught
# ------------------------------------------------------------

def test_reversed_sequence_detected():
    reset_trace()

    with pytest.raises(RuntimeError):
        record_transition("RecommendationAgent")   # Cannot start here


# ------------------------------------------------------------
# Test 5 — Reset works correctly
# ------------------------------------------------------------

def test_reset_allows_fresh_run():
    reset_trace()

    # First run
    record_transition("Researcher")
    record_transition("Analyzer")
    record_transition("Verifier")
    record_transition("RecommendationAgent")

    # Reset
    reset_trace()

    # Second run must be independent
    record_transition("Researcher")
    record_transition("Analyzer")
    record_transition("Verifier")
    record_transition("RecommendationAgent")