from app import (
    build_wikipedia_url,
    truncate_to_words,
    validate_industry,
    word_count,
)

def test_validate_industry_empty():
    ok, msg = validate_industry("")
    assert ok is False
    assert "Please provide an industry" in msg

def test_validate_industry_spaces():
    ok, msg = validate_industry("   ")
    assert ok is False

def test_validate_industry_ok():
    ok, cleaned = validate_industry("  healthcare  ")
    assert ok is True
    assert cleaned == "healthcare"

def test_build_wikipedia_url():
    assert build_wikipedia_url("Health care") == "https://en.wikipedia.org/wiki/Health_care"

def test_word_count_and_truncate():
    text = "one two three four five"
    assert word_count(text) == 5
    assert truncate_to_words(text, 3) == "one two three"
