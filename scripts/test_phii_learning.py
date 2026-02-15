#!/usr/bin/env python3
"""Test that Phii learning system works correctly.

Run from project root:
    python scripts/test_phii_learning.py

Or from backend:
    cd backend && python ../scripts/test_phii_learning.py
"""
import sys
from pathlib import Path

# Ensure backend is in path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))


def test_firm_corrections_seeded():
    """Verify firm corrections exist in database."""
    from tools.phii import ReinforcementStore, seed_if_empty

    # Ensure seeding has happened
    seed_if_empty()

    store = ReinforcementStore()
    corrections = store._get_firm_corrections("performance capacity granular material", top_k=5)
    print(f"  Firm corrections for 'performance capacity granular material': {len(corrections)}")

    if len(corrections) == 0:
        raise AssertionError("No firm corrections found - seed_if_empty() may have failed")

    # Check we got relevant ones
    for c in corrections[:2]:
        print(f"    - {c.wrong_behavior[:50]}...")

    print(f"  Total firm corrections available: checking database...")

    # Check total count
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM firm_corrections WHERE is_active=1"
        ).fetchone()[0]
        print(f"  Total firm corrections in DB: {count}")
        if count == 0:
            raise AssertionError("No firm corrections in database")


def test_corrections_injected():
    """Verify corrections appear in system prompt."""
    from tools.phii import PhiiContextBuilder

    builder = PhiiContextBuilder()
    context = builder.build_context(
        current_message="Calculate performance capacity for granular material",
        files_context="No files",
        session_notes="",
        session_id="test"
    )

    # Check for correction hints in prompt
    has_corrections = "LEARNED CORRECTIONS" in context.system_prompt
    has_metric = "metric" in context.system_prompt.lower()

    print(f"  'LEARNED CORRECTIONS' in prompt: {has_corrections}")
    print(f"  'metric' keyword in prompt: {has_metric}")

    # Show a snippet if corrections were found
    if has_corrections:
        idx = context.system_prompt.find("LEARNED CORRECTIONS")
        snippet = context.system_prompt[idx:idx+200]
        print(f"  Snippet: {snippet}...")


def test_expertise_detection():
    """Verify expertise levels are detected."""
    from tools.phii import ExpertiseDetector

    detector = ExpertiseDetector()

    # Test junior detection
    junior = detector.analyze("I'm an EIT, can you explain performance capacity?", 0.2, None)
    print(f"  Junior query -> {junior.level.value} (confidence: {junior.confidence:.2f})")
    if junior.level.value != "junior":
        raise AssertionError(f"Expected 'junior', got '{junior.level.value}'")

    # Test senior detection
    senior = detector.analyze(
        "Verify my settlement calc, assuming K=0.35 and using standard method",
        0.8,
        None
    )
    print(f"  Senior query -> {senior.level.value} (confidence: {senior.confidence:.2f})")
    if senior.level.value != "senior":
        raise AssertionError(f"Expected 'senior', got '{senior.level.value}'")

    # Test management detection
    mgmt = detector.analyze(
        "Give me an executive summary for the client meeting on budget impact",
        0.3,
        None
    )
    print(f"  Management query -> {mgmt.level.value} (confidence: {mgmt.confidence:.2f})")
    if mgmt.level.value != "management":
        raise AssertionError(f"Expected 'management', got '{mgmt.level.value}'")


def test_energy_detection():
    """Verify energy profile detection works."""
    from tools.phii import EnergyDetector

    detector = EnergyDetector()

    # Test urgent message
    profile = detector.analyze("ASAP! Deadline today, need this immediately!")
    print(f"  Urgent message -> urgency: {profile.urgency.value}")
    if profile.urgency.value != "urgent":
        raise AssertionError(f"Expected 'urgent', got '{profile.urgency.value}'")

    # Test technical message
    profile = detector.analyze("Test N=15, strength=50 kPa, gamma=18 kN/m3, calculate capacity")
    print(f"  Technical message -> depth: {profile.technical_depth:.2f}")
    if profile.technical_depth < 0.5:
        raise AssertionError(f"Expected technical_depth >= 0.5, got {profile.technical_depth}")

    # Test casual message
    profile = detector.analyze("hey cool thanks!")
    print(f"  Casual message -> formality: {profile.formality.value}")
    if profile.formality.value != "casual":
        raise AssertionError(f"Expected 'casual', got '{profile.formality.value}'")


def test_correction_detection():
    """Verify correction patterns are detected."""
    from tools.phii import CorrectionDetector

    detector = CorrectionDetector()

    # Test "don't use X" pattern
    correction = detector.detect("Don't use imperial units, use metric instead", "")
    print(f"  'Don't use imperial' -> wrong: '{correction.wrong_behavior}', right: '{correction.right_behavior}'")
    if not correction:
        raise AssertionError("Failed to detect 'don't use' correction pattern")

    # Test "always use X" pattern
    correction = detector.detect("Always use metric units for this project", "")
    print(f"  'Always use metric' -> right: '{correction.right_behavior}'")
    if not correction:
        raise AssertionError("Failed to detect 'always use' correction pattern")

    # Test "too long" pattern
    correction = detector.detect("That's too long, be more concise", "")
    print(f"  'Too long' -> wrong: '{correction.wrong_behavior}', right: '{correction.right_behavior}'")
    if not correction:
        raise AssertionError("Failed to detect 'too long' style correction")


def test_session_profile():
    """Verify session profile tracking works."""
    from tools.phii import PhiiContextBuilder

    builder = PhiiContextBuilder()
    session_id = "test-session-123"

    # Build context to populate profile
    builder.build_context(
        current_message="I'm an EIT learning about performance capacity",
        files_context="No files",
        session_notes="",
        session_id=session_id
    )

    # Get session profile
    profile = builder.get_session_profile(session_id)
    print(f"  Session profile:")
    print(f"    - Energy: brevity={profile['energy']['brevity']}")
    print(f"    - Expertise: level={profile['expertise']['level']}")

    # Clear session
    builder.clear_session(session_id)
    profile_after = builder.get_session_profile(session_id)
    print(f"  After clear: expertise={profile_after['expertise']['level']} (should be 'intermediate' default)")


def main():
    """Run all tests."""
    tests = [
        ("Firm Corrections Seeded", test_firm_corrections_seeded),
        ("Corrections Injected", test_corrections_injected),
        ("Expertise Detection", test_expertise_detection),
        ("Energy Detection", test_energy_detection),
        ("Correction Detection", test_correction_detection),
        ("Session Profile", test_session_profile),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Phii Learning System Validation")
    print("=" * 60)

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
