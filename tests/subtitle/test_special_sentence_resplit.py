#!/usr/bin/env python3
"""
Test the special sentence type re-splitting functionality
"""

import sys

from lattifai.alignment.tokenizer import LatticeTokenizer


def test_resplit_special_sentence_types():
    """Test the _resplit_special_sentence_types method"""

    test_cases = [
        # (input, expected_output)
        ("[APPLAUSE] &gt;&gt; MIRA MURATI:", ["[APPLAUSE]", "&gt;&gt; MIRA MURATI:"]),
        ("[MUSIC] &gt;&gt; SPEAKER:", ["[MUSIC]", "&gt;&gt; SPEAKER:"]),
        ("[APPLAUSE] MIRA MURATI:", ["[APPLAUSE]", "MIRA MURATI:"]),
        ("[SOUND EFFECT] &gt;&gt; HOST: Hello there", ["[SOUND EFFECT]", "&gt;&gt; HOST: Hello there"]),
        ("This is a normal sentence.", ["This is a normal sentence."]),
        ("[EVENT] Speaker: This is the content", ["[EVENT]", "Speaker: This is the content"]),
        ("[APPLAUSE] >> SPEAKER:", ["[APPLAUSE]", ">> SPEAKER:"]),
    ]

    print("=" * 60)
    print("Test Special Sentence Type Re-splitting")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, (input_text, expected_output) in enumerate(test_cases, 1):
        result = LatticeTokenizer._resplit_special_sentence_types(input_text)
        is_passed = result == expected_output

        status = "✓ PASS" if is_passed else "✗ FAIL"
        print(f"\nTest Case {i}: {status}")
        print(f"  Input:    {input_text}")
        print(f"  Expected: {expected_output}")
        print(f"  Got:      {result}")

        if is_passed:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_resplit_special_sentence_types()
    sys.exit(0 if success else 1)
