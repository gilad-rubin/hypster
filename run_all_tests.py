#!/usr/bin/env python3
"""Run all manual tests."""

import os
import subprocess
import sys

# Add src to path to import hypster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_test_file(test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print("=" * 50)

    result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return False


def main():
    """Run all test files."""
    test_files = [
        "test_implementation.py",
        "run_basic_tests.py",
        "run_select_tests.py",
        "run_multi_tests.py",
        "run_nesting_tests.py",
    ]

    passed = 0
    total = len(test_files)

    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1
            print(f"✅ {test_file} PASSED")
        else:
            print(f"❌ {test_file} FAILED")

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS: {passed}/{total} test files passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Hypster v2 implementation is working correctly!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
