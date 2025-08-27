"""Run all test modules to verify they work."""

import subprocess
import sys


def run_tests():
    """Run pytest on all our test modules."""

    test_modules = [
        "tests/test_basic_parameters.py",
        "tests/test_select_parameters.py",
        "tests/test_multi_parameters.py",
        "tests/test_explicit_naming.py",
        "tests/test_return_types.py",
        "tests/test_nesting.py",
        "tests/test_validation.py",
        "tests/test_function_signature.py",
        "tests/test_edge_cases.py",
        "tests/test_error_handling.py",
    ]

    print("Running all test modules...")
    print("=" * 60)

    for module in test_modules:
        print(f"\n🧪 Testing {module}")
        print("-" * 40)

        # Run pytest on each module individually
        result = subprocess.run([sys.executable, "-m", "pytest", module, "-v"], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {module} - PASSED")
        else:
            print(f"❌ {module} - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

    print("\n" + "=" * 60)
    print("Test run complete!")


if __name__ == "__main__":
    run_tests()
