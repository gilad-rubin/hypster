#!/usr/bin/env python3
"""
Comprehensive package verification script for Hypster.

This script verifies that the built package contains all required metadata
and files for successful PyPI publishing.
"""

import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List


def find_dist_files() -> Dict[str, Path]:
    """Find wheel and source distribution files."""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        raise FileNotFoundError("dist/ directory not found. Run 'uv build' first.")

    wheel_file = None
    sdist_file = None

    for file in dist_dir.iterdir():
        if file.suffix == ".whl":
            wheel_file = file
        elif file.suffix == ".gz" and ".tar" in file.name:
            sdist_file = file

    if not wheel_file:
        raise FileNotFoundError("No wheel file found in dist/")
    if not sdist_file:
        raise FileNotFoundError("No source distribution found in dist/")

    return {"wheel": wheel_file, "sdist": sdist_file}


def extract_wheel_metadata(wheel_path: Path) -> Dict[str, Any]:
    """Extract metadata from wheel file."""
    metadata: Dict[str, Any] = {}

    with zipfile.ZipFile(wheel_path, "r") as wheel:
        # Find METADATA file
        metadata_files = [f for f in wheel.namelist() if f.endswith("METADATA")]
        if not metadata_files:
            raise ValueError(f"No METADATA file found in {wheel_path}")

        # Read metadata
        metadata_content = wheel.read(metadata_files[0]).decode("utf-8")

        # Parse metadata fields - handle multiple values for same key
        for line in metadata_content.split("\n"):
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Handle multiple classifiers
                if key in metadata:
                    if isinstance(metadata[key], list):
                        metadata[key].append(value)
                    else:
                        metadata[key] = [metadata[key], value]
                else:
                    metadata[key] = value

        # List all files in the wheel
        metadata["files"] = wheel.namelist()

    return metadata


def verify_required_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Verify that all required metadata fields are present."""
    required_fields = ["Name", "Version", "Summary", "Author-email", "License", "Classifier", "Requires-Python"]

    missing_fields = []
    for field in required_fields:
        if field not in metadata:
            missing_fields.append(field)

    return missing_fields


def verify_package_structure(metadata: Dict[str, Any]) -> List[str]:
    """Verify that the package has the expected structure."""
    issues = []
    files = metadata.get("files", [])

    # Check for Python package files
    py_files = [f for f in files if f.endswith(".py")]
    if not py_files:
        issues.append("No Python files found in package")

    # Check for hypster package
    hypster_files = [f for f in files if "hypster" in f and f.endswith(".py")]
    if not hypster_files:
        issues.append("No hypster package files found")

    # Check for __init__.py
    init_files = [f for f in files if f.endswith("__init__.py")]
    if not init_files:
        issues.append("No __init__.py files found")

    return issues


def print_metadata_summary(metadata: Dict[str, Any]) -> None:
    """Print a summary of the package metadata."""
    print("\n📋 Package Metadata Summary:")
    print("=" * 40)

    key_fields = ["Name", "Version", "Summary", "Author-email", "License", "Requires-Python"]
    for field in key_fields:
        value = metadata.get(field, "NOT FOUND")
        print(f"{field:15}: {value}")

    # Show classifiers
    classifier_value = metadata.get("Classifier", [])
    if isinstance(classifier_value, str):
        classifiers = [classifier_value]
    elif isinstance(classifier_value, list):
        classifiers = classifier_value
    else:
        classifiers = []

    if classifiers:
        print(f"\nClassifiers ({len(classifiers)}):")
        for classifier in classifiers[:5]:  # Show first 5
            print(f"  - {classifier}")
        if len(classifiers) > 5:
            print(f"  ... and {len(classifiers) - 5} more")


def main() -> int:
    """Main verification function."""
    print("🔍 Hypster Package Verification")
    print("=" * 50)

    try:
        # Find distribution files
        print("\n1. Finding distribution files...")
        dist_files = find_dist_files()
        print(f"   ✅ Found wheel: {dist_files['wheel'].name}")
        print(f"   ✅ Found sdist: {dist_files['sdist'].name}")

        # Extract and verify wheel metadata
        print("\n2. Extracting wheel metadata...")
        metadata = extract_wheel_metadata(dist_files["wheel"])
        print(f"   ✅ Extracted metadata from {dist_files['wheel'].name}")

        # Verify required metadata
        print("\n3. Verifying required metadata fields...")
        missing_fields = verify_required_metadata(metadata)
        if missing_fields:
            print("   ❌ Missing required metadata fields:")
            for field in missing_fields:
                print(f"      - {field}")
            return 1
        else:
            print("   ✅ All required metadata fields present")

        # Verify package structure
        print("\n4. Verifying package structure...")
        structure_issues = verify_package_structure(metadata)
        if structure_issues:
            print("   ❌ Package structure issues:")
            for issue in structure_issues:
                print(f"      - {issue}")
            return 1
        else:
            print("   ✅ Package structure looks good")

        # Print summary
        print_metadata_summary(metadata)

        print("\n🎉 Package verification completed successfully!")
        print("\n📦 The package should upload to PyPI without issues.")

        return 0

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
