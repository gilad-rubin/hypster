# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- CI workflow (ruff, mypy, pytest with coverage)
- Docs deploy workflow for GitHub Pages
- Release workflow (PyPI via Trusted Publisher)
- PEP 561 typing support (`py.typed`, `hp.pyi`) to improve IDE hints
- Coverage configuration and README badges

### Changed
- Packaging configuration for Hatch (wheel/sdist includes for typing files)
