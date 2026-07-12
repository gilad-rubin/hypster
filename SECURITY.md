# Security Policy

Please report vulnerabilities via GitHub Security Advisories or email me@giladrubin.com. We aim to respond within 7 days.

## Trust model

Hypster config functions are plain Python that you write and import yourself; `instantiate()`, `explore()`, and `interact()` execute that code directly, so only run config functions you trust. Hypster accepts JSON-compatible data (values dicts, rule/schema dicts) and validates it, but never runs code-executing deserializers such as `pickle`, and never uses `exec`/`eval`, on untrusted input.
