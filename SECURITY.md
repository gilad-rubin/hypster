# Security Policy

Please report vulnerabilities via GitHub Security Advisories or email me@giladrubin.com. We aim to respond within 7 days.

## Trust model

Hypster config functions are plain Python that you write and import yourself; `instantiate()`, `explore()`, and `interact()` execute that code directly, so only run config functions you trust. Hypster itself performs no deserialization, `exec`/`eval`, or pickling of untrusted data.
