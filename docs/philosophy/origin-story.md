# Origin Story

Hypster grew from a recurring problem in AI and ML projects: configuration is often more dynamic than a settings file, but experiment logs still need stable, replayable parameters.

In practice, a workflow might need to switch between model families, providers, retrieval modes, preprocessing steps, local and remote execution, or production and evaluation paths. Each branch has different parameters. Logging every possible value is noisy, but silently ignoring inactive values makes experiments hard to trust.

Hypster's answer is a small define-by-run API:

* write the config as plain Python
* use `hp.*` calls for the values that matter
* compose nested configs with `hp.nest`
* inspect active branches with `explore`
* run with `instantiate`
* log and replay with `instantiate_with_params`

The goal is not to hide Python behind a configuration language. The goal is to make the parts of Python configuration that matter for reproducibility explicit.
