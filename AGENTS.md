# Environment
* use `uv` as python env manager, please use `uv run` to run program and `uv add` to add more dependencies.

# Shell Usage

* Never use things like `gh pr list --state all --limit 1` which will lead to shell interaction (like typing q to exit). Using this kind of commnad, you will never get any valid feedback from the shell and the whole pipeline will be paused.