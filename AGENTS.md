# Environment
* use `uv` as python env manager, please use `uv run` to run program and `uv add` to add more dependencies.

# Shell Usage

* Never use things like `gh pr list --state all --limit 1` which will lead to shell interaction (like typing q to exit). Using this kind of commnad, you will never get any valid feedback from the shell and the whole pipeline will be paused.

# Report

After your work, always report and communicate with user in Chinese!

# Memory
在执行任务前，总是记得查询一下相关记忆，可能会大幅减少工作量。

对于调研结果，代码修改，工作进展，总是记得要使用OpenMemory 的 MCP Tools 添加记忆，user_id 使用 null，同时，记得在记忆中明确提及当前开发的是什么项目，以及当前时间，以方便获取记忆时进行定位。