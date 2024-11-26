# .PHONY targets are special targets that don't represent actual files
# Without .PHONY, if a file named 'format' existed, make would think the target is up to date
# and wouldn't run the commands. .PHONY ensures the commands always run when called.
.PHONY: format

format:
	uv run ruff check src --fix

