.PHONY: install
install: ## Install the virtual environment and install the prek hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run prek install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running prek"
	@uv run prek run --all-files
	@echo "🚀 Static type checking: Running ty"
	@uv run ty check

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs
docs: ## Build the Sphinx/MyST-NB docs (executes + caches notebooks)
	@uv run --group docs sphinx-build -b html docs docs/_build/html

.PHONY: docs-ci
docs-ci: ## Build docs in smoke mode, warnings-as-errors (CI gate)
	@IMPULSO_DOCS_CI=1 uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

.PHONY: docs-test
docs-test: docs-ci ## Alias for docs-ci: fail on any warning

.PHONY: docs-serve
docs-serve: docs ## Serve the built docs locally on :8000
	@uv run --group docs python -m http.server 8000 --directory docs/_build/html

.PHONY: docs-clean
docs-clean: ## Remove built docs (keeps the notebook exec cache)
	@rm -rf docs/_build/html docs/_build/doctrees

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
