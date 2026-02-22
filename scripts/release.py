# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Bump version, commit, tag, push, and create a GitHub release."""

import subprocess
import sys


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main() -> None:
    bump = sys.argv[1] if len(sys.argv) > 1 else "patch"
    if bump not in ("major", "minor", "patch"):
        print(f"Invalid bump type: {bump}. Use major, minor, or patch.", file=sys.stderr)
        sys.exit(1)

    # Check for clean working tree (aside from what we're about to change)
    status = run("git status --porcelain -uno")
    if status:
        print(f"Working tree is not clean:\n{status}", file=sys.stderr)
        sys.exit(1)

    run(f"uv version --bump {bump}")
    version = run("uv version").split()[-1]
    tag = f"v{version}"

    print(f"Releasing {tag}")

    run("git add pyproject.toml uv.lock")
    run(f'git commit -m "Bump version to {version}"')
    run(f"git tag {tag}")
    run("git push && git push --tags")
    run(f"gh release create {tag} --generate-notes")

    print(f"Released {tag}")


if __name__ == "__main__":
    main()
