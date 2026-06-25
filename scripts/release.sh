#!/usr/bin/env bash
#
# Cut a release: bump the version in pyproject.toml AND uv.lock together,
# commit, and tag. Pushing the tag is left as a deliberate manual step
# because it triggers the PyPI publish (which cannot be undone).
#
# Usage:
#   ./scripts/release.sh [patch|minor|major]   # default: patch
#
# The key point: `uv version --bump` rewrites pyproject.toml and uv.lock in
# lockstep, so the lockfile can never go stale and `uv sync --locked` (CI)
# can never fail on a version mismatch again.

set -euo pipefail

part="${1:-patch}"

case "$part" in
  patch|minor|major) ;;
  *) echo "error: part must be patch, minor or major (got '$part')" >&2; exit 1 ;;
esac

# Refuse to run on a dirty tree so the bump lands as an isolated commit.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty -- commit or stash your changes first." >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "master" ]]; then
  echo "warning: you are on '$branch', not master. Releases usually come from master." >&2
fi

old="$(uv version --short)"
uv version --bump "$part"           # updates pyproject.toml + uv.lock atomically
new="$(uv version --short)"

git commit -am "verbump to v${new}"
git tag "v${new}"

echo
echo "Bumped ${old} -> ${new}, committed, and tagged v${new}."
echo "When you're ready to publish to PyPI, push the commit and the tag:"
echo
echo "    git push && git push origin v${new}"
echo
