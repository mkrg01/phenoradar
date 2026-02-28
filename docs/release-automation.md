# Release Automation (Developer Runbook)

This page is for current and future maintainers of this repository.
It explains what runs automatically on PRs, on `main` merges, and during
package release to PyPI.

## Workflows

- `.github/workflows/ci.yml`
  - Trigger: `pull_request`, `push` to `main`
  - Purpose: lint, type-check, test
- `.github/workflows/release-please.yml`
  - Trigger: `push` to `main`
  - Purpose: create/update release PR, then create tag/release when that PR is merged
- `.github/workflows/release.yml`
  - Trigger: tag push `v*`
  - Purpose: build package, publish to PyPI, upload release assets

## What runs when

1. Open or update a pull request:
   - `ci.yml` runs on `pull_request`.
2. Merge a regular PR into `main`:
   - `ci.yml` runs again on `push(main)`.
   - `release-please.yml` runs and updates or opens a release PR.
3. Merge the release PR into `main`:
   - `release-please.yml` runs again.
   - It creates a Git tag `vX.Y.Z` and GitHub release.
4. Tag `vX.Y.Z` is pushed:
   - `release.yml` runs:
     - `build`: validate tag/version, build and check distributions.
     - `publish-pypi`: publish artifacts to PyPI via OIDC.
     - `github-release`: attach artifacts to the GitHub release.

Important: merging a feature/fix PR into `main` does not publish to PyPI immediately.
Publish happens only after the release PR is merged and the version tag is created.

## One-time setup

1. Configure PyPI Trusted Publisher:
   - owner/repo: this repository
   - workflow: `.github/workflows/release.yml`
   - environment: blank (no environment gate)
2. Add repository secret `RELEASE_PLEASE_TOKEN` (recommended):
   - token must be able to create PRs/releases/tags
   - this helps ensure release-please-created tags reliably trigger downstream workflows

If PyPI Trusted Publisher was created with environment `pypi`, update it (or recreate it)
to blank environment.

## Commit and versioning notes

- Use conventional commit prefixes (`feat:`, `fix:`, etc.) on merged PRs.
- `release-please` determines version/changelog updates from commit history.
- In normal operation, do not hand-edit version files for each release.
- This repository uses release-please in non-manifest mode.
- Primary version source is `pyproject.toml` (`project.version`).
- Tags remain `vX.Y.Z` (no component prefix).

## Troubleshooting checklist

1. Release PR is not created or updated:
   - check `release-please.yml` run on latest `push(main)`
   - verify `RELEASE_PLEASE_TOKEN` exists and has required scopes
   - verify merged commits use conventional commit style
   - if migrating from another versioning setup, ensure expected baseline tags (`vX.Y.Z`) exist
2. Release PR merged but `release.yml` did not run:
   - confirm tag `vX.Y.Z` was actually created
   - confirm tag push event exists in Actions history
3. `release.yml` fails with tag/version mismatch:
   - `pyproject.toml` version must match `GITHUB_REF_NAME` without leading `v`
   - recover by using the next release PR/version instead of forcing manual tags
4. PyPI publish fails (OIDC/trusted publisher):
   - confirm trusted publisher points to this repo + `release.yml`
   - confirm trusted publisher environment is blank
