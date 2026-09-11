# Releasing Leo

Leo software releases are automated. The public software version can advance without changing the model schema, training-policy version, execution-semantics version, dataset schema, CUDA ABI, checkpoint schema, or `PSCLS100` format.

## One-time GitHub setup

The `main` branch is protected and releases must go through the same pull-request/status-check path as normal changes. Create a fine-grained GitHub personal access token scoped only to `saravanaspar/Leo` with:

- **Contents:** Read and write
- **Pull requests:** Read and write
- **Actions:** Read

Save it under **Repository Settings -> Secrets and variables -> Actions -> New repository secret** as:

```text
RELEASE_TOKEN
```

The token is needed because pull requests created with the workflow's built-in `GITHUB_TOKEN` do not trigger normal `pull_request` workflows. Using `RELEASE_TOKEN` keeps the protected `cpu` check in the release path.

## Create a release

1. Merge all changes intended for the release into `main`.
2. Make sure `CHANGELOG.md` has the release notes under `## Unreleased`.
3. Open **GitHub -> Actions -> release -> Run workflow**.
4. Enter a stable SemVer value such as `1.0.2` (a leading `v` is also accepted).
5. Run the workflow.

The workflow then:

1. validates and normalizes the version;
2. updates the workspace software version in `Cargo.toml`;
3. refreshes the Leo package versions in `Cargo.lock` through Cargo;
4. updates `CITATION.cff` and its release date;
5. updates the software-release row in `docs/SEMANTICS.md` without changing semantic/schema versions;
6. moves the current `CHANGELOG.md` `Unreleased` entries into `## vX.Y.Z - YYYY-MM-DD` and leaves a fresh `## Unreleased` section;
7. runs the repository build and CPU/static quality gate;
8. creates `release/vX.Y.Z` and opens a release PR;
9. waits for the protected `cpu` status check;
10. squash-merges the release PR into `main`;
11. creates the immutable `vX.Y.Z` tag; and
12. creates the GitHub Release using the matching changelog section as release notes.

Release tags are never moved or overwritten. If a tag already exists at a different commit, the workflow fails rather than mutating it.

## README versioning

The README intentionally uses the stable title:

```text
# Leo
```

It does not embed a release number in the title or prose. The release badge resolves the latest GitHub Release dynamically, so publishing a release updates what visitors see without another README edit.

## Local release-script checks

You can inspect or test the release metadata helper without publishing anything:

```bash
python3 scripts/release.py current
python3 scripts/release.py normalize v1.0.2
```

Do not run `prepare` on a working branch unless you intentionally want to create release metadata changes. The GitHub release workflow is the normal release path.
