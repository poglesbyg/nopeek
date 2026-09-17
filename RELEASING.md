# Releasing

A PyPI version is permanent. It can be yanked, but the number can never be
reused, so the checks below run before anything is uploaded rather than after.

## One-time setup: Trusted Publishing

No API token needs to exist anywhere. PyPI verifies the release workflow's
identity directly.

The project exists on PyPI, so this is an ordinary trusted publisher rather
than the *pending* kind (pending publishers are only for projects that have
never been published).

1. Go to **https://pypi.org/manage/project/nopeek/settings/publishing/** — or
   Your projects → Manage → **Publishing** in the sidebar.
2. Under **GitHub**, fill in:
   - Owner: `poglesbyg`
   - Repository name: `nopeek`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The environment is optional as far as PyPI is concerned, but `release.yml`
   declares `environment: pypi`, so it has to match or the OIDC claim will be
   rejected.
3. In the GitHub repo, create an environment named `pypi`
   (**Settings → Environments → New environment**). Adding yourself as a
   required reviewer means every upload needs an explicit approval.

## Rehearsing it against TestPyPI

An untested publish workflow gets tested by a real release, which is a poor
place to find out the OIDC claim is misconfigured. `release.yml` therefore has a
second way in: **Actions → release → Run workflow** publishes to TestPyPI
instead, as a throwaway `.devN` version numbered from the run, so a rehearsal
never spends a real version and can be repeated as often as you like.

A dispatch can only reach TestPyPI and a release can only reach PyPI. They are
separate jobs with separate conditions and separate environments, so neither can
be mistaken for the other.

Setting it up mirrors the PyPI side, on a separate account:

1. Register at https://test.pypi.org (its accounts and tokens are entirely
   separate from PyPI's) and enable 2FA.
2. Because nothing has been published there yet, this one *is* a **pending**
   publisher: **Your projects → Publishing → Add a pending publisher**, with
   PyPI project name `nopeek`, owner `poglesbyg`, repository `nopeek`, workflow
   `release.yml`, environment `testpypi`.
3. Create a GitHub environment named `testpypi`. Leave this one without a
   required reviewer — the point of a rehearsal is that it is cheap to run.

A green dispatch exercises the same OIDC handshake, the same environment gate
and the same publishing action as the real thing. Only the index differs.

## Cutting a release

1. Bump `version` in `pyproject.toml`. Nothing else: `__version__` is read from
   installed metadata, so there is no second copy to forget.
2. Commit, tag and push:
   ```bash
   git commit -am "Release 0.1.0" && git tag v0.1.0 && git push --follow-tags
   ```
3. Publish a GitHub Release for the tag. That fires `release.yml`, which runs
   the suite, checks the tag against `pyproject.toml`, builds, runs
   `twine check`, and uploads.

Worth a dispatch against TestPyPI first if anything about the workflow has
changed since the last release.

## Publishing by hand instead

If Trusted Publishing is not set up yet:

Create the token first, at PyPI → Account settings → **API tokens**. It is a
long string beginning `pypi-AgEIcHlwaS5vcmc...`; `pypi-...` below is a
placeholder, and pasting it literally gets you
`403 Invalid or non-existent authentication information`. Scope it to this
project once the project exists — the first upload needs an account-wide token.

```bash
uv build
uv run --with "twine>=6.1" twine check dist/*
export UV_PUBLISH_TOKEN="pypi-AgEI..."   # the real token, not this
uv publish
```

Putting the token in the environment rather than on the command line keeps it
out of your shell history.

## Before the first upload

Worth doing once, by hand, because the first version of a package is the one
nobody can fix later:

```bash
uv build
tar tzf dist/*.tar.gz            # sdist should contain tests and conftest.py
python -m zipfile -l dist/*.whl  # wheel should contain nopeek/ and py.typed, nothing else
```

Then install the built wheel into an empty environment and check it from a
consumer's point of view — that the import works, that the pytest fixture is
discovered through the entry point, and that a planted leak is still caught.
An editable install in the development checkout will pass even when packaging
is broken, so it proves nothing on its own.

Consider uploading to TestPyPI first. It needs its own account and its own
token — TestPyPI credentials are entirely separate from PyPI's:

```bash
UV_PUBLISH_TOKEN="pypi-AgENdGVzdC..." \
  uv publish --publish-url https://test.pypi.org/legacy/
```
