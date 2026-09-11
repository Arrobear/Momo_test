"""Read immutable revisions without changing the source repository checkout."""

import subprocess


def git_output(repo_dir, *arguments):
    return subprocess.check_output(
        ["git", *arguments], cwd=repo_dir, text=True, encoding="utf-8", errors="strict"
    ).strip()


def resolve_commit(repo_dir, revision):
    return git_output(repo_dir, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
