import subprocess


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8", stderr=subprocess.PIPE).strip()


def make_repository(root):
    repo = root / "demo"
    package = repo / "src" / "demo"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text('def echo(value):\n    """Return a value."""\n    return value\n', encoding="utf-8")
    git(repo, "init")
    git(repo, "config", "user.email", "momo@example.invalid")
    git(repo, "config", "user.name", "Momo Tests")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "reference")
    reference = git(repo, "rev-parse", "HEAD")
    source.write_text('def echo(value):\n    """Return a string."""\n    return str(value)\n', encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "candidate")
    return repo, reference, git(repo, "rev-parse", "HEAD")
