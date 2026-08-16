import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


class TestEnvironmentError(RuntimeError):
    pass


def normalize_python_version(version):
    value = str(version or "").strip()
    if not re.match(r"^\d+\.\d+\.\d+$", value):
        raise TestEnvironmentError(
            "python_version must be an exact major.minor.patch version: %r" % value
        )
    return value


def query_python_version(python_executable):
    command = [
        str(python_executable),
        "-c",
        (
            "import json,sys;"
            "print(json.dumps({'version':'.'.join(map(str,sys.version_info[:3])),"
            "'executable':sys.executable}))"
        ),
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return json.loads(output.strip().splitlines()[-1])
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise TestEnvironmentError(
            "cannot execute Python interpreter %s: %s" % (python_executable, error)
        )


def assert_exact_python(python_executable, required_version):
    required_version = normalize_python_version(required_version)
    info = query_python_version(python_executable)
    if info["version"] != required_version:
        raise TestEnvironmentError(
            "Python version mismatch: required=%s actual=%s executable=%s"
            % (required_version, info["version"], info["executable"])
        )
    return info


def decode_requirements(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le"):
        try:
            return raw.decode(encoding)
        except UnicodeError:
            continue
    raise TestEnvironmentError("cannot decode requirements file: %s" % path)


def prepare_requirements(source_path, destination_path):
    """Convert BugsInPy requirements to UTF-8 and remove target-repo installs."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if not source_path.exists():
        destination_path.write_text("", encoding="utf-8")
        return {
            "source": None,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "removed_local_project_lines": [],
        }

    text = decode_requirements(source_path)
    kept = []
    removed = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-e ") and ("git+" in stripped or "file:" in stripped):
            removed.append(stripped)
            continue
        kept.append(line.rstrip())

    normalized = "\n".join(kept).strip() + "\n" if kept else ""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(normalized, encoding="utf-8")
    return {
        "source": str(source_path),
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "removed_local_project_lines": removed,
    }


def _venv_python(env_dir):
    env_dir = Path(env_dir)
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _local_python_candidates(required_version, explicit_python=None):
    major_minor = ".".join(required_version.split(".")[:2])
    candidates = []
    if explicit_python:
        candidates.append(str(explicit_python))
    candidates.extend(
        [
            "python%s" % required_version,
            "python%s" % major_minor,
        ]
    )
    seen = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        try:
            assert_exact_python(resolved, required_version)
            yield Path(resolved).resolve()
        except TestEnvironmentError:
            continue


def _uv_command(algorithm_python):
    executable = shutil.which("uv")
    if executable:
        return [executable]
    module_command = [str(algorithm_python), "-m", "uv"]
    result = subprocess.run(
        module_command + ["--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return module_command if result.returncode == 0 else None


def _pyenv_command():
    executable = shutil.which("pyenv")
    return [executable] if executable else None


def _run_checked(command, cwd=None, env=None, timeout=None):
    try:
        subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            check=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise TestEnvironmentError(
            "command failed: %s\n%s" % (" ".join(map(str, command)), error)
        )


class StrictTestEnvironment:
    def __init__(
        self,
        required_version,
        env_dir,
        algorithm_python,
        provider="auto",
        explicit_python=None,
        index_url=DEFAULT_INDEX_URL,
    ):
        self.required_version = normalize_python_version(required_version)
        self.env_dir = Path(env_dir).resolve()
        self.algorithm_python = Path(algorithm_python).resolve()
        self.provider = provider
        self.explicit_python = explicit_python
        self.index_url = index_url
        self.python = _venv_python(self.env_dir)
        self.provider_used = None
        self.requirements_metadata = {}
        self.uv_command = None

    def create(self):
        if self.env_dir.exists():
            shutil.rmtree(self.env_dir)
        self.env_dir.parent.mkdir(parents=True, exist_ok=True)

        local_python = next(
            _local_python_candidates(self.required_version, self.explicit_python),
            None,
        )
        if self.provider in ("auto", "local") and local_python is not None:
            _run_checked([str(local_python), "-m", "venv", str(self.env_dir)])
            self.provider_used = "local"
        elif self.provider == "local":
            raise TestEnvironmentError(
                "exact local Python %s was not found" % self.required_version
            )
        else:
            provider_errors = []
            if self.provider in ("auto", "uv"):
                uv = _uv_command(self.algorithm_python)
                if uv is None:
                    provider_errors.append("uv is not installed")
                else:
                    try:
                        _run_checked(
                            uv + ["python", "install", self.required_version],
                            timeout=900,
                        )
                        _run_checked(
                            uv
                            + [
                                "venv",
                                "--python",
                                self.required_version,
                                str(self.env_dir),
                            ],
                            timeout=300,
                        )
                        self.provider_used = "uv"
                        self.uv_command = uv
                    except TestEnvironmentError as error:
                        provider_errors.append("uv: %s" % error)
                        if self.env_dir.exists():
                            shutil.rmtree(self.env_dir)
                if self.provider == "uv" and self.provider_used is None:
                    raise TestEnvironmentError("; ".join(provider_errors))

            if self.provider_used is None and self.provider in ("auto", "pyenv"):
                pyenv = _pyenv_command()
                if pyenv is None:
                    provider_errors.append("pyenv is not installed")
                else:
                    try:
                        _run_checked(
                            pyenv + ["install", "-s", self.required_version],
                            timeout=3600,
                        )
                        prefix = subprocess.check_output(
                            pyenv + ["prefix", self.required_version],
                            text=True,
                        ).strip()
                        pyenv_python = Path(prefix) / "bin" / "python"
                        assert_exact_python(pyenv_python, self.required_version)
                        _run_checked(
                            [str(pyenv_python), "-m", "venv", str(self.env_dir)],
                            timeout=300,
                        )
                        self.provider_used = "pyenv"
                    except (
                        OSError,
                        subprocess.CalledProcessError,
                        TestEnvironmentError,
                    ) as error:
                        provider_errors.append("pyenv: %s" % error)
                        if self.env_dir.exists():
                            shutil.rmtree(self.env_dir)
                if self.provider == "pyenv" and self.provider_used is None:
                    raise TestEnvironmentError("; ".join(provider_errors))

            if self.provider_used is None:
                raise TestEnvironmentError(
                    "cannot provision exact Python %s: %s"
                    % (self.required_version, "; ".join(provider_errors))
                )

        assert_exact_python(self.python, self.required_version)
        pip_check = subprocess.run(
            [str(self.python), "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if pip_check.returncode != 0:
            _run_checked(
                [str(self.python), "-m", "ensurepip", "--upgrade"],
                timeout=300,
            )
        return self

    def install_requirements(self, requirements_path, metadata):
        self.requirements_metadata = dict(metadata)
        requirements_path = Path(requirements_path)
        if requirements_path.exists() and requirements_path.stat().st_size:
            if self.provider_used == "uv":
                command = self.uv_command + [
                    "pip",
                    "install",
                    "--python",
                    str(self.python),
                    "--index-url",
                    self.index_url,
                    "--prerelease",
                    "allow",
                    "-r",
                    str(requirements_path),
                ]
            else:
                command = [
                    str(self.python),
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    self.index_url,
                    "--trusted-host",
                    re.sub(r"^https?://", "", self.index_url).split("/", 1)[0],
                    "--timeout",
                    "60",
                    "--retries",
                    "3",
                    "--pre",
                    "-r",
                    str(requirements_path),
                ]
            _run_checked(command, timeout=1800)
        assert_exact_python(self.python, self.required_version)

    def runtime_env(self, worktree, extra_pythonpath=None):
        worktree = Path(worktree).resolve()
        python_paths = [
            str(worktree),
            str(worktree / "src"),
            str(worktree / "lib"),
        ]
        if extra_pythonpath:
            for item in str(extra_pythonpath).split(os.pathsep):
                if item:
                    path = Path(item)
                    python_paths.append(
                        str(path if path.is_absolute() else worktree / path)
                    )
        env = dict(os.environ)
        env.update(
            {
                "VIRTUAL_ENV": str(self.env_dir),
                "PATH": str(self.python.parent) + os.pathsep + env.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join(python_paths),
                "__PYVENV_LAUNCHER__": str(self.python),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PIP_INDEX_URL": self.index_url,
            }
        )
        return env

    def run_setup(self, setup_path, worktree, extra_pythonpath=None):
        setup_path = Path(setup_path)
        if not setup_path.exists() or not setup_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip():
            return
        env = self.runtime_env(worktree, extra_pythonpath)
        _run_checked(
            ["/bin/bash", str(setup_path.resolve())],
            cwd=str(Path(worktree).resolve()),
            env=env,
            timeout=1800,
        )
        assert_exact_python(self.python, self.required_version)

    def install_target(self, worktree):
        worktree = Path(worktree).resolve()
        if (worktree / "setup.py").exists():
            command = [
                str(self.python),
                "setup.py",
                "develop",
            ]
        elif self.provider_used == "uv":
            command = self.uv_command + [
                "pip",
                "install",
                "--python",
                str(self.python),
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                str(worktree),
            ]
        else:
            command = [
                str(self.python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                str(worktree),
            ]
        _run_checked(command, cwd=str(worktree), timeout=1800)
        assert_exact_python(self.python, self.required_version)

    def manifest(self):
        info = assert_exact_python(self.python, self.required_version)
        return {
            "required_python": self.required_version,
            "actual_python": info["version"],
            "python_match": info["version"] == self.required_version,
            "python_executable": info["executable"],
            "provider": self.provider_used,
            "environment": str(self.env_dir),
            "requirements": self.requirements_metadata,
        }
