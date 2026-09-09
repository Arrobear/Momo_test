import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_PYTHON_SEED_DIR = (
    Path(__file__).resolve().parent.parent / ".momo_runtime" / "python-seeds"
)


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
        for line in reversed(output.strip().splitlines()):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                data = json.loads(stripped)
            except ValueError:
                continue
            if "version" in data and "executable" in data:
                return data
        raise ValueError("no JSON python version payload found in output: %r" % output)
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


def _normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", str(name or "").strip()).lower()


def _requirement_package_name(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:\[|===|==|~=|!=|<=|>=|<|>|$)", stripped)
    return _normalize_package_name(match.group(1)) if match else None


NON_INSTALLABLE_REQUIREMENT_NAMES = {"pkg-resources"}
WINDOWS_ONLY_REQUIREMENT_NAMES = {
    "pydivert",
    "pypiwin32",
    "pywin32",
    "pywin32-ctypes",
}
UNAVAILABLE_EXACT_REQUIREMENTS = {
    ("pythonlabs", "1.0.2"),
    ("requests-async", "0.5.0"),
}
REWRITTEN_REQUIREMENTS = {
    ("mysql-connector-python", "8.0.19"): "mysql-connector-python==8.0.21",
    ("mysql-connector-python", "8.0.20"): "mysql-connector-python==8.0.21",
    ("numpy", "1.19.0rc2"): "numpy==1.19.0",
    ("scipy", "1.5.0rc1"): "scipy==1.5.0",
    ("ruamel-yaml-clib", "0.2.0"): "ruamel.yaml.clib==0.2.8",
}

FASTPARQUET_040_BUILD_REQUIREMENTS = [
    "numpy==1.18.4",
    "Cython==0.29.19",
    "six==1.15.0",
    "thrift==0.13.0",
]


def _unsupported_requirement_names():
    names = set(NON_INSTALLABLE_REQUIREMENT_NAMES)
    if os.name != "nt":
        names.update(WINDOWS_ONLY_REQUIREMENT_NAMES)
    return names


def _rewrite_problematic_requirement(line):
    stripped = line.strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.!+-]+)$", stripped)
    if not match:
        return None
    key = (_normalize_package_name(match.group(1)), match.group(2))
    return REWRITTEN_REQUIREMENTS.get(key)


def _is_unavailable_exact_requirement(line):
    stripped = line.strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.!+-]+)$", stripped)
    if not match:
        return False
    key = (_normalize_package_name(match.group(1)), match.group(2))
    return key in UNAVAILABLE_EXACT_REQUIREMENTS


def _has_exact_requirement(requirements_path, package_name, version):
    normalized_name = _normalize_package_name(package_name)
    pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.!+-]+)$"
    )
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.match(stripped)
        if not match:
            continue
        if (
            _normalize_package_name(match.group(1)) == normalized_name
            and match.group(2) == version
        ):
            return True
    return False


def prepare_requirements(
    source_path,
    destination_path,
    target_package_names=(),
    extra_requirements=(),
):
    """Convert BugsInPy requirements to UTF-8 and remove target-repo installs."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    normalized_targets = {
        _normalize_package_name(name) for name in target_package_names if name
    }
    if not source_path.exists():
        added_project = [
            str(requirement).strip()
            for requirement in extra_requirements
            if str(requirement).strip()
        ]
        normalized = (
            "\n".join(added_project).strip() + "\n" if added_project else ""
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(normalized, encoding="utf-8")
        return {
            "source": None,
            "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "removed_local_project_lines": [],
            "removed_unsupported_lines": [],
            "rewritten_unavailable_lines": [],
            "added_project_requirement_lines": added_project,
        }

    text = decode_requirements(source_path)
    kept = []
    removed_local = []
    removed_unsupported = []
    rewritten_unavailable = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-e ") and ("git+" in stripped or "file:" in stripped):
            removed_local.append(stripped)
            continue
        package_name = _requirement_package_name(stripped)
        if package_name in normalized_targets:
            removed_local.append(stripped)
            continue
        if package_name in _unsupported_requirement_names():
            removed_unsupported.append(stripped)
            continue
        if _is_unavailable_exact_requirement(stripped):
            removed_unsupported.append(stripped)
            continue
        rewritten = _rewrite_problematic_requirement(stripped)
        if rewritten:
            rewritten_unavailable.append({"from": stripped, "to": rewritten})
            kept.append(rewritten)
            continue
        kept.append(line.rstrip())

    existing = {
        _normalize_package_name(_requirement_package_name(line) or line)
        for line in kept
        if str(line).strip()
    }
    added_project = []
    for requirement in extra_requirements:
        requirement = str(requirement).strip()
        if not requirement:
            continue
        package_name = _normalize_package_name(
            _requirement_package_name(requirement) or requirement
        )
        if package_name in existing:
            continue
        kept.append(requirement)
        existing.add(package_name)
        added_project.append(requirement)

    normalized = "\n".join(kept).strip() + "\n" if kept else ""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(normalized, encoding="utf-8")
    return {
        "source": str(source_path),
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "removed_local_project_lines": removed_local,
        "removed_unsupported_lines": removed_unsupported,
        "rewritten_unavailable_lines": rewritten_unavailable,
        "added_project_requirement_lines": added_project,
    }


def _venv_python(env_dir):
    env_dir = Path(env_dir)
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _project_seed_python_candidates(required_version):
    seed_root = Path(
        os.environ.get("MOMO_PYTHON_SEED_DIR", str(DEFAULT_PYTHON_SEED_DIR))
    )
    version_digits = "".join(required_version.split("."))
    patterns = [
        "py%s*/bin/python" % version_digits,
        "python-%s*/bin/python" % required_version,
        "%s*/bin/python" % required_version,
    ]
    for pattern in patterns:
        for candidate in sorted(seed_root.glob(pattern)):
            yield str(candidate)


def _local_python_candidates(required_version, explicit_python=None):
    major_minor = ".".join(required_version.split(".")[:2])
    candidates = []
    if explicit_python:
        candidates.append(str(explicit_python))
    candidates.extend(_project_seed_python_candidates(required_version))
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


def _trusted_host_from_index(index_url):
    return re.sub(r"^https?://", "", index_url).split("/", 1)[0]


def _python_base_prefix(python_executable):
    try:
        return Path(
            subprocess.check_output(
                [
                    str(python_executable),
                    "-c",
                    "import sys; print(sys.base_prefix)",
                ],
                text=True,
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        return None


def _upgrade_installer_tools(python_executable, index_url):
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--index-url",
        index_url,
        "--trusted-host",
        _trusted_host_from_index(index_url),
        "--timeout",
        "60",
        "--retries",
        "3",
        "pip<25",
        "setuptools<70",
        "wheel<0.43",
    ]
    _run_checked(command, timeout=900)


def _setup_script_is_target_install_only(setup_path):
    try:
        text = Path(setup_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    commands = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        commands.append(stripped.rstrip(";"))
    if len(commands) != 1:
        return False
    command = commands[0]
    return bool(
        re.match(
            r"^(?:python(?:\d(?:\.\d+)?)?)\s+setup\.py\s+"
            r"(?:develop|install)(?:\s+--user)?$",
            command,
        )
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
        _upgrade_installer_tools(self.python, self.index_url)
        return self

    def install_requirements(self, requirements_path, metadata):
        self.requirements_metadata = dict(metadata)
        requirements_path = Path(requirements_path)
        if requirements_path.exists() and requirements_path.stat().st_size:
            if _has_exact_requirement(requirements_path, "fastparquet", "0.4.0"):
                self.install_requirements_without_dependencies(
                    FASTPARQUET_040_BUILD_REQUIREMENTS
                )
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
                    _trusted_host_from_index(self.index_url),
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

    def install_requirements_without_dependencies(self, requirements):
        requirements = [str(item).strip() for item in requirements if str(item).strip()]
        if not requirements:
            return
        command = [
            str(self.python),
            "-m",
            "pip",
            "install",
            "--index-url",
            self.index_url,
            "--trusted-host",
            _trusted_host_from_index(self.index_url),
            "--timeout",
            "60",
            "--retries",
            "3",
            "--no-deps",
            *requirements,
        ]
        _run_checked(command, timeout=1800)
        assert_exact_python(self.python, self.required_version)

    def runtime_env(self, worktree, extra_pythonpath=None):
        worktree = Path(worktree).resolve()
        home_dir = self.env_dir / "home"
        cache_dir = self.env_dir / "cache"
        config_dir = self.env_dir / "config"
        ansible_home = home_dir / ".ansible"
        ansible_tmp = ansible_home / "tmp"
        ansible_remote_tmp = ansible_home / "remote_tmp"
        for path in (
            home_dir,
            cache_dir,
            config_dir,
            ansible_home,
            ansible_tmp,
            ansible_remote_tmp,
        ):
            path.mkdir(parents=True, exist_ok=True)
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
                "HOME": str(home_dir),
                "XDG_CACHE_HOME": str(cache_dir),
                "XDG_CONFIG_HOME": str(config_dir),
                "ANSIBLE_HOME": str(ansible_home),
                "ANSIBLE_LOCAL_TEMP": str(ansible_tmp),
                "ANSIBLE_REMOTE_TEMP": str(ansible_remote_tmp),
                "ANSIBLE_NOCOLOR": "1",
                "ANSIBLE_RETRY_FILES_ENABLED": "0",
            }
        )
        return env

    def run_setup(self, setup_path, worktree, extra_pythonpath=None):
        setup_path = Path(setup_path)
        if not setup_path.exists():
            return
        setup_bytes = setup_path.read_bytes()
        if not setup_bytes.strip():
            return
        if _setup_script_is_target_install_only(setup_path):
            return
        env = self.runtime_env(worktree, extra_pythonpath)
        executable_setup = setup_path.resolve()
        normalized_setup = None
        if b"\r" in setup_bytes:
            normalized_setup = self.env_dir / ".momo_setup.sh"
            normalized_setup.write_bytes(
                setup_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            )
            executable_setup = normalized_setup
        try:
            _run_checked(
                ["/bin/bash", str(executable_setup)],
                cwd=str(Path(worktree).resolve()),
                env=env,
                timeout=1800,
            )
        finally:
            if normalized_setup is not None:
                normalized_setup.unlink(missing_ok=True)
        assert_exact_python(self.python, self.required_version)

    def install_env(self, worktree):
        env = self.runtime_env(worktree)
        base_prefix = _python_base_prefix(self.python)
        if base_prefix and base_prefix != self.env_dir and base_prefix.exists():
            base_bin = base_prefix / ("Scripts" if os.name == "nt" else "bin")
            base_lib = base_prefix / "lib"
            base_include = base_prefix / "include"
            pkg_config = base_lib / "pkgconfig"
            if base_bin.exists():
                env["PATH"] = (
                    str(base_bin) + os.pathsep + env.get("PATH", "")
                )
            if pkg_config.exists():
                old = env.get("PKG_CONFIG_PATH")
                env["PKG_CONFIG_PATH"] = (
                    str(pkg_config)
                    if not old
                    else str(pkg_config) + os.pathsep + old
                )
            if base_include.exists():
                old = env.get("CPATH")
                env["CPATH"] = (
                    str(base_include)
                    if not old
                    else str(base_include) + os.pathsep + old
                )
            if base_lib.exists():
                old = env.get("LIBRARY_PATH")
                env["LIBRARY_PATH"] = (
                    str(base_lib)
                    if not old
                    else str(base_lib) + os.pathsep + old
                )
                old = env.get("DYLD_FALLBACK_LIBRARY_PATH")
                env["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    str(base_lib)
                    if not old
                    else str(base_lib) + os.pathsep + old
                )
        if self.provider_used == "uv":
            env.pop("__PYVENV_LAUNCHER__", None)
            original_pythonpath = os.environ.get("PYTHONPATH")
            if original_pythonpath:
                env["PYTHONPATH"] = original_pythonpath
            else:
                env.pop("PYTHONPATH", None)
        return env

    def install_target(self, worktree):
        worktree = Path(worktree).resolve()
        if (worktree / "setup.py").exists():
            command = [
                str(self.python),
                "setup.py",
                "develop",
                "--no-deps",
            ]
        else:
            command = [
                str(self.python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                str(worktree),
            ]
        _run_checked(
            command,
            cwd=str(worktree),
            env=self.install_env(worktree),
            timeout=1800,
        )
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
