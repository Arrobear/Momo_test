"""
AST-based source code resolver for libraries under dl_lib/.
Reads Python source files directly without importing the target library.
"""
import ast
import os
import re
import textwrap
from pathlib import Path

ROOT = Path(os.environ.get("MOMO_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
DL_LIB_ROOT = ROOT / "documentation" / "dl_lib"
TARGET_REPO = os.environ.get("MOMO_TARGET_REPO")

# Cache: {lib_gitname: {api_name: (file_path, node)}}
_cache = {}


def _find_package_root(lib_gitname):
    """Locate the checked-out repository that contains the target library."""
    if TARGET_REPO:
        target = Path(TARGET_REPO).resolve()
        if target.exists():
            return target

    base = DL_LIB_ROOT / lib_gitname
    if not base.exists():
        return None
    return base


def _build_index(lib_gitname):
    """Build a mapping for functions, classes, methods and common source layouts."""
    repo_root = _find_package_root(lib_gitname)
    if repo_root is None:
        _cache[lib_gitname] = {}
        return

    index = {}
    ignored_parts = {".git", ".tox", ".venv", "venv", "__pycache__"}

    if repo_root.is_dir():
        for py_file in sorted(repo_root.rglob("*.py")):
            if any(part in ignored_parts for part in py_file.parts):
                continue
            _index_file(py_file, repo_root, index)
    elif repo_root.suffix == ".py":
        _index_file(repo_root, repo_root.parent, index)

    _cache[lib_gitname] = index


def _index_file(py_file, repo_root, index):
    """Parse a file and index definitions under repository and import aliases."""
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return

    rel_path = py_file.relative_to(repo_root)
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")

    module_prefixes = [".".join(parts)]
    if len(parts) > 1 and parts[0] in {"src", "lib"}:
        module_prefixes.append(".".join(parts[1:]))
    module_prefixes = [prefix.strip(".") for prefix in module_prefixes]

    def register(node, qualified_name):
        for module_prefix in module_prefixes:
            full_name = ".".join(
                part for part in (module_prefix, qualified_name) if part
            )
            index.setdefault(full_name, (str(py_file), node))
        index.setdefault(node.name, (str(py_file), node))

    def visit_definition(node, parent_name=""):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        qualified_name = ".".join(
            part for part in (parent_name, node.name) if part
        )
        register(node, qualified_name)
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                visit_definition(child, qualified_name)

    for node in ast.iter_child_nodes(tree):
        visit_definition(node)


def _ensure_index(lib_gitname):
    if lib_gitname not in _cache:
        _build_index(lib_gitname)


def resolve_api_source(api_name, lib_gitname):
    """
    Resolve an API name to its source location and AST node.
    Returns dict with: file, start_line, end_line, source_code, node, api_type
    or None if not found.
    """
    _ensure_index(lib_gitname)
    index = _cache.get(lib_gitname, {})

    # Try exact match first
    key = api_name
    node_info = index.get(key)
    if node_info is None:
        # Try matching just the last component (function/class name)
        short_name = api_name.rsplit(".", 1)[-1]
        node_info = index.get(short_name)

    if node_info is None:
        return None

    file_path_str, node = node_info
    file_path = Path(file_path_str)

    # Extract source code
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        source_lines = source.splitlines(keepends=True)
    except Exception:
        return None

    start_line = node.lineno
    end_line = getattr(node, "end_lineno", None) or start_line

    # Get the lines for this definition
    code_lines = source_lines[start_line - 1:end_line]
    code = "".join(code_lines)
    code = textwrap.dedent(code)

    # Remove leading docstring
    code = re.sub(
        r'^[ \t]*[ruRU]*[\'"]{3}[\s\S]*?[\'"]{3}\n?',
        "", code, count=1, flags=re.MULTILINE
    )
    code = code.lstrip("\n")

    api_type = "class" if isinstance(node, ast.ClassDef) else "function"

    return {
        "file": file_path_str,
        "start_line": start_line,
        "end_line": end_line,
        "source_code": code,
        "node": node,
        "api_type": api_type,
    }


def get_api_type(api_name, lib_gitname):
    """Return 'class', 'function', or 'unknown' for an API name."""
    result = resolve_api_source(api_name, lib_gitname)
    if result is None:
        return "unknown"
    return result["api_type"]


def get_docstring_from_source(api_name, lib_gitname):
    """
    Extract docstring from an API's source definition.
    Falls back to parent class docstring if the target has none.
    """
    result = resolve_api_source(api_name, lib_gitname)
    if result is None:
        return None

    node = result["node"]
    doc = ast.get_docstring(node)

    if doc:
        return doc

    # Fallback: if this is a method, try to find parent class docstring
    file_path = Path(result["file"])
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return None

    # Walk the AST to find if this method is inside a class
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == node.name:
                    class_doc = ast.get_docstring(parent)
                    if class_doc:
                        return class_doc
    return None


def get_source_ast(api_name, lib_gitname):
    """
    Return the AST tree of just the function/class source code.
    Returns None if the API cannot be resolved or parsed.
    """
    result = resolve_api_source(api_name, lib_gitname)
    if result is None:
        return None
    try:
        return ast.parse(result["source_code"])
    except SyntaxError:
        return None
