"""
AST-based source code resolver for libraries under dl_lib/.
Reads Python source files directly without importing the target library.
"""
import ast
import os
import re
import textwrap
from pathlib import Path

ROOT = Path(os.environ.get("MOMO_ROOT_PATH", Path(__file__).resolve().parent.parent))
DL_LIB_ROOT = Path(os.environ.get("MOMO_REPOSITORY_ROOT", ROOT / "dl_lib"))

# Cache: {lib_gitname: {api_name: (file_path, node)}}
_cache = {}


def _find_package_root(lib_gitname):
    """Locate the Python package directory under dl_lib/{lib_gitname}."""
    base = DL_LIB_ROOT / lib_gitname
    if not base.exists():
        return None

    # Try common layouts in order
    candidates = []

    # Single-module: dl_lib/black/black.py — prefer this over sub-packages
    single_module = base / f"{lib_gitname}.py"
    if single_module.exists():
        return base

    # src-layout: dl_lib/black/src/black/
    src_dir = base / "src"
    if src_dir.is_dir():
        for entry in src_dir.iterdir():
            if entry.is_dir() and (entry / "__init__.py").exists():
                candidates.append(entry)
            elif entry.suffix == ".py":
                candidates.append(entry)

    # flat-layout: dl_lib/black/black/ (package dir)
    for entry in base.iterdir():
        if entry.is_dir() and entry.name not in ("src", ".git", "__pycache__", "tests", "docs", "test") and (entry / "__init__.py").exists():
            if entry not in candidates:
                candidates.append(entry)

    # If we found package dirs, use the first one
    if candidates:
        return candidates[0]

    # Fallback: return base itself (also handles single .py files)
    return base


def _build_index(lib_gitname):
    """Build a mapping of api_name -> (file_path, ast_node) for all top-level definitions."""
    pkg_root = _find_package_root(lib_gitname)
    if pkg_root is None:
        _cache[lib_gitname] = {}
        return

    index = {}

    if pkg_root.is_dir():
        # Walk all .py files
        for py_file in pkg_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            _index_file(py_file, pkg_root, index)
    elif pkg_root.suffix == ".py":
        _index_file(pkg_root, pkg_root.parent, index)

    _cache[lib_gitname] = index


def _index_file(py_file, pkg_root, index):
    """Parse a .py file and add its top-level definitions to the index."""
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return

    # Compute module path from file path
    if pkg_root.is_dir():
        rel_path = py_file.relative_to(pkg_root)
    else:
        rel_path = py_file.relative_to(pkg_root.parent)
    parts = list(rel_path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    module_prefix = ".".join(parts)

    for node in ast.iter_child_nodes(tree):
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    break

        if name:
            full_name = f"{module_prefix}.{name}" if module_prefix else name
            # Strip leading dots from relative imports
            full_name = full_name.lstrip(".")
            index[full_name] = (str(py_file), node)
            # Also index just the short name for direct lookups
            if name not in index:
                index[name] = (str(py_file), node)


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
