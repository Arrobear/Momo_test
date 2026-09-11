import json
import os
from pathlib import Path

from rq3.manifest import validate_api_names


def requested_api_names():
    raw = os.environ.get("MOMO_API_INCLUDE")
    if not raw:
        return None
    return validate_api_names(json.loads(raw))


def filter_api_definition(path, requested=None):
    requested = requested if requested is not None else requested_api_names()
    if not requested:
        return None
    requested = validate_api_names(requested)

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    by_name = {}
    for line in lines:
        if line.strip():
            by_name.setdefault(line.split("(", 1)[0].strip(), []).append(line)
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise RuntimeError(f"Requested APIs were not extracted: {missing}")

    selected = [line for name in requested for line in by_name[name]]
    path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(
        f"[API filter] selected {len(selected)}/{len(lines)} APIs: "
        f"{', '.join(requested)}",
        flush=True,
    )
    return selected
