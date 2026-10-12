"""Check local Markdown links and anchors without requiring network access."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def main() -> None:
    failures: list[str] = []
    for document in ROOT.rglob("*.md"):
        if ".venv" in document.parts or "node_modules" in document.parts:
            continue
        for target in LINK.findall(document.read_text()):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http:", "https:", "mailto:")):
                continue
            candidate = (document.parent / target).resolve()
            if not candidate.exists():
                failures.append(f"{document}: {target}")
    if failures:
        raise SystemExit("Broken local links:\n" + "\n".join(failures))
    print("local links ok")


if __name__ == "__main__":
    main()
