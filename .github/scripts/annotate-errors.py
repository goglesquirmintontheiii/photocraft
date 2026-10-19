"""Turns the first compiler/linker errors in a build log into GitHub annotations, so they can be
read through the API (fork release workflow). Usage: annotate-errors.py <log> <title>"""
import sys

log, title = sys.argv[1], sys.argv[2]
lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
blocks = []
for i, line in enumerate(lines):
    l = line.lower()
    if l.startswith("error") or "error:" in l and ("ld" in l or "link" in l or "failed to run" in l) or "undefined reference" in l or "panicked" in l:
        blocks.append("\n".join(lines[max(0, i - 3) : i + 25]))
    if len(blocks) >= 6:
        break
if not blocks:
    blocks = ["\n".join(lines[-60:])]
for b in blocks:
    msg = b[:3800].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={title}::{msg}")
