"""Run the suite across every downloaded model, smallest first.

The goal is the smallest model that scores 100%, so results are reported in
ascending size with the file size alongside — that is the number that decides
whether it fits a given card.
"""

import glob
import os
import subprocess
import sys

MODELS_DIR = "D:/dev/handy-models"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "cur_prompt.txt"
ONLY = sys.argv[2:] if len(sys.argv) > 2 else None

# Models whose chat template needs deliberation suppressed differ from those
# that never deliberate; the sidecar's no_think switch is harmless on the
# latter, so it is applied uniformly except where thinking is being tested.
paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.gguf")), key=os.path.getsize)
if ONLY:
    paths = [p for p in paths if any(o.lower() in os.path.basename(p).lower() for o in ONLY)]

rows = []
for path in paths:
    name = os.path.basename(path).replace(".gguf", "")
    size_gb = os.path.getsize(path) / 1e9
    print(f"\n{'=' * 70}\n{name}  ({size_gb:.2f} GB)\n{'=' * 70}", flush=True)
    try:
        out = subprocess.run(
            [sys.executable, "bench.py", "--model", path, "--prompt", PROMPT,
             "--label", name],
            capture_output=True, text=True, timeout=900,
        ).stdout
    except subprocess.TimeoutExpired:
        print("  TIMED OUT")
        rows.append((size_gb, name, None, None))
        continue

    score = next((l.strip() for l in out.splitlines() if "/68" in l), "no result")
    print("  " + score)
    for line in out.splitlines():
        if line.startswith("  FAIL"):
            print("   " + line.strip())
    passed = None
    if "/68" in score:
        try:
            passed = int(score.split("/68")[0].split()[-1])
        except (ValueError, IndexError):
            pass
    rows.append((size_gb, name, passed, score))

print(f"\n\n{'=' * 70}\nSUMMARY (ascending size)\n{'=' * 70}")
for size_gb, name, passed, _ in rows:
    mark = "  <-- 100%" if passed == 68 else ""
    got = f"{passed}/68" if passed is not None else "  n/a"
    print(f"  {size_gb:5.2f} GB  {got:>7}  {name}{mark}")
