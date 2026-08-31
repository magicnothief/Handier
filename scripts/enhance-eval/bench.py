"""Run the self-correction suite against a model, prompt and strategy.

Usage:
  python bench.py --model PATH --prompt FILE [--think] [--two-pass] [--label L]

Prints a one-line summary plus every failure, so a run can be compared to
another without re-reading the whole transcript.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

from suite import ALL, check

EXE = "D:/python/Handy-Flow/src-tauri/binaries/handy-llm-x86_64-pc-windows-msvc.exe"

# Stage one of the two-pass strategy: a focused question rather than an open
# rewrite. Small models answer "what did they take back?" far more reliably
# than "rewrite this correctly", which is what the verifier results suggested.
DETECT_PROMPT = (
    "You read a dictated sentence and report whether the speaker took something back.\n"
    "\n"
    "People retract mid-sentence with phrases like: no wait, never mind, sorry, scratch that,\n"
    "I mean, I meant, or rather, actually no, make that, correction, hold on, hang on, strike\n"
    "that, forget that, my mistake, that's wrong. Any equivalent wording counts.\n"
    "\n"
    "If the speaker replaced something, reply with exactly one line:\n"
    "REPLACE <the wording they abandoned> WITH <the wording they chose>\n"
    "\n"
    "If they did not take anything back, reply with exactly:\n"
    "NONE\n"
    "\n"
    "Reply with nothing else."
)


def apply_prompt(detection):
    """Stage two: apply a detected replacement, stated plainly."""
    return (
        "You edit a dictated sentence.\n"
        "\n"
        f"The speaker took something back: {detection}\n"
        "\n"
        "Rewrite the sentence so it says only what they settled on. Delete the wording they\n"
        "abandoned and the phrase that signalled the change. Remove filler words and fix\n"
        "punctuation and capitalisation. Change nothing else.\n"
        "\n"
        "Reply with the edited sentence and nothing else."
    )


class Sidecar:
    def __init__(self, model, think, cpu=False):
        self.think = think
        self.budget = 1400 if think else 260
        self.errlog = open("stderr-bench.log", "w", encoding="utf-8")
        self.p = subprocess.Popen(
            [EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.errlog, text=True, encoding="utf-8", bufsize=1,
        )
        self.n = 0
        load = {"cmd": "load", "id": 1, "path": model}
        if cpu:
            load["gpu_layers"] = 0
        r = self._send(load)
        if not r.get("ok"):
            print("LOAD FAILED:", r.get("error"))
            sys.exit(1)

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            print("SIDECAR DIED — see stderr-bench.log")
            sys.exit(1)
        return json.loads(line)

    def gen(self, system, user, budget=None):
        self.n += 1
        r = self._send({
            "cmd": "generate", "id": 100 + self.n, "system": system, "user": user,
            "max_tokens": budget or self.budget, "no_think": not self.think,
        })
        return (r.get("text") or "").strip(), r.get("elapsed_ms") or 0

    def close(self):
        self._send({"cmd": "shutdown", "id": 99999})
        self.p.wait(timeout=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--two-pass", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--show-passes", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    system = open(args.prompt, encoding="utf-8").read().strip() if args.prompt else ""
    sc = Sidecar(args.model, args.think, cpu=args.cpu)

    passed, failures, times = 0, [], []
    t0 = time.time()
    for case in ALL:
        cid, text, _req, _forb, kind = case
        start = time.time()
        if args.two_pass:
            detection, _ = sc.gen(DETECT_PROMPT, text)
            first = detection.splitlines()[0].strip() if detection else "NONE"
            if first.upper().startswith("NONE") or not first.upper().startswith("REPLACE"):
                out, _ = sc.gen(system, text)
            else:
                out, _ = sc.gen(apply_prompt(first), text)
        else:
            out, _ = sc.gen(system, text)
        times.append((time.time() - start) * 1000)

        reason = check(case, out)
        if reason is None:
            passed += 1
            if args.show_passes:
                print(f"  ok   [{kind}] {cid}: {out}")
        else:
            failures.append(f"  FAIL [{kind}] {cid}: {reason}\n         in : {text}\n         out: {out}")

    sc.close()
    total = len(ALL)
    pct = 100.0 * passed / total
    print(f"\n=== {args.label} ===")
    print(f"  {passed}/{total} ({pct:.0f}%)  median {int(statistics.median(times))}ms  "
          f"wall {int(time.time() - t0)}s")
    for f in failures:
        print(f)
    return passed, total


if __name__ == "__main__":
    main()
