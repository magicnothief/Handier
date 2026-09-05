"""Run the self-correction suite against a model, prompt and strategy.

Usage:
  python bench.py --model PATH --prompt FILE [--think] [--two-pass] [--label L]

Prints a one-line summary plus every failure, so a run can be compared to
another without re-reading the whole transcript.
"""

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

from suite import ALL, check

EXE = os.environ.get(
    "HANDY_LLM_BIN",
    "D:/python/Handy-Flow/src-tauri/binaries/handy-llm-x86_64-pc-windows-msvc.exe",
)

# Shared with the corpus builder and the app, so the bench cannot measure a
# prompt different from the one the model was trained on or will be served.
ALPACA_INSTRUCTION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "enhance-train" / "alpaca_instruction.txt"
).read_text(encoding="utf-8").strip()

ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
)

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
    def __init__(self, model, think, cpu=False, switch=True, omit_system=False):
        self.think = think
        # A fine-tuned model has never seen the `/no_think` switch, so appending
        # it puts a stray token in the middle of the transcript it is asked to
        # edit. Suppress it for anything that does not deliberate by default.
        self.switch = switch
        self.budget = 1400 if think else 260
        self.errlog = open("stderr-bench.log", "w", encoding="utf-8")
        # The sidecar reads this at prompt-build time, so it has to be set on
        # the process rather than passed per request.
        env = {**os.environ}
        env["HANDY_LLM_OMIT_EMPTY_SYSTEM"] = "1" if omit_system else "0"
        self.p = subprocess.Popen(
            [EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.errlog, text=True, encoding="utf-8", bufsize=1, env=env,
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
            "max_tokens": budget or self.budget,
            "no_think": self.switch and not self.think,
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
    ap.add_argument("--no-switch", action="store_true",
                    help="never append /no_think (for fine-tuned, non-reasoning models)")
    ap.add_argument("--omit-system", action="store_true",
                    help="send no system message at all, instead of an empty one. "
                         "Some chat templates render these differently; which one a "
                         "fine-tune wants is a measurement, so try both.")
    ap.add_argument("--alpaca", action="store_true",
                    help="send the Alpaca prompt as the user turn, for a fine-tune "
                         "trained on instruction/input/output. Implies --no-switch.")
    args = ap.parse_args()

    system = open(args.prompt, encoding="utf-8").read().strip() if args.prompt else ""

    # An Alpaca fine-tune wants the whole prompt as one turn, not an instruction
    # in the system slot. Measured on checkpoint-5000, splitting it across system
    # and user made the model repeat itself until the token budget ran out and
    # never emit a stop token; the same model given the assembled prompt answered
    # cleanly in a sixth of the time. Any chat template the GGUF carries still
    # wraps this, which is fine -- the Alpaca text is what the model keys on.
    def wrap(text: str) -> str:
        if not args.alpaca:
            return text
        return (
            f"{ALPACA_PREAMBLE}### Instruction:\n{ALPACA_INSTRUCTION}"
            f"\n\n### Input:\n{text}\n\n### Response:\n"
        )

    if args.alpaca:
        system = ""

    sc = Sidecar(args.model, args.think, cpu=args.cpu,
                 switch=not (args.no_switch or args.alpaca),
                 omit_system=args.omit_system)

    passed, failures, times = 0, [], []
    # Split the score by direction. Failing to cut a retraction leaves noise;
    # damaging a sentence that needed no edit destroys what the speaker meant,
    # so the two numbers are not interchangeable and a single total hides which
    # kind of mistake a model makes.
    by_kind = {}
    t0 = time.time()
    for case in ALL:
        cid, text, _req, _forb, kind = case
        start = time.time()
        if args.two_pass:
            detection, _ = sc.gen(DETECT_PROMPT, text)
            first = detection.splitlines()[0].strip() if detection else "NONE"
            if first.upper().startswith("NONE") or not first.upper().startswith("REPLACE"):
                out, _ = sc.gen(system, wrap(text))
            else:
                out, _ = sc.gen(apply_prompt(first), text)
        else:
            out, _ = sc.gen(system, wrap(text))
        times.append((time.time() - start) * 1000)

        reason = check(case, out)
        tally = by_kind.setdefault(kind, [0, 0])
        tally[1] += 1
        if reason is None:
            tally[0] += 1
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
    print("  " + "  ".join(f"{k}: {ok}/{n}" for k, (ok, n) in sorted(by_kind.items())))
    for f in failures:
        print(f)
    return passed, total


if __name__ == "__main__":
    main()
