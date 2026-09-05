# Enhancement eval

The self-correction behaviour — cutting the wording a speaker retracted — is the
hardest thing this layer does and the easiest to fool yourself about. An early
smoke test here reported it working when its cases were, verbatim, the few-shot
examples in the prompt: it measured memorisation, not capability.

This suite exists so that does not happen again. 68 cases: 46 retractions that
must be cut, 22 preservations that must survive untouched. The preservation
cases deliberately contain the same signal words as the retractions, used
innocently ("sorry I am late", "wait for the build"), because a model that keys
on vocabulary rather than meaning will damage them.

## Benching a new model

Needs a built sidecar (`bun run build:sidecar`) and a GGUF on disk. Three steps,
then two numbers.

### 1. Pick the flags that match how the model was trained

Get this wrong and you measure a format mismatch, not the model. It is the single
most common way to conclude a good checkpoint is bad.

| What you are benching                                            | Flags                                 |
| ---------------------------------------------------------------- | ------------------------------------- |
| Fine-tune on the **chat** corpus (`handy_editor.jsonl`)          | `--no-switch`                         |
| Fine-tune on the **Alpaca** corpus (`handy_editor_alpaca.jsonl`) | `--alpaca`                            |
| Stock instruct model (Qwen, Gemma, Llama…)                       | `--prompt shipped_prompt.txt`         |
| Stock **reasoning** model                                        | `--prompt shipped_prompt.txt --think` |

Why they exist:

- `--no-switch` stops the harness appending `/no_think` to the transcript. A
  fine-tune has never seen that token and will try to edit it.
- `--alpaca` assembles the full Alpaca prompt — preamble, `### Instruction:`,
  `### Input:`, `### Response:` — into **one** turn, reading
  `../enhance-train/alpaca_instruction.txt` so the bench cannot drift from what
  the model was trained on and what the app sends.
- Omitting `--prompt` sends no instruction, which is what a fine-tune wants.

**Not sure which?** A converted base model often still carries a chat template,
so you cannot tell from the GGUF. Spend a dozen generations finding out instead
of guessing — run the same four cases under each shape and look at the output.
Runaway repetition that never stops is what a format mismatch looks like:

```
--- instruction in system slot
out: Book it for six people. Make that six people. If you have a house, you can
     use it for that. If you have a house, you can use it for that. …   [619ms]

--- whole Alpaca prompt as one turn
out: Book it for six people.                                            [103ms]
```

To see the exact string the sidecar builds — rather than reasoning about the
template — set `HANDY_LLM_DEBUG_PROMPT=1` in its environment and read stderr.

### 2. Run the suite

```bash
python bench.py --model D:/dev/handy-models/mine.Q4_K_M.gguf --no-switch --label mine
```

```
=== mine ===
  66/68 (97%)  median 94ms  wall 6s
  cut: 44/46  keep: 22/22
  FAIL [cut] pos-start: kept 'friday'
         in : no wait not friday the meeting is on saturday
         out: No wait, not Friday, the meeting is on Saturday.
```

**Read the second line, not the first.** `cut` is retractions that had to be
deleted; `keep` is sentences that had to survive untouched. They are not
interchangeable: a model that deletes eagerly gets a good total while destroying
text that was already correct, which is the failure users actually notice. A
drop in `keep` is much worse than the same drop in `cut`.

Every failure prints its input and output, so you can see _how_ it was wrong.

### 2b. For a chat fine-tune, try both system-turn shapes

`llama.cpp` does not run the GGUF's jinja template; it matches the template to a
built-in family and renders that. Its chatml renderer emits the system block even
when the content is empty, where the LFM2 template
(`../enhance-train/lfm2_chatml_template.jinja`) drops it:

```text
default:         <|im_start|>system\n<|im_end|>\n<|im_start|>user\n{transcript}...
--omit-system:   <|im_start|>user\n{transcript}...
```

Both go through the template, so this is a real comparison and not a fallback.
Which one a checkpoint wants cannot be deduced — measure it:

```bash
python bench.py --model mine.gguf --no-switch                  --label with-block
python bench.py --model mine.gguf --no-switch --omit-system    --label no-block
```

Measured on two chat fine-tunes, the block wins — and on the second the wrong
shape is not a degradation but a collapse:

| Model          | empty block (default) | `--omit-system` |
| -------------- | --------------------: | --------------: |
| current editor |             **68/68** |            2/68 |
| earlier editor |             **66/68** |           59/68 |

On the earlier one the loss also landed where it hurts most: `keep` fell from
22/22 to 18/22, so the wrong shape made it start damaging sentences that needed
no edit at all.

To see the exact string rather than infer it, run the sidecar with
`HANDY_LLM_DEBUG_PROMPT=1` and read stderr. `HANDY_LLM_BIN` points the harness at
a build other than the staged one.

### 3. Run the held-out eval

The suite is 68 hand-written cases and cannot carry a decision on its own — a
Q2_K_L build once scored 66/68 on it while being broken. This is the number to
trust:

```bash
python eval_heldout.py --model D:/dev/handy-models/mine.Q4_K_M.gguf \
    --eval D:/dev/handy-train-alpaca/handy_eval.jsonl --label mine
```

```
=== mine :: mine.Q4_K_M.gguf ===
  rows 2152   median 104ms
  edit    exact 1934/2152 (89.9%)   mean word-F1 0.983
```

2,152 rows held out **by source**, not sliced off the training file. Use the same
prompt flags as step 2, with one difference: this script has no `--no-switch`,
because _not_ appending `/no_think` is already its default. Pass `--switch` only
for a stock reasoning model.

It takes a few minutes; `--limit 600` gives a good estimate in a fraction of the
time, and `--show 0` hides the example failures.

### Comparing models fairly

- Same eval file, same flags, same `--seed` if you use `--limit`.
- Sampling is greedy, so scores are exactly reproducible. **Latency is not** —
  the same build measured 94–126 ms across runs depending on machine load. Do not
  read anything into a 20 ms difference.
- Quantisations need their own runs. Q8_0 is not automatically better than
  Q4_K_M; on the current editor they tie.
- Close the app first, or its sidecar will hold the GPU and skew everything.

## What it measured

Ranked by size; all with the shipped prompt, on an RTX 3060.

| Model                             | Size        | Score     |
| --------------------------------- | ----------- | --------- |
| Qwen3 0.6B                        | 0.40 GB     | 32/68     |
| LFM2.5 1.2B                       | 0.73 GB     | 30/68     |
| Gemma 3 1B                        | 0.81 GB     | 27/68     |
| Qwen3 1.7B                        | 1.11 GB     | 35/68     |
| Qwen2.5 1.5B                      | 1.12 GB     | 52/68     |
| **Qwen3 4B Instruct, UD-IQ3_XXS** | **1.67 GB** | **68/68** |
| Qwen2.5 3B                        | 1.93 GB     | 67/68     |
| Phi-4-mini                        | 2.49 GB     | 58/68     |
| Qwen3 4B Instruct, Q4_K_M         | 2.50 GB     | 68/68     |
| Qwen3 8B                          | 5.03 GB     | 68/68     |

Three findings worth keeping:

- **Nothing below 4B parameters reached 100%**, and scale alone did not fix it
  either: 8B scored the same as 4B. Prompt wording closed the last two cases on
  both.
- **Quantisation matters less than parameter count.** The 4B at IQ3_XXS
  (1.67 GB) beats Qwen2.5 3B at Q4 (1.93 GB) while being smaller.
- **A two-pass "detect then apply" strategy made things worse** — 66/68 down to
  43/68 on the same model — because a wrong detection poisons the second stage.
  Single pass is both simpler and better.

## Fine-tuning a small model

A full fine-tune of `LiquidAI/LFM2.5-350M` on `google-research-datasets/disfl_qa`,
exported at three quantisations. Scored with no system prompt, since the fine-tune
was trained without one; the base model is scored with the shipped prompt, which is
how the app runs it. `--no-switch` throughout, because a fine-tune has never seen
the `/no_think` token and appending it drops a stray word into the transcript.

| Model                                       | Size        | cut (46)  | keep (22) | Score     | Median    |
| ------------------------------------------- | ----------- | --------- | --------- | --------- | --------- |
| LFM2.5-350M stock, no prompt                | 0.23 GB     | 0/46      | 2/22      | 2/68      | 78 ms     |
| LFM2.5-350M stock, shipped prompt           | 0.23 GB     | 12/46     | 17/22     | 29/68     | 111 ms    |
| **ckpt-810 fine-tune, Q4_K_M**              | **0.23 GB** | **38/46** | **19/22** | **57/68** | **94 ms** |
| ckpt-810 fine-tune, Q8_0                    | 0.38 GB     | 36/46     | 14/22     | 50/68     | 95 ms     |
| ckpt-810 fine-tune, F16                     | 0.71 GB     | 35/46     | 13/22     | 48/68     | 101 ms    |
| ckpt-810 fine-tune, Q4_K_M + shipped prompt | 0.23 GB     | 36/46     | 18/22     | 54/68     | 107 ms    |
| Qwen3 4B Instruct UD-IQ3_XXS (shipped)      | 1.67 GB     | 46/46     | 22/22     | 68/68     | 350 ms    |

- **Fine-tuning worked.** 29/68 to 57/68 against the same base model at the same
  size, and it improved both directions at once — cutting went 12/46 to 38/46
  without costing preservation. At 94 ms it is 3.7x faster than the shipped 4B
  and a seventh of the size.
- **It is not yet a replacement.** The eleven remaining failures are not near
  misses; most are invented text. Trained on disfl_qa alone the model learned that
  the answer is always a SQuAD question, and answers `we we need to to check the
logs` with `What is the name of the log file?`. See
  `../enhance-train/README.md` for the corpus change that targets this.
- **The heavier quantisations scored worse**, consistently on both sub-scores
  (Q4_K_M > Q8_0 > F16). That is the opposite of the usual ordering and 68 cases
  is too few to trust it; treat it as a reason to score every quantisation rather
  than as an established result.
- **Giving the fine-tune a system prompt made it worse** (57 to 54) and changed
  how it failed: it began copying the prompt's own rules and few-shot examples
  into the output. A model fine-tuned without a prompt should be run without one.
  Caveat on the ckpt-810 rows: its training corpus and this suite share their signal
  vocabulary, so they are not held out the way the base-model rows above are.

## Second fine-tune (ckpt-10000), on the rebuilt corpus

Trained on `D:/dev/handy-train-v2` — declarative sources added, disfl_qa capped,
long-form included, promptless. Same suite, same harness:

| Model                        | Size        | cut (46)  | keep (22) | Score     | Median    |
| ---------------------------- | ----------- | --------- | --------- | --------- | --------- |
| ckpt-810 Q4_K_M (previous)   | 0.23 GB     | 38/46     | 19/22     | 57/68     | 94 ms     |
| ckpt-10000 Q2_K_L            | 0.18 GB     | 44/46     | 22/22     | 66/68     | 47 ms     |
| **ckpt-10000 Q4_K_M**        | **0.23 GB** | **44/46** | **22/22** | **66/68** | **92 ms** |
| ckpt-10000 F16               | 0.71 GB     | 44/46     | 22/22     | 66/68     | 98 ms     |
| Qwen3 4B Instruct UD-IQ3_XXS | 1.67 GB     | 46/46     | 22/22     | 68/68     | 350 ms    |

57 → 66, and preservation went from 19/22 to a clean 22/22 — the invented-question
failure mode is gone. The two remaining misses are a retraction that opens the
sentence (`no wait not friday the meeting is on saturday`) and one where it moved
the replacement onto the wrong slot (`send the audit report to the finance team`
instead of `the quarterly report to the audit team`).

## Held-out eval

`bench.py` measures 68 hand-written editing cases. It does not test the verifier at
all, and the training corpus now shares its signal vocabulary. `eval_heldout.py`
scores `handy_eval.jsonl`, which is held out by source. 1200 sampled rows — 489
edit, 711 verify:

| Model                      | edit exact |   word-F1 |    verify |    SAME | CHANGED | Median |
| -------------------------- | ---------: | --------: | --------: | ------: | ------: | -----: |
| **ckpt-10000 Q4_K_M**      |  **96.9%** | **0.999** | **99.7%** | 390/390 | 319/321 |  88 ms |
| ckpt-10000 F16             |      97.5% |     0.999 |     99.6% | 390/390 | 318/321 |  90 ms |
| ckpt-10000 Q2_K_L          |      55.0% |     0.911 |     23.9% |  25/390 | 145/321 |  66 ms |
| Qwen3 4B (shipped prompts) |      35.0% |     0.915 |     83.5% | 313/390 | 281/321 | 242 ms |

Four things this says, in order of how much they should change what you do:

- **Q2_K_L is a trap.** It scores 66/68 on the suite and is broken here: its
  verifier answers CHANGED for almost everything (SAME 25/390), which in the app
  means nearly every edit gets thrown away. The suite could not see this because
  it never asks the model to verify. Do not ship it. Q4_K_M is the floor.
- **F16 buys nothing** over Q4_K_M at 3x the size. Ship Q4_K_M.
- **The edit exact-match flatters the fine-tune**, because the reference style is
  precisely what it was trained to emit. Qwen3-4B's 35% is mostly a different but
  valid house style, which is why word-F1 (0.915 vs 0.999) is the fairer column.
- **The verify column flatters the fine-tune too**, and more seriously: the
  corruptions in this eval come from the same `corrupt()` generator used in
  training, so the model has learned that distribution. It measures "can it catch
  the corruptions we synthesise", not "can it catch this editor's real mistakes".
  The honest version is to run the verifier over the editor's own live output, the
  way `verify_pipeline.py` did for the prompted model. That number does not exist
  yet for the fine-tune.

## Does the verify pass earn its place?

`verify_value.py` answers the question the held-out eval could not: run the editor
for real, label its output against the reference, then ask the verifier about the
editor's _own_ text. 400 held-out edits, ckpt-10000 Q4_K_M, promptless:

```
400 edits   editor correct 390 (97.5%)   wrong 10 (2.5%)

LLM verify pass:
  caught          0 of 10 bad edits
  FALSE REJECT    1 of 390 good edits
  net            -1 edits

deterministic triage alone:
  flags           2 of 10 bad edits
  flags         172 of 390 good edits   (44% would be referred)
```

**The verify pass is net-negative.** It caught nothing and destroyed a good edit —
a correct long-range retraction, which is the flagship feature and the exact class
of bug that prompted this work in the first place.

Why it fails despite scoring 99.7% on the held-out set: the eval's corruptions come
from `corrupt()`, which swaps entities, reorders clauses and inserts words. The
editor's real mistakes are nothing like that. Reading all ten: it drops a leading
prepositional phrase (`Between Bingen and Bonn,`), or keeps a false start it should
have cut (`We you'd have to just sit and wait`). The verifier has never seen an
error of that shape. A 99.7% in-distribution score translated to a 0% catch rate.

A second finding from the same run: **the true editor error rate is well under
2.5%.** Of the ten flagged, only about three are genuine mistakes. The rest are
disagreements with a reference inherited from raw Switchboard annotation, and in
several the model's answer is plainly better than the reference:

```
want , but, they're not very old, so they couldn't do a whole lot, yet.
got  They're not very old, so they couldn't do a whole lot, yet.
```

So the base rate of real errors is around 1%. At that rate a verifier has to be
near-perfect to break even, because it sees ~99 good edits for every bad one.

**Conclusion: drop the LLM verify pass.** Keep `sanity_check` (length ratio, empty
output) — deterministic, free, and it catches the gross failures. The corpus budget
it frees is better spent on the editor, and better still on the _specific_ patterns
above than on raw volume.

## Third fine-tune (editor-only corpus) — regressed

Trained on `D:/dev/handy-train-editor`, the editor-only build, with completion-only
loss enabled. It is worse than the run before it on every axis.

| Model                       | Q4_K_M    | Q8_0      |
| --------------------------- | --------- | --------- |
| ckpt-810 (run 1)            | 57/68     | 50/68     |
| **ckpt-10000 (run 2)**      | **66/68** | **66/68** |
| ckpt-3000 (run 3)           | 37/68     | 62/68     |
| final export (run 3, later) | 25/68     | 61/68     |

Held-out, 900 rows of `handy-train-editor/handy_eval.jsonl` (edit rows only):

| Model             | exact | word-F1 |
| ----------------- | ----: | ------: |
| ckpt-10000 Q4_K_M | 90.1% |   0.986 |
| ckpt-3000 Q8_0    | 78.7% |   0.955 |
| ckpt-3000 Q4_K_M  | 26.8% |   0.467 |

Two things stand out.

**Training longer made it worse**, not better: Q4_K_M went 37 -> 25 between the two
run-3 checkpoints. So this is not simple undertraining.

**The run is quantisation-fragile.** Run 2 was invariant — 66/66/66 across Q2*K_L,
Q4_K_M and F16. Run 3 swings 25-37 at Q4 against 61-62 at Q8. Comparing per-block
Q8_0 scales between run 2 and run 3 showed a ratio of 1.00x on every tensor, so
this is \_not* weight blow-up from too high a learning rate; that check turned out
to be insensitive, because block scales are dominated by the base model and barely
move under fine-tuning.

The failure shapes point somewhere else — the model under-generates:

```
in : it takes two weeks i mean three weeks       out: (empty)
in : wait for the build to finish before deploy  out: (empty)
in : let's meet tuesday actually no thursday     out: Let's meet.
```

Empty and truncated completions are what a **misaligned response mask** produces,
not what a bad learning rate produces. `train_on_responses_only` needs the exact
turn markers for the model's own template — for LFM2 that is `<|im_start|>user` /
`<|im_start|>assistant`, not the Llama-3 or Gemma strings the tooling defaults to.
If the mask lands in the wrong place the model is trained on almost nothing but the
end of the turn, which is consistent with everything above.

**Keep ckpt-10000 Q4_K_M until a run-3 variant beats it.** The cheapest
discriminating experiment is to repeat run 3 with completion-only loss off and
nothing else changed.
