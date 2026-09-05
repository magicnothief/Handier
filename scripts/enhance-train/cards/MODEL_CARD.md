---
base_model: LiquidAI/LFM2.5-350M
datasets:
  - MagicNoThief/handy-dictation-editing
language:
  - en
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE
tags:
  - dictation
  - disfluency-removal
  - text-editing
  - speech
  - gguf
  - lfm2.5
  - edge
---

# Handy Editor 350M

A 350M model that turns a raw dictated transcript into the text the speaker meant
to write.

```
in : um so the meeting is uh moved to friday no wait thursday at three
out: The meeting is Thursday at three.
```

Three jobs that in speech are one job: drop filler words, repair punctuation and
capitalisation, and — the part models of this size usually get wrong — when the
speaker changes their mind mid-sentence, delete the wording they abandoned and
keep only what they settled on.

It runs in **~100 ms in 229 MB** and scores **68/68** on a self-correction suite
— the same score a 4B general-purpose model needs 350 ms and 1.67 GB to reach.
7× smaller and ~3× faster at no cost in accuracy: this exists to run
locally, on a laptop with no GPU, without adding a pause you can feel before your
text appears.

Fine-tuned from [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M)
for [Handy](https://github.com/MagicNoThief/Handy-Flow)'s on-device enhancement
layer.

## Prompt format

Chat format, with the transcript as the **user** turn and **no instruction**. The
task is in the weights; adding the instruction back measured _worse_ (57/68 →
54/68 on an earlier checkpoint) because the model started copying the prompt's
own rules into its output.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MagicNoThief/handy-editor-lfm2.5-350m"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

messages = [{"role": "user", "content": "um so the meeting is uh moved to friday no wait thursday at three"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# Editing is near-deterministic. Sampling turns a working transcript into a
# creative one, which is the one failure users cannot forgive.
out = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### With llama.cpp, send an _empty_ system message

Not an absent one — the two are not the same thing here, and the difference is
worth 7 points.

`llama.cpp` does not evaluate the GGUF's jinja template; it matches it to a
built-in family and renders that. Its chatml renderer emits the system block even
when the content is empty, which the jinja template does not:

```text
<|im_start|>system
<|im_end|>
<|im_start|>user
um so the meeting is uh moved to friday no wait thursday at three<|im_end|>
<|im_start|>assistant
```

| System message                         |     Suite |
| -------------------------------------- | --------: |
| Empty string (renders the block above) | **66/68** |
| Omitted entirely                       |     59/68 |

In the 59/68 run the model began answering `SAME` and `CHANGED` to editing
requests — it had stopped recognising the shape of its own input.

With **transformers** this does not arise: `apply_chat_template` uses the jinja
source, which drops an empty system block, so the plain user-only message list in
the snippet above is already correct.

## Evaluation

Two independent measurements. Both matter, and a model can pass one while failing
the other.

| Metric                                             |     Q4_K_M |    Q8_0 |     F16 |
| -------------------------------------------------- | ---------: | ------: | ------: |
| Self-correction suite (68 cases)                   |  **68/68** |   68/68 |   68/68 |
| — `cut` (46 cases that must delete)                |  **46/46** |   46/46 |   46/46 |
| — `keep` (22 cases that must **not**)              |  **22/22** |   22/22 |   22/22 |
| Held-out exact match (2,152 rows, editor-only set) |  **97.4%** |   97.7% |   97.7% |
| Held-out mean word-F1                              |  **0.999** |   0.999 |   0.999 |
| Median latency (RTX 3060, Vulkan)                  |    ~100 ms | ~100 ms | ~120 ms |
| Size                                               | **229 MB** |  379 MB |  711 MB |

Generation is greedy, so the scores are reproducible rather than a good sample.
The latencies are not: across repeated runs the same build measured 94-126 ms
depending on what else the machine was doing. Treat them as an order of
magnitude and measure on your own hardware if it matters.

**`keep` is the number to watch.** A model that deletes eagerly scores well on a
single total while destroying sentences that were already correct, and that is
the failure a user actually notices. 22/22 means it never touched a sentence that
did not need touching.

The held-out set is held out **by source**, not by slicing the training file:
real utterances come from a split the training build never reads, synthetic rows
use a different seed with every training input excluded by hand, and the builder
asserts zero overlap. The suite is 68 hand-written cases, independent of both.

Reproduce with [`scripts/enhance-eval/`](https://github.com/MagicNoThief/Handy-Flow/tree/main/scripts/enhance-eval):

```bash
python bench.py --model handy-editor-350m.Q4_K_M.gguf --no-switch --label mine
python eval_heldout.py --model handy-editor-350m.Q4_K_M.gguf \
    --eval handy_eval.jsonl --label mine
```

### Why this checkpoint

Four fine-tunes were compared on the same held-out set:

| Run            | Base             | Corpus view |     Suite |         Held-out exact |
| -------------- | ---------------- | ----------- | --------: | ---------------------: |
| **this model** | LFM2.5-350M      | chat        | **68/68** |              **97.4%** |
| earlier        | LFM2.5-350M      | chat        |     66/68 |                  89.9% |
| 3,500 steps    | LFM2.5-350M-Base | Alpaca      |     61/68 |                  65.4% |
| 5,000 steps    | LFM2.5-350M-Base | Alpaca      |     60/68 | 44.2% (600-row sample) |

The two runs from the base checkpoint got _worse_ between step 3,500 and 5,000,
failing by producing the right edit and then continuing ("The conference is in
Austin. My mistake is in Vienna. That's not right. Austin. …") until the host's
length guard rejected the whole thing. Their prompt format was separately
verified correct, so that is a training result rather than a data-pipeline one.

Two variables move at once here — base checkpoint and corpus view — so this table
says which artefact to use, not which of the two mattered.

### Which file do I want?

| You have                      | Take              | Why                                      |
| ----------------------------- | ----------------- | ---------------------------------------- |
| < 6 GB RAM, or a busy machine | `Q4_K_M` (229 MB) | 97.4%, and 150 MB cheaper                |
| headroom to spare             | `Q8_0` (379 MB)   | **Lossless** — scores identically to F16 |
| plans to requantise           | `F16` (711 MB)    | Nothing else; see below                  |

Both run at the same speed (~100 ms), so this is purely a memory decision.

The interesting result is that **Q8_0 and F16 score identically** — 2103/2152
each, not approximately but exactly. Q8_0 therefore costs nothing in quality
against the full-precision weights, and F16 buys only disk. Q4_K_M is the only
one carrying measurable quantisation loss, and it is 6 rows in 2,152: real, but
far too small to notice in use. Prefer Q8_0 if the memory is free, Q4_K_M if it
is not, and do not agonise over it.

**Q2_K_L is published nowhere, and you should not make one.** It is the reason
this section exists:

|        |        Suite | Held-out exact |
| ------ | -----------: | -------------: |
| Q4_K_M | 68/68 (100%) |          97.4% |
| Q2_K_L |  62/68 (91%) |      **47.9%** |

On the 68-case suite Q2*K_L looks merely a little degraded — 91%, a number plenty
of people would ship on. On the held-out set it gets \_half* the edits wrong. A
68-case suite is structurally unable to see that, which is why any quantisation
you make must be run through both evaluations before you trust it. Do not infer
quality from the suite alone.

## Using it in Handy

1. **Settings → Models → Enhancement Models → Your Own Model → Choose a GGUF
   file…**
2. Leave **How to prompt this model** on **Fine-tuned for editing**. That is what
   sends the empty system turn instead of Handy's own instruction prompt.

## Limitations

- **English only.**
- **Opinionated punctuation.** Sentence case, full stops added, serial commas
  absent. It will impose that style on your dictation.
- **It deletes on purpose.** Cutting retracted wording is the feature, so its
  mistakes look like missing words rather than garbled ones. Handy keeps the raw
  transcript in history for exactly this reason; any host should do the same.
- **Not a general instruction-following model.** One task, one format. It will
  not do anything else usefully, and it has no chat ability worth the name.
- **Short utterances dominate its training.** Long-form dictation is ~10% of the
  corpus, and the two suite failures are both long-range retractions.
- **It cannot judge its own edits.** Asked whether an edit preserved meaning it
  gives a confident, meaningless answer: measured over 400 live edits, that pass
  caught 0 of 10 bad edits and rejected 1 good one. Do not build a verification
  step on it.

## Training

|                |                                                                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Base           | [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M)                                                              |
| Data           | [`handy-dictation-editing`](https://huggingface.co/datasets/MagicNoThief/handy-dictation-editing), 89,996 rows (15.6% real speech) |
| Format         | chat (`messages`), no system turn                                                                                                  |
| Context length | 512 (truncates nothing; max row is 426 tokens)                                                                                     |

The corpus is 19% examples that need **no** edit. That share is load-bearing:
trained only on corrections, a model learns that something must always be
deleted, and starts eating sentences that were fine.

## Licence

Inherits the base model's licence: **LFM Open License v1.0** (`lfm1.0`). The
terms are the base model's copy, which `license_link` points at directly:
[`LiquidAI/LFM2.5-350M/LICENSE`](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).

Training data is CC-BY-4.0 and requires attribution to
[`disfl_qa`](https://huggingface.co/datasets/google-research-datasets/disfl_qa),
[`nyralabs/disfluency_speech_english`](https://huggingface.co/datasets/nyralabs/disfluency_speech_english)
and [`amaai-lab/DisfluencySpeech`](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech).
