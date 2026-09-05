<!--
  Template for the Hugging Face model repo. Two things to do before publishing:

  1. Replace every TODO. The evaluation numbers are deliberately blank — this
     model has not been trained yet, and a card with invented scores is worse
     than one with none.
  2. Copy `license`, `license_name` and `license_link` verbatim from the base
     model's card (LiquidAI publishes LFM2.5 under its own open licence, not a
     standard SPDX one). Do not guess them.
-->

---

base_model: LiquidAI/LFM2.5-350M-Base
datasets:

- TODO/handy-dictation-editing
  language:
- en
  library_name: transformers
  pipeline_tag: text2text-generation
  license: other
  license_name: TODO-copy-from-base-model
  license_link: TODO-copy-from-base-model
  tags:
- dictation
- disfluency-removal
- text-editing
- speech
- alpaca
- gguf

---

# TODO-model-name

A 350M editor that turns a raw dictated transcript into the text the speaker
meant to write.

```
in : um so the meeting is uh moved to friday no wait thursday at three
out: The meeting is Thursday at three.
```

It does three things, which in speech are one thing: drops filler words, repairs
punctuation and capitalisation, and — the part general-purpose models of this
size get wrong — when the speaker changes their mind mid-sentence it deletes the
wording they abandoned and keeps only what they settled on.

Fine-tuned from [`LiquidAI/LFM2.5-350M-Base`](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
on [TODO/handy-dictation-editing](https://huggingface.co/datasets/TODO/handy-dictation-editing),
for [Handy](https://github.com/zselybence/Handier)'s on-device enhancement
layer: a dictation app that has to clean up your words locally, on a laptop with
no GPU, without adding a noticeable pause before the text appears.

## Prompt format

This is fine-tuned from a **base** checkpoint, so it has **no chat template**.
It expects the Alpaca prompt, and the instruction below is the one it was trained
on — send it **verbatim**. A fine-tune given a paraphrase of its own instruction
degrades quietly rather than failing loudly.

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Rewrite the dictated transcript as the speaker meant to write it. Remove filler words and hesitations, repair punctuation and capitalisation, and when the speaker corrects themselves keep only the wording they settled on. Reply with the edited text and nothing else.

### Input:
{your transcript}

### Response:
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TODO/model-name"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

INSTRUCTION = (
    "Rewrite the dictated transcript as the speaker meant to write it. Remove "
    "filler words and hesitations, repair punctuation and capitalisation, and "
    "when the speaker corrects themselves keep only the wording they settled "
    "on. Reply with the edited text and nothing else."
)
PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

text = "um so the meeting is uh moved to friday no wait thursday at three"
inputs = tok(PROMPT.format(instruction=INSTRUCTION, input=text), return_tensors="pt")
# Editing is near-deterministic; sampling turns a working transcript into a
# creative one.
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Evaluation

<!-- Fill these in from an actual run. Do not carry over the numbers from the
     instruct-model fine-tune below: it is a different base and a different
     prompt format. -->

Two independent measurements, both in
[`scripts/enhance-eval/`](https://github.com/zselybence/Handier/tree/main/scripts/enhance-eval):

| Metric                                               | Result      |
| ---------------------------------------------------- | ----------- |
| Self-correction suite (68 cases, `bench.py`)         | TODO / 68   |
| — of which `cut` (46) / `keep` (22)                  | TODO / TODO |
| Held-out exact match (`eval_heldout.py`, 2,152 rows) | TODO %      |
| Held-out word F1                                     | TODO        |
| Median latency                                       | TODO ms     |
| Size (Q4_K_M GGUF)                                   | TODO MB     |

```bash
python bench.py --model /path/to/model.gguf --no-switch --label my-run
python eval_heldout.py --model /path/to/model.gguf --data handy_eval_alpaca.jsonl
```

The two measure different things and both matter. The suite is 68 hand-written
cases split into edits that _must_ cut something and edits that must **not**; the
held-out set is 2,152 rows from sources the training build never read. A model
can score well on one and badly on the other — an earlier quantisation passed the
suite at 66/68 while collapsing on the held-out set.

**Report `keep` separately from `cut`.** A model that deletes aggressively scores
well on a single total while destroying sentences that were already correct, and
that is the failure users actually notice.

### For reference: the instruct-model sibling

A different run — LFM2.5-350M **Instruct**, chat format, empty system turn —
reached 66/68 on the suite at 92 ms and 229 MB (Q4_K_M), against 68/68 at 350 ms
and 1.67 GB for a Qwen3-4B baseline. That is context for what this size can do,
**not** a result for this model.

## Quantisation

Q4_K_M is the floor that has been checked; F16 bought nothing over it at three
times the size. Verify any quantisation you publish against **both** evaluations
above — a quant that passes the 68-case suite can still be broken in ways only
the held-out set exposes.

## Using it in Handy

Handy speaks to a local GGUF directly:

1. Convert and quantise to GGUF.
2. **Settings → Models → Enhancement Models → Your Own Model → Choose a GGUF
   file…**
3. Set **How to prompt this model** to **Fine-tuned from a base model**. That is
   what makes the app send the Alpaca prompt above instead of its own instruction
   prompt; leaving it on the default sends the wrong format.

## Limitations

- **English only.**
- **Opinionated punctuation.** Sentence case, full stops added, serial commas
  absent. It will impose that style.
- **It deletes on purpose.** Cutting retracted wording is the feature, so a
  mistake looks like missing words rather than garbled ones. Handy keeps the raw
  transcript in history for this reason; any host should do the same.
- **Not a general instruction-following model.** It was trained on one task with
  one instruction and will not do anything else usefully.
- **Short utterances dominate its training.** Long-form dictation is ~10% of the
  corpus.
- **It cannot judge its own edits.** Asking it whether an edit preserved meaning
  produces a confident, meaningless answer; measured, that pass caught 0 of 10
  bad edits and rejected a good one.

## Training

<!-- Fill in from the run that produced this model. -->

|                |                                                             |
| -------------- | ----------------------------------------------------------- |
| Base           | `LiquidAI/LFM2.5-350M-Base`                                 |
| Data           | 89,996 Alpaca rows (15.6% real speech)                      |
| Method         | TODO (full fine-tune / LoRA / QLoRA)                        |
| Context length | TODO (512 truncates nothing, by 2 tokens; 640 gives margin) |
| Epochs / steps | TODO                                                        |
| LR / scheduler | TODO                                                        |
| Hardware       | TODO                                                        |

Train on completions only if your trainer supports it, and give it the **right
turn markers** — an Alpaca run splits on `### Response:\n`. Getting that wrong
does not error; it produces a model that emits empty or truncated completions.

## Licence

Inherits the base model's licence. Copy `license`, `license_name` and
`license_link` from
[`LiquidAI/LFM2.5-350M-Base`](https://huggingface.co/LiquidAI/LFM2.5-350M-Base)
rather than guessing.

The training data is CC-BY-4.0 and requires attribution to `disfl_qa`,
`nyralabs/disfluency_speech_english` and `amaai-lab/DisfluencySpeech`.
