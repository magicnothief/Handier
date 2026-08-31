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

## Running it

Needs a built sidecar (`bun run build:sidecar`) and a GGUF on disk.

```bash
python bench.py --model /path/to/model.gguf --prompt prompt.txt          # one model
python bench.py --model ... --prompt ... --cpu                           # force CPU
python sweep.py prompt.txt                                               # every model in the dir
```

Dump the prompt the app actually sends, so the eval tests the shipped text:

```bash
cd ../../src-tauri
cargo test --lib enhance::prompt::tests::dump_default_prompt -- --nocapture \
  | sed -n '/<<<PROMPT_START>>>/,/<<<PROMPT_END>>>/p' | sed '1d;$d' > prompt.txt
```

## What it measured

Ranked by size; all with the shipped prompt, on an RTX 3060.

| Model | Size | Score |
| --- | --- | --- |
| Qwen3 0.6B | 0.40 GB | 32/68 |
| LFM2.5 1.2B | 0.73 GB | 30/68 |
| Gemma 3 1B | 0.81 GB | 27/68 |
| Qwen3 1.7B | 1.11 GB | 35/68 |
| Qwen2.5 1.5B | 1.12 GB | 52/68 |
| **Qwen3 4B Instruct, UD-IQ3_XXS** | **1.67 GB** | **68/68** |
| Qwen2.5 3B | 1.93 GB | 67/68 |
| Phi-4-mini | 2.49 GB | 58/68 |
| Qwen3 4B Instruct, Q4_K_M | 2.50 GB | 68/68 |
| Qwen3 8B | 5.03 GB | 68/68 |

Three findings worth keeping:

- **Nothing below 4B parameters reached 100%**, and scale alone did not fix it
  either: 8B scored the same as 4B. Prompt wording closed the last two cases on
  both.
- **Quantisation matters less than parameter count.** The 4B at IQ3_XXS
  (1.67 GB) beats Qwen2.5 3B at Q4 (1.93 GB) while being smaller.
- **A two-pass "detect then apply" strategy made things worse** — 66/68 down to
  43/68 on the same model — because a wrong detection poisons the second stage.
  Single pass is both simpler and better.
