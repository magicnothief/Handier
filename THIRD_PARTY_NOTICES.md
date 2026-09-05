# Third-party notices

Handy-Flow is distributed under the MIT licence (see [`LICENSE`](LICENSE)). It
also ships, links, or downloads third-party work that carries its own terms.

This file records what those are. It is a summary for people redistributing the
app — it is not legal advice, and the authoritative text is always the upstream
project's own licence file.

## Upstream project

Handy-Flow is a fork of **[Handy](https://github.com/cjpais/Handy)**, MIT
licensed, Copyright (c) 2025 CJ Pais. The MIT terms and copyright notice in
[`LICENSE`](LICENSE) are inherited from it and must be preserved in any further
redistribution.

## Bundled in the installer

### The enhancement sidecar (`handy-llm`)

A separate executable shipped alongside the app. It statically links:

| Component                                                                | Licence           |
| ------------------------------------------------------------------------ | ----------------- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) and `ggml`            | MIT               |
| [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2), `llama-cpp-sys-2` | MIT OR Apache-2.0 |

Because llama.cpp is linked into a binary we distribute, its MIT notice travels
with the installer. The crate's vendored copy of llama.cpp does not include a
`LICENSE` file — crate packaging strips it — so the authoritative text is the
[upstream repository](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).

On Windows the sidecar is built with the Vulkan backend and links the system
Vulkan loader dynamically; no Vulkan SDK component is redistributed.

### Rust dependencies

436 crates in the application's shipped dependency graph, and 26 in the
sidecar's. The distribution is overwhelmingly permissive:

| Licence                                                            | Crates (app) |
| ------------------------------------------------------------------ | -----------: |
| MIT OR Apache-2.0 (in its various spellings)                       |         ~310 |
| MIT                                                                |          118 |
| Apache-2.0                                                         |           10 |
| MPL-2.0                                                            |           19 |
| Unicode-3.0                                                        |           18 |
| BSD-2/3-Clause, ISC, Zlib, BSL-1.0, CDLA-Permissive-2.0, Unlicense |          ~20 |

**No GPL, LGPL or AGPL code is linked**, which is what makes MIT redistribution
of the whole straightforward.

The 19 **MPL-2.0** components are worth naming because MPL is weak copyleft:

- `symphonia` and its codec/format crates (audio decoding)
- `cssparser`, `cssparser-macros`, `selectors`, `dtoa-short` (webview styling)
- `option-ext`

MPL-2.0 is file-level copyleft. Linking them into an MIT application is
permitted; the obligation is to make the source of _those files_ available if you
modify them. Handy-Flow does not modify any of them, so pointing at their upstream
repositories satisfies it.

Regenerate this inventory with:

```bash
cd src-tauri && cargo tree --edges normal --prefix none --no-dedupe
cd src-tauri && cargo metadata --format-version 1   # `license` field per package
```

## Downloaded at runtime, not bundled

Models are **not** shipped in the installer. The app fetches them from Hugging
Face on request, so each user obtains them directly from the publisher under that
publisher's terms. Handy-Flow does not redistribute model weights.

That said, the licences differ, and two of them are **not** open-source licences:

| Model family                      | Licence                         | Note                                                |
| --------------------------------- | ------------------------------- | --------------------------------------------------- |
| Qwen2.5 / Qwen3                   | Apache-2.0                      | Permissive                                          |
| SmolLM2                           | Apache-2.0                      | Permissive                                          |
| LFM2.5 (incl. the default editor) | **LFM Open License v1.0**       | Liquid AI's own terms; review before commercial use |
| Gemma 3                           | **Gemma Terms of Use**          | Google's own terms, with a use policy               |
| Llama 3.2                         | **Llama 3.2 Community License** | Meta's own terms, with an acceptable-use policy     |

The `license` field in
[`src-tauri/src/enhance/models.json`](src-tauri/src/enhance/models.json) records
this per model, and the app shows it in the model picker.

Transcription models (Whisper, Parakeet, Moonshine and relatives) are inherited
from upstream Handy and carry their own terms, likewise fetched at runtime.

### The default enhancement model

[`MagicNoThief/handy-editor-lfm2.5-350m`](https://huggingface.co/MagicNoThief/handy-editor-lfm2.5-350m)
is a fine-tune of `LiquidAI/LFM2.5-350M` and inherits the **LFM Open License
v1.0**.

Its training corpus,
[`MagicNoThief/handy-dictation-editing`](https://huggingface.co/datasets/MagicNoThief/handy-dictation-editing),
is CC-BY-4.0 and requires attribution to:

- [`google-research-datasets/disfl_qa`](https://huggingface.co/datasets/google-research-datasets/disfl_qa) — CC-BY-4.0
- [`nyralabs/disfluency_speech_english`](https://huggingface.co/datasets/nyralabs/disfluency_speech_english) — Apache-2.0
- [`amaai-lab/DisfluencySpeech`](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech) — Apache-2.0

> `disfl_qa` declares CC-BY-4.0 and derives from SQuAD, which is CC-BY-SA-4.0.
> The corpus follows the upstream declaration. If you need certainty about
> whether a share-alike obligation reaches the derivative, confirm it rather than
> relying on this note.
