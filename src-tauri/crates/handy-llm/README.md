# handy-llm

Out-of-process GGUF inference for Handy-Flow's transcript enhancement layer.

Reads line-delimited JSON on stdin, writes one JSON response per line on stdout.
Diagnostics go to stderr, so they can never corrupt the response stream.

## Why a separate process

`transcribe-cpp` (Handy's speech backend) and `llama.cpp` each vendor a complete
ggml tree. They cannot share a binary:

- Their `ggml_*` symbols collide at link time.
- Worse in practice, merely adding `llama-cpp-2` to the `handy` crate changes
  `transcribe-cpp-sys`'s `OUT_DIR`, which changes the `%LOCALAPPDATA%\tcs\<hash>`
  junction it uses to dodge Windows' path limit, forcing a from-scratch rebuild
  that then fails. The breakage is in the _native build_, upstream of linking.

Running inference in its own process also means an out-of-memory kill costs the
user a fallback to raw dictation rather than taking down the app mid-sentence,
and the process can be terminated to reclaim RAM when the feature is idle.

## Protocol

```jsonc
// requests (one per line on stdin)
{"cmd":"ping","id":1}
{"cmd":"load","id":2,"path":"C:/models/model.gguf","threads":4,"ctx":2048,"gpu_layers":null}
{"cmd":"generate","id":3,"system":"...","user":"...","max_tokens":200,"no_think":true}
{"cmd":"unload","id":4}
{"cmd":"shutdown","id":5}

// responses (one per line on stdout)
{"id":3,"ok":true,"text":"Send it to Jane.","elapsed_ms":132}
{"id":3,"ok":false,"error":"no model is loaded"}
```

`gpu_layers` omitted means "offload everything if this build has a GPU backend";
`0` forces CPU. A failed GPU load falls back to CPU automatically, so one binary
serves both a GPU machine and a low-end one.

`no_think` appends the soft switch that reasoning models (Qwen3 and relatives)
understand as "answer directly". Without it such a model spends its whole token
budget deliberating and never emits an answer — measured at ~4 s of pure
thinking versus ~1 s for the same edit with it suppressed.

## Building

```bash
cargo build --release                                  # Vulkan (default)
cargo build --release --no-default-features            # CPU only
cargo build --release --no-default-features -F metal   # macOS
cargo build --release --no-default-features -F cuda    # NVIDIA-only
```

### Windows, with the Vulkan backend

The GPU build needs three things beyond the usual toolchain. Each of these was a
hard build failure, not a warning:

1. **`LIBCLANG_PATH`** pointing at an LLVM `bin` directory — `bindgen` needs
   `libclang.dll`.
2. **The Ninja generator.** The Visual Studio generator runs concurrent `CL.EXE`
   processes that contend over the same scratch `.pdb` inside the nested
   `vulkan-shaders-gen` sub-build, failing with
   `C1041: cannot open program database`.
3. **A short `CARGO_TARGET_DIR`.** The `vulkan-shaders-gen` ExternalProject
   nests deeply enough to overflow `CMAKE_OBJECT_PATH_MAX` (260) from a normal
   crate path. This is the same limit `transcribe-cpp-sys` works around with its
   `%LOCALAPPDATA%\tcs` junction; `llama-cpp-sys-2` has no such workaround.

```powershell
$ninja = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe"
$env:LIBCLANG_PATH   = "D:\dev\LLVM\bin"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "$ninja\ninja.exe"
$env:CARGO_TARGET_DIR = "D:\hl"   # must be short
$env:PATH = "$ninja;$env:PATH"
cargo build --release
```

A CPU-only build (`--no-default-features`) needs none of this beyond
`LIBCLANG_PATH`.

## Measured behaviour

Qwen3 0.6B Q4_K_M, Ryzen 5 5500 / RTX 3060, four dictation-length transcripts:

| Backend | Load                           | Median pass | Range      |
| ------- | ------------------------------ | ----------- | ---------- |
| Vulkan  | 1431 ms (incl. 877 ms warm-up) | 132 ms      | 113–249 ms |
| CPU     | 333 ms                         | 482 ms      | 360–832 ms |

Output was byte-identical between the two.

Vulkan compiles compute pipelines lazily, and *which* pipelines depends on the
shape of the batch. `Engine::load` therefore runs a throwaway generation after a
GPU load, sized like a real enhancement prompt, so the user's first dictation
does not pay that cost.

Sizing that warm-up correctly mattered more than having one. With a token-sized
dummy prompt the first genuine enhancement still took **13.1 s**; with a
prompt-shaped one it takes **137 ms**. A warm-up that does not resemble the real
request compiles the wrong pipelines and buys nothing.

The graphics driver also caches compiled pipelines on disk, and that cache is
cold for a newly installed binary. Measured on a freshly staged build: 4.4 s for
the very first enhancement, then 131 ms on every subsequent run. So the first
dictation after an install or update is slower than steady state, and the
warm-up shortens that without removing it.
