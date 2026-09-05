# Releasing Handier

The fork inherits upstream Handy's release machinery, and several parts of it
point at **upstream's** accounts and infrastructure. Those have been repointed;
this file records what was changed and what still needs a human.

## Before the first release

- [x] **Updater signing keypair generated.** `plugins.updater.pubkey` in
      `src-tauri/tauri.conf.json` is the fork's own minisign key,
      `EF8FF290B74E73B4` — not upstream's `BAB72095206601F9`. Verify with:

      ```bash
      grep -o '"pubkey": "[^"]*"' src-tauri/tauri.conf.json |
        sed 's/.*": "//; s/"$//' | base64 -d
      ```

      The matching private key and its password are repository secrets
      (`TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`).
      Regenerate with `bun tauri signer generate -w ~/.tauri/handier.key`;
      never commit the private key.

      Publishing under upstream's key would be worse than having none: the
      installed app verifies updates against whatever key it shipped with, so a
      mismatch strands every user on the version they installed.

- [ ] **Make the model repositories public**, or the default editor cannot be
      downloaded by anyone but you. Both are currently private:
      [`handy-editor-lfm2.5-350m`](https://huggingface.co/MagicNoThief/handy-editor-lfm2.5-350m)
      and
      [`handy-dictation-editing`](https://huggingface.co/datasets/MagicNoThief/handy-dictation-editing).

- [ ] **Add the fine-tuned editor to the catalogue.** It is currently reachable
      only through **Your Own Model** (a local GGUF), because
      `src-tauri/src/enhance/models.json` has no entry for it and the Hugging
      Face repo is private. Once the repo is public, add an entry with
      `repo_id: MagicNoThief/handy-editor-lfm2.5-350m`,
      `filename: handy-editor-350m-Q4_K_M.gguf`, `prompt_style: "tuned"` and
      `license: lfm1.0`. Moving `default_editor` onto it is a separate decision:
      it is 7× smaller and 3× faster than the current default at the same suite
      score, but it is English-only and single-purpose, where Qwen3-4B is not.

      With a catalogue entry the RAM-based quant selection becomes possible too —
      `min_ram_mb` and `catalog::fitting()` already exist for it.

- [ ] **Confirm the dataset licence question.** `disfl_qa` declares CC-BY-4.0 but
      derives from SQuAD, which is CC-BY-SA-4.0. See
      [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Versions and tags

Handier continues upstream's version line rather than restarting it. The first
release is **`0.10.0`**, tagged **`v0.10.0`**.

Two alternatives were rejected:

- **Reset to `0.1.0`** — understates a codebase forked at 0.9.6, and would
  read to users as less mature than the thing it is built on.
- **`0.9.6-handier.1`** — semver sorts a prerelease _below_ its base version,
  so the updater would consider `0.9.6` newer and never offer the fork's own
  builds.

`0.10.0` sorts correctly, reads as "0.9.6 plus something", and cannot be
confused with an upstream release because the two now read different feeds.

Tags are `vMAJOR.MINOR.PATCH`, matching upstream's convention and what
`release.yml` expects — it reads the version from `tauri.conf.json`, so bump
all three of `tauri.conf.json`, `package.json` and `src-tauri/Cargo.toml`
together before dispatching it.

## Already changed for the fork

These were upstream's and would have misbehaved if shipped as-is:

| What                         | Was                                     | Now                             |
| ---------------------------- | --------------------------------------- | ------------------------------- |
| `bundle.identifier`          | `com.pais.handy`                        | `com.magicnothief.handier`      |
| `productName`                | `Handy`                                 | `Handier`                       |
| `updater.endpoints`          | `cjpais/Handy` releases                 | `MagicNoThief/Handier` releases |
| `bundle.windows.signCommand` | CJ Pais's Azure Trusted Signing account | removed                         |

The updater one mattered most: with upstream's endpoint **and** upstream's
identifier, a released fork would have found upstream's `latest.json`, judged it
an update, and installed plain Handy over itself — silently removing the
enhancement layer.

### Consequences of the identifier change

The app-data directory moves. Anyone upgrading from a build that used the old
identifier keeps their settings only if they copy the folder:

```powershell
# Windows
Copy-Item "$env:APPDATA\com.pais.handy\*" "$env:APPDATA\com.magicnothief.handier\" -Recurse
```

```bash
# macOS
cp -R ~/Library/Application\ Support/com.pais.handy/ ~/Library/Application\ Support/com.magicnothief.handier/
# Linux
cp -R ~/.config/com.pais.handy/ ~/.config/com.magicnothief.handier/
```

### Windows builds are unsigned

Removing `signCommand` means Windows SmartScreen will warn on first run. To sign,
add your own `bundle.windows.signCommand` or set `TAURI_WINDOWS_SIGNTOOL_PATH`
with a certificate. This is a cost/identity decision, not a technical blocker.

## What a successful build produces

Verified on Windows with a throwaway signing key:

```
src-tauri/target/release/bundle/
  msi/Handier_0.10.0_x64_en-US.msi        73 MB
  nsis/Handier_0.10.0_x64-setup.exe       28 MB
```

The MSI contains 35 files including `handy.exe`, **`handy-llm.exe`** (the
enhancement sidecar) and the ggml Vulkan/CPU runtime DLLs. If `handy-llm.exe` is
missing from a build, the enhancement layer will be absent at runtime and the app
will report the sidecar as unavailable — check that `bun run build:sidecar` ran
before `tauri build`.

## Build order

The sidecar is an `externalBin` and must exist **before** `tauri build`, or
bundling fails on a missing binary:

```bash
bun install
bun run build:sidecar        # or build:sidecar:cpu
bun run tauri build
```

On Windows the GPU sidecar build additionally needs `LIBCLANG_PATH`, the Ninja
generator and a short `CARGO_TARGET_DIR` — see
[`src-tauri/crates/handy-llm/README.md`](src-tauri/crates/handy-llm/README.md).

## Gates

All of these are enforced in CI (`.github/workflows/`) and pass locally:

```bash
bun run check:translations      # 23 locales, key parity
bun run check:model-languages
bun run lint
bun run format:check            # prettier + cargo fmt
cd src-tauri && cargo test      # 358 tests
bun run test:playwright
```

Not in CI, but worth running when the enhancement model changes — see
[`scripts/enhance-eval/README.md`](scripts/enhance-eval/README.md):

```bash
cd scripts/enhance-eval
python bench.py --model <gguf> --no-switch --label release       # 68-case suite
python eval_heldout.py --model <gguf> --eval <handy_eval.jsonl>  # 2,152 rows
```

A quantisation that passes the 68-case suite can still be broken; the held-out
set is what catches it.

## Cutting the release

`.github/workflows/release.yml` is `workflow_dispatch` and reads the version from
`src-tauri/tauri.conf.json`. It creates a draft release, builds per platform, and
attaches the artefacts. Publish the draft once the artefacts look right.
