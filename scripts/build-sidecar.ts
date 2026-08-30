#!/usr/bin/env bun
/**
 * Build the `handy-llm` inference sidecar and stage it for Tauri bundling.
 *
 * Tauri's `externalBin` expects `src-tauri/binaries/<name>-<target-triple>`,
 * so this builds the crate and copies the result under that name.
 *
 * The sidecar is a separate crate with its own target directory (see
 * `src-tauri/crates/handy-llm/README.md` for why it cannot live in the app
 * crate), which means it is not built by `cargo build` on the app and needs
 * this step before `tauri build`.
 *
 * Usage:
 *   bun run scripts/build-sidecar.ts              # GPU build for the host
 *   bun run scripts/build-sidecar.ts --cpu        # CPU-only
 *   bun run scripts/build-sidecar.ts --target x   # cross-compile target
 */

import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const crateDir = join(repoRoot, "src-tauri", "crates", "handy-llm");
const binariesDir = join(repoRoot, "src-tauri", "binaries");

const args = process.argv.slice(2);
const cpuOnly = args.includes("--cpu");
const targetArg = args.indexOf("--target");
const explicitTarget = targetArg >= 0 ? args[targetArg + 1] : undefined;

/** Host target triple, as rustc reports it. */
function hostTriple(): string {
  const out = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
  const line = out.split("\n").find((l) => l.startsWith("host:"));
  if (!line)
    throw new Error("could not determine host target triple from rustc -vV");
  return line.slice("host:".length).trim();
}

const target = explicitTarget ?? hostTriple();
const isWindows = target.includes("windows");
const isMac = target.includes("apple");
const exeSuffix = isWindows ? ".exe" : "";

/**
 * Pick the GPU backend for the target.
 *
 * Vulkan covers AMD, Intel and NVIDIA and is already a Handy build dependency
 * on Windows and Linux; macOS uses Metal. `--cpu` opts out entirely, which is
 * the right choice when the build machine has no GPU SDK.
 */
function featureArgs(): string[] {
  if (cpuOnly) return ["--no-default-features"];
  if (isMac) return ["--no-default-features", "--features", "metal"];
  return []; // default feature is vulkan
}

/**
 * Environment workarounds for the Windows Vulkan build.
 *
 * Both are load-bearing and both fail confusingly when missing:
 * `vulkan-shaders-gen` is a nested cmake ExternalProject that overflows
 * `CMAKE_OBJECT_PATH_MAX` from a normal crate path, and the Visual Studio
 * generator's concurrent CL.EXE processes contend over one scratch .pdb.
 */
function buildEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (!isWindows || cpuOnly) return env;

  if (!env.CMAKE_GENERATOR) env.CMAKE_GENERATOR = "Ninja";
  // A short path, outside the deep crate directory.
  if (!env.CARGO_TARGET_DIR) env.CARGO_TARGET_DIR = "C:\\hl";

  // cmake resolves the generator via PATH, and winget installs ninja to a
  // location it does not add there. Point at it explicitly when we can find
  // it, so the default build works without the caller preparing anything.
  if (!env.CMAKE_MAKE_PROGRAM) {
    const ninja = findNinja();
    if (ninja) {
      env.CMAKE_MAKE_PROGRAM = ninja;
      env.PATH = `${dirname(ninja)};${env.PATH ?? ""}`;
    } else {
      console.warn(
        "warning: ninja was not found. The Visual Studio generator fails on " +
          "the nested vulkan-shaders-gen build; install it with " +
          "`winget install Ninja-build.Ninja` or pass --cpu.",
      );
    }
  }
  return env;
}

/** Locate `ninja.exe`, including the winget package directory. */
function findNinja(): string | undefined {
  try {
    const found = execFileSync("where", ["ninja"], { encoding: "utf8" })
      .split(/\r?\n/)
      .find((l) => l.trim().endsWith(".exe"));
    if (found) return found.trim();
  } catch {
    // `where` exits non-zero when nothing matches; fall through.
  }
  const local = process.env.LOCALAPPDATA;
  if (!local) return undefined;
  const wingetPath = join(
    local,
    "Microsoft",
    "WinGet",
    "Packages",
    "Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ninja.exe",
  );
  return existsSync(wingetPath) ? wingetPath : undefined;
}

const env = buildEnv();
const targetDir = env.CARGO_TARGET_DIR ?? join(crateDir, "target");

console.log(`building handy-llm for ${target}${cpuOnly ? " (CPU only)" : ""}`);
if (env.CARGO_TARGET_DIR) {
  console.log(`  target dir: ${env.CARGO_TARGET_DIR}`);
}

try {
  execFileSync(
    "cargo",
    ["build", "--release", "--target", target, ...featureArgs()],
    { cwd: crateDir, env, stdio: "inherit" },
  );
} catch {
  console.error(
    "\nhandy-llm build failed. On Windows with the Vulkan backend this usually " +
      "means LIBCLANG_PATH is unset, ninja is not on PATH, or CARGO_TARGET_DIR " +
      "is too long. See src-tauri/crates/handy-llm/README.md. " +
      "Pass --cpu to build without a GPU backend.",
  );
  process.exit(1);
}

const built = join(targetDir, target, "release", `handy-llm${exeSuffix}`);
if (!existsSync(built)) {
  console.error(`expected binary not found at ${built}`);
  process.exit(1);
}

mkdirSync(binariesDir, { recursive: true });
const staged = join(binariesDir, `handy-llm-${target}${exeSuffix}`);
rmSync(staged, { force: true });
copyFileSync(built, staged);

console.log(`staged ${staged}`);
