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
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
} from "node:fs";
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

  // bindgen needs libclang on every platform, and a terminal opened before
  // LLVM was installed will not have it. Finding it here means the build does
  // not depend on which shell the caller happens to be in.
  if (!env.LIBCLANG_PATH) {
    const libclang = findLibclang();
    if (libclang) env.LIBCLANG_PATH = libclang;
  }

  if (!isWindows || cpuOnly) return env;

  // Same reasoning as libclang: the SDK is usually installed, just not visible
  // to a terminal that was already open when it was.
  if (!env.VULKAN_SDK) {
    const sdk = findVulkanSdk();
    if (sdk) env.VULKAN_SDK = sdk;
  }

  if (!env.CMAKE_GENERATOR) env.CMAKE_GENERATOR = "Ninja";
  // A short path, because the nested vulkan-shaders-gen build overflows
  // CMAKE_OBJECT_PATH_MAX from a normal crate path. Kept on the same drive as
  // the checkout rather than hardcoding C:, which may be short on space and is
  // not necessarily where the developer keeps build output.
  if (!env.CARGO_TARGET_DIR) {
    // Deliberately terse. Every character here is charged against
    // CMAKE_OBJECT_PATH_MAX by the deeply nested shader build, and a
    // descriptive name is enough on its own to push it over the limit.
    env.CARGO_TARGET_DIR = repoRoot.slice(0, 2) + "/hl";
  }

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

/**
 * Locate a directory containing libclang, which bindgen needs.
 *
 * Checks the usual install locations rather than only the environment, so the
 * build works in a terminal that predates the LLVM install.
 */
function findLibclang(): string | undefined {
  const lib = isWindows
    ? "libclang.dll"
    : isMac
      ? "libclang.dylib"
      : "libclang.so";
  const candidates = isWindows
    ? [
        "C:/Program Files/LLVM/bin",
        "D:/dev/LLVM/bin",
        join(process.env.LOCALAPPDATA ?? "", "Programs", "LLVM", "bin"),
      ]
    : [
        "/usr/lib/llvm/lib",
        "/usr/lib",
        "/usr/local/lib",
        "/opt/homebrew/opt/llvm/lib",
      ];

  for (const dir of candidates) {
    if (dir && existsSync(join(dir, lib))) return dir;
  }
  // The user-level environment may know even when this process does not,
  // which is exactly the case in a terminal opened before the install.
  if (isWindows) {
    try {
      const out = execFileSync(
        "powershell",
        [
          "-NoProfile",
          "-Command",
          "[Environment]::GetEnvironmentVariable('LIBCLANG_PATH','User')",
        ],
        { encoding: "utf8" },
      ).trim();
      if (out && existsSync(join(out, lib))) return out;
    } catch {
      // Not fatal; cargo will report the missing library itself.
    }
  }
  return undefined;
}

/**
 * Locate an installed Vulkan SDK.
 *
 * Prefers the user-level environment variable the installer sets, then falls
 * back to scanning the default install root for the newest version.
 */
function findVulkanSdk(): string | undefined {
  try {
    // Ask for a single value so the reply needs no parsing.
    const out = execFileSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        "$u = [Environment]::GetEnvironmentVariable('VULKAN_SDK','User'); $m = [Environment]::GetEnvironmentVariable('VULKAN_SDK','Machine'); if ($u) { $u } elseif ($m) { $m }",
      ],
      { encoding: "utf8" },
    ).trim();
    if (out && existsSync(join(out, "Include", "vulkan"))) return out;
  } catch {
    // Fall through to the directory scan.
  }

  const roots = ["C:/VulkanSDK", "D:/dev/VulkanSDK", "D:/VulkanSDK"];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    const versions = readdirSync(root)
      .filter((name) => existsSync(join(root, name, "Include", "vulkan")))
      .sort()
      .reverse();
    if (versions[0]) return join(root, versions[0]);
  }
  return undefined;
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

/**
 * Cross-compilation arguments.
 *
 * Passing `--target` for the host would nest the output under an extra
 * directory named for the triple, and on Windows those 24 characters are
 * enough to overflow the nested shader build's path budget.
 */
function targetArgs(): string[] {
  return explicitTarget ? ["--target", explicitTarget] : [];
}

/** Where cargo puts the binary, mirroring `targetArgs`. */
function builtSubdir(): string[] {
  return explicitTarget ? [explicitTarget, "release"] : ["release"];
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
    ["build", "--release", ...targetArgs(), ...featureArgs()],
    { cwd: crateDir, env, stdio: "inherit" },
  );
} catch {
  // Name what is actually missing. The underlying errors point away from their
  // cause — a missing libclang surfaces as a bindgen panic, and a too-long
  // target path as a linker error about missing shader symbols.
  const problems: string[] = [];
  if (!env.LIBCLANG_PATH) {
    problems.push(
      "  - libclang was not found. Install LLVM (`winget install LLVM.LLVM`) " +
        "or set LIBCLANG_PATH to a directory containing libclang.",
    );
  }
  if (isWindows && !cpuOnly && !env.CMAKE_MAKE_PROGRAM) {
    problems.push(
      "  - ninja was not found. Install it (`winget install Ninja-build.Ninja`); " +
        "the Visual Studio generator fails on the nested vulkan-shaders-gen build.",
    );
  }
  if (isWindows && !cpuOnly && !env.VULKAN_SDK) {
    problems.push(
      "  - VULKAN_SDK is not set. Install the LunarG SDK " +
        "(`winget install KhronosGroup.VulkanSDK`) and open a new terminal.",
    );
  }

  console.error("\nhandy-llm build failed.");
  if (problems.length > 0) {
    console.error("Likely cause:\n" + problems.join("\n"));
  } else {
    console.error(
      "The prerequisites this script checks are all present, so the cargo " +
        "output above is the real error.",
    );
  }
  console.error(
    "\nSee src-tauri/crates/handy-llm/README.md. " +
      "`bun run build:sidecar:cpu` builds without a GPU backend and needs only libclang.",
  );
  process.exit(1);
}

const built = join(targetDir, ...builtSubdir(), `handy-llm${exeSuffix}`);
if (!existsSync(built)) {
  console.error(`expected binary not found at ${built}`);
  process.exit(1);
}

mkdirSync(binariesDir, { recursive: true });
const staged = join(binariesDir, `handy-llm-${target}${exeSuffix}`);
rmSync(staged, { force: true });
copyFileSync(built, staged);

console.log(`staged ${staged}`);
