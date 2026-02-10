# Distributed Llama – Agent Handbook
Use this file as the field manual for any autonomous agent working inside `distributed-llama`. It captures how to build, test, format, and reason about the code so you can stay aligned with maintainers on the first try.

## Repository Orientation
- Core inference engine lives in `src/`, with neural-network primitives under `src/nn/` and application glue in `src/app.cpp`, `src/dllama.cpp`, and `src/dllama-api.cpp`.
- CLI helpers and orchestrators are plain C++ frontends; `launch.py` automates model download plus scripted launches.
- Conversion utilities reside in `converter/` (pure Python) and emit `.m` model blobs plus tokenizer tables.
- Docs for deployment and hardware are under `docs/` (see HOW_TO_RUN_* files for per-platform expectations).
- Make targets drop binaries in the repo root; test artifacts such as `nn-cpu-test` coexist next to `dllama`.
- No `.cursor/rules` or `.cursorrules` files exist; there is also no `.github/copilot-instructions.md`, so this handbook is the canonical policy source.

## Toolchain & Dependencies
- Minimum compiler is GCC/Clang with C++11 support; Makefile enforces `-std=c++11`, `-Werror`, and CPU-tuned `-march=native -mtune=native` unless cross-compiling.
- Linux/macOS builds depend on `build-essential` (CI installs via `sudo apt-get install build-essential`).
- Windows CI installs `make` through Chocolatey and otherwise relies on MSYS/MinGW toolchains; sockets require `ws2_32` (Makefile links automatically).
- Vulkan backend is optional; define `DLLAMA_VULKAN=1` and provide `glslc` plus Vulkan SDK headers/libs (`VK_SDK_PATH` on Windows or system `libvulkan` on Unix).
- Python utilities assume Python 3.10+ with `requests`-compatible stdlib (only `urllib.request` is used) and `multiprocessing` for CPU count detection.
- Network builds do not pull any package manager dependencies automatically; ensure `glslc`, `python`, and `make` exist before invoking automation scripts.

## Build Commands
- Always run from the repo root (`distributed-llama`).
- `make dllama` builds the main CLI capable of `inference`, `chat`, `worker`, and `dllama-api` modes.
- `make dllama-api` emits the API server binary (same deps as `dllama`, just different entry point).
- `make nn-cpu-test`, `make nn-cpu-ops-test`, `make tokenizer-test`, and `make nn-vulkan-test` (if Vulkan enabled) build the standalone regression binaries used by CI.
- Use `make clean` to clear `*.o`, binaries, and perf helpers before fresh builds.
- Build with sanitizers by exporting `DEBUG=1` (injects `-g -fsanitize=address`); production builds leave `DEBUG` unset for `-O3`.
- Enable additional warnings for variable length arrays via `WVLA=1 make ...` (adds `-Wvla-extension`).

## Tests & Verification
- CI workflow `.github/workflows/main.yml` compiles `dllama` plus the three core test binaries on Ubuntu amd64, Ubuntu arm64, and Windows; align local scripts with the same order.
- Default regression suite is plain executables, not a framework: run them directly after building to exercise deterministic math paths.
- Example full sweep: `make nn-cpu-test nn-cpu-ops-test tokenizer-test && ./nn-cpu-test && ./nn-cpu-ops-test && ./tokenizer-test`.
- `nn-cpu-test` validates RMS norm and executor plumbing; `nn-cpu-ops-test` covers fused ops; `tokenizer-test` ensures vocabulary and serialization invariants.
- For Vulkan, also invoke `DLLAMA_VULKAN=1 make nn-vulkan-test && ./nn-vulkan-test` after ensuring GPU drivers and shaders compile (`glslc` turns `.comp` into `.spv`).
- There is no Python unit-test suite; manually run converter scripts with sample checkpoints when modifying them.

## Running a Single Test Binary
- Because every test lives in its own executable, running a “single test” means building and executing just that target.
- Pattern: `make <target> && ./<target>`, e.g. `make nn-cpu-test && ./nn-cpu-test` or `make tokenizer-test && ./tokenizer-test`.
- If you only need to re-run an executable after source edits, you can skip `make` when the binary is fresh, but prefer rebuilding to respect `-Werror` guarantees.
- When profiling failures, you can pass environment variables such as `ASAN_OPTIONS` for sanitizer builds; there is no argument parsing inside the test binaries.

## Python Utilities & Launch Flow
- `launch.py` downloads predefined models + tokenizers from Hugging Face, writes them into `models/<model-name>/`, optionally creates `run_<model>.sh`, and executes `./dllama chat` or `./dllama inference`.
- Invoke it as `python launch.py <model-id> [-skip-run] [-skip-script] [-y]`; see the `MODELS` dict for valid identifiers like `llama3_1_8b_instruct_q40`.
- The script recompiles `dllama` on demand if binary missing (`os.system('make dllama')`). Keep this behavior in mind when editing `launch.py` to avoid recursive builds.
- Converter scripts (e.g., `converter/convert-llama.py`, `convert-hf.py`, tokenizer writers) expect `huggingface_hub` checkpoints to already be synced locally; they emit `.m` weight blobs and `.t` tokenizers consumed by `dllama`.
- Python code avoids third-party deps; stick to the standard library so users can run scripts on Windows/macOS without pip installs.

## GPU / Vulkan Builds
- Guard GPU code with `#if defined(DLLAMA_VULKAN)`; touching GPU paths without that macro should not break CPU builds.
- Shader sources live in `src/nn/vulkan/*.comp`; Makefile auto-generates `.spv` binaries when `DLLAMA_VULKAN=1`.
- Expect `NnVulkanDevice` usage in `src/nn/nn-vulkan.cpp`; new GPU features must still degrade gracefully when the macro is undefined.
- Document new GPU requirements inside `docs/HOW_TO_RUN_GPU.md` alongside build steps so operators can replicate them.

## General Code Style
- Target portability across Linux, macOS, Windows, ARM64, and x86_64; avoid platform APIs unless wrapped behind portability layers (see `nn-network.*`).
- Keep code warning-clean under `-Werror`; prefer resolving the warning rather than silencing it.
- Stick with spaces (4-wide) as seen across `src/*.cpp`.
- Prefer small, single-purpose functions; `AppCliArgs::parse` is the reference for CLI parsing with early validation.
- Use RAII whenever possible (`std::unique_ptr` with custom deleters for configs, `std::vector` buffers, scoped file handles).

## C++ Specific Guidelines
- Includes: standard headers first (`<cassert>`), then STL, then project headers using quotes (e.g., `#include "nn/nn-network.hpp"`).
- Files typically expose helper `static` functions at the top for option parsing or string conversions.
- Favor enums and `switch`-like `if` chains for translating metadata; throw `std::runtime_error` with concrete context when unsupported values appear.
- Use plain old data structs (`LlmHeader`, `AppCliArgs`) initialized explicitly; zero memory via `std::memset` before populating.
- Avoid heap allocations inside hot loops—allocate buffers once (see `NnNetExecution`) and reuse them.
- When manual memory is unavoidable (e.g., worker host arrays), allocate with `new[]` and free inside destructors.
- Check invariants with `assert` for programmer errors but use exceptions for user input/reportable failures.
- Logging is done via `printf` to keep dependencies minimal; use emojis sparingly but consistently (already present, keep style).
- Threading: rely on `NnNetExecution` batch sizing and `NnExecutor` for synchronization; do not spawn raw threads per feature unless architecture requires it.

## Python Style Guidelines
- Scripts stick to the standard library, camel_case for locals, and descriptive names (see `downloadFile`, `writeRunFile`).
- Always guard CLI entry points with `if __name__ == '__main__':` and ensure `os.chdir` is used before interacting with relative paths.
- Prefer helper dictionaries (like `MODELS`) plus small helper functions instead of large monoliths.
- Provide friendly terminal output (emojis already in use) but never prompt without supporting `-y` or similar automation bypass flags.
- Keep files ASCII-only; avoid f-string emojis unless consistent with existing output.

## Imports & Includes
- Keep `<...>` system headers grouped and ordered alphabetically when practical, followed by blank line and project headers.
- Guard optional dependencies with macros near includes (example: `#if defined(DLLAMA_VULKAN)` before including Vulkan headers).
- Do not rely on transitive includes; explicitly include the headers you use to keep `-Werror` builds deterministic.
- For Python, place stdlib imports at the top, alphabetized; avoid relative imports in converter scripts by keeping them self-contained.

## Naming Conventions
- Types use `CamelCase` with `Nn`/`Llm` prefixes (e.g., `NnExecutorDevice`).
- Functions and methods use `CamelCase` when mirroring existing naming (`setBatchSize`, `loadLlmNetWeight`); free functions default to `snake_case` only in tests or helper scopes.
- Constants and macros are `UPPER_SNAKE` (`DIM`, `N_BATCHES`).
- CLI flags mirror the long-option names defined in `AppCliArgs::parse`; avoid introducing new defaults without documenting them in `README.md` and `docs/HOW_TO_RUN_*`.

## Error Handling & Validation
- Fail fast with descriptive `std::runtime_error` messages that include offending values, as seen in `parseFloatType` and `loadLlmHeader`.
- Validate all user-supplied numbers (`nThreads`, `gpu segments`, worker addresses) before use and throw when invalid.
- Networking failures should raise/print via `NnTransferSocketException` or `NnExecutorException`; wrap loops with `try/catch` to keep workers alive.
- Python scripts prompt before overwriting files and retry downloads up to 8 times; preserve these UX guarantees in new flows.

## Logging & UX Notes
- Use concise `printf` statements with emoji prefixes (🛑, 🚁, 📊) to stay consistent with current CLI tone.
- Keep user-facing text English-only; docs and CLI outputs should remain friendly but direct.
- Provide actionable follow-ups when errors occur (e.g., mention expected format for GPU segments or worker addresses).

## Networking & Concurrency
- Worker loops (`runWorkerApp`) must remain resilient: throttle busy polls, use `network->setTurbo(true/false)` to toggle blocking vs non-blocking sockets, and always print when states change.
- When modifying message formats, update `NnRootConfigWriter`/`NnWorkerConfigReader` together and keep compatibility guards.
- Synchronization uses `NnNetworkNodeSynchronizer` and `NnFakeNodeSynchronizer`; new code should integrate with these abstractions rather than introducing bespoke barriers.

## Documentation & Comments
- This repo avoids redundant comments; only annotate when behavior is non-obvious (e.g., why Qwen3 paths have extra RMS normalization passes).
- Keep docs in `docs/*.md` updated when CLI flags, hardware requirements, or supported models change; cross-reference README sections for discoverability.
- ASCII art and emoji tables are encouraged if they clarify topology or commands.

## Continuous Integration Expectations
- Any change that touches neural math, tokenizer logic, or CLI parsing should run `make dllama nn-cpu-test nn-cpu-ops-test tokenizer-test` locally before submission.
- CI currently runs only on `main` and `feat/nn`; if you add workflows, keep them minimal to respect arm64 runner availability.
- Do not rely on `gh` or git metadata inside build scripts; CI clones shallow and expects standalone Make targets.

## Contribution Flow
- Keep diffs focused: avoid formatting churn because `-Werror` enforces consistent style already.
- Multi-node behavior must default to backward-compatible settings (`netTurbo` on root, toggled on workers when throughput increases).
- Update `README.md` or relevant HOW_TO_* doc whenever you add CLI switches, new models, or environmental requirements.

## Final Notes
- There are no Cursor or Copilot override files, so treat this handbook plus the existing README/docs as the source of truth for style and commands.
- When in doubt, mirror the patterns demonstrated in `src/app.cpp`, `src/llm.cpp`, and `launch.py`; they represent the maintainers' preferred idioms.

## Model Artifacts & Storage
- Keep converted `.m` weight files and `.t` tokenizers out of source control; scripts assume they live in `models/<name>/` beside generated `run_<name>.sh` files.
- When adding new presets to `launch.py`, include both the weight URL list and tokenizer URL, plus correct quantization pair (weight float type + buffer float type).
- Update documentation tables (README + docs) when you introduce new SKUs so that download instructions remain synchronized.
- Preserve backward compatibility of file headers; `loadLlmHeader` rejects unknown magic IDs, so version bumps must be communicated before merging.

## Benchmarking & Profiling
- `dllama inference` accepts `--steps`, `--prompt`, `--nthreads`, and `--benchmark` toggles; prefer using these rather than editing code for perf studies.
- Use built-in network telemetry: `network->enablePerformanceMonitoring(true)` already logs per-link stats and bottlenecks; extend those functions if you need extra counters.
- Flamegraphs and perf traces should be generated via external tooling (e.g., `perf record` or `Instruments`), keeping repo scripts agnostic.
- For sanitizer runs, export `DEBUG=1 ASAN_OPTIONS=abort_on_error=1` before invoking `make` to match CI expectations for reproducible crashes.

## CLI & API Behavior
- `dllama chat` and `dllama inference` share `runInferenceApp`; new CLI switches must be plumbed through `AppCliArgs`, validated, and surfaced in README tables.
- CLI arguments follow the `--flag value` pattern; avoid introducing equals-sign forms to keep parsing simple.
- `dllama worker` loops forever, accepting root connections sequentially; design new worker features around this hot loop without blocking handshake steps.
- `dllama-api` exposes network endpoints (see `src/dllama-api.cpp`); keep its argument names consistent with the CLI to minimize operator confusion.

## Reference Docs & Links
- General usage: `README.md` (project overview) plus `docs/HOW_TO_RUN_LINUX_MACOS_WIN.md`, `docs/HOW_TO_RUN_RASPBERRYPI.md`, and `docs/HOW_TO_RUN_GPU.md` for platform details.
- Conversion: `docs/HOW_TO_CONVERT_HF_MODEL.md` describes expected inputs/outputs for converter scripts; update it when file layouts change.
- Architecture: see `docs/MOE_GUIDE.md`, `docs/FEED_FORWARD_GUIDE.md`, and `docs/NODE_DISTRIBUTION_GUIDE.md` for conceptual explanations.
- Performance research lives in top-level `*_REPORT.md` files (e.g., `NETWORK_OPTIMIZATION_REPORT.md`); cite them when adjusting scheduling or parallelism heuristics.
