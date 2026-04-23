# OpenMM Runtime Bootstrap Design

## Goal

Make this repo run locally on an AMD/ROCm Linux machine without requiring a manual `LD_LIBRARY_PATH` export, while preserving the existing OpenMM fallback order:

`HIP -> CUDA -> OpenCL -> CPU -> Reference`

The same repo must continue to run correctly on external servers with NVIDIA GPUs or CPU-only hosts.

## Problem Statement

On the local AMD workstation, the ROCm runtime is installed and the GPU is visible to ROCm tooling, but OpenMM cannot load the HIP plugin because the required ROCm shared libraries are not visible to the process at plugin-load time.

That causes OpenMM to skip `HIP`, skip `CUDA`, skip `OpenCL`, and fall through to `CPU`, even though the AMD GPU is otherwise usable.

This is a repo usability problem, not a platform-selection problem. The current platform-selection logic is already correct for the desired fallback behavior.

## Constraints

- The fix must be repo-local, not a system-wide linker or driver change.
- The fix must not break NVIDIA or CPU-only hosts.
- The existing fallback order and non-fatal fallback behavior must remain unchanged.
- The fix must work when the repo is executed through its normal Python entrypoints.
- The fix should stay narrow in scope and avoid inventing new launch wrappers unless necessary.

## Existing Behavior

The repo already prefers GPU platforms first in `src/membrane_openmm/simulation.py`, trying:

1. `HIP`
2. `CUDA`
3. `OpenCL`
4. `CPU`
5. `Reference`

If a platform is unavailable or fails to initialize, the repo logs that condition and tries the next option.

This behavior should remain intact.

## Chosen Approach

Add a small host-aware OpenMM runtime bootstrap inside the `membrane_openmm` package. The bootstrap will run before this package's modules import OpenMM, discover usable ROCm library roots on Linux, and preload the required HIP runtime libraries with `ctypes.CDLL(..., RTLD_GLOBAL)` when they are available.

This is preferred over a shell wrapper because:

- it works from the repo's normal Python entrypoints
- it stays local to this project
- it does not depend on users remembering a special launcher
- it preserves the current fallback logic instead of replacing it

This is preferred over a one-time process re-exec because the preload path is simpler and was validated to work in-process.

## Architecture

### New bootstrap module

Introduce a focused runtime module under `src/membrane_openmm/` responsible only for runtime library preparation before OpenMM import.

Responsibilities:

- detect whether the current host is a Linux system where ROCm preload is relevant
- discover candidate ROCm installation roots
- identify required HIP runtime shared libraries under a candidate root
- preload those libraries with `ctypes.CDLL(..., RTLD_GLOBAL)`
- emit concise diagnostic logging about success or failure

Non-responsibilities:

- selecting an OpenMM platform
- changing fallback behavior
- changing system linker configuration
- probing CUDA or OpenCL installations

### Initialization point

Call the bootstrap from `src/membrane_openmm/__init__.py` before any package submodules import OpenMM.

This is required because several package modules import OpenMM at module import time:

- `src/membrane_openmm/charmm_gui.py`
- `src/membrane_openmm/workflow.py`
- `src/membrane_openmm/simulation.py`

By initializing from package import, the normal repo entrypoints gain the behavior automatically.

## Runtime Behavior

### Host-aware operation

The bootstrap should be conservative:

- On non-Linux hosts: no-op.
- On Linux hosts with no detectable ROCm installation: no-op.
- On Linux hosts with ROCm present but unusable: log once and allow normal OpenMM fallback behavior.
- On Linux hosts with ROCm present and preloadable: preload HIP dependencies before OpenMM import.

This gives the desired cross-host behavior:

- AMD host: `HIP` becomes available and is tried first.
- NVIDIA host: ROCm preload usually becomes a no-op, `HIP` stays unavailable, `CUDA` is tried next.
- CPU-only host: `HIP`, `CUDA`, and likely `OpenCL` remain unavailable, and the repo falls through to `CPU`.

### Candidate root discovery order

On Linux, search ROCm roots in this order:

1. `ROCM_PATH` environment variable
2. `HIP_PATH` environment variable
3. resolved install root from `hipconfig` if `hipconfig` exists in `PATH`
4. `/opt/rocm`
5. versioned `/opt/rocm-*` directories, in descending version order when sortable

The search should deduplicate equivalent paths and ignore non-existent roots.

### Required libraries

The preload logic should verify the required shared libraries exist under the candidate root's `lib/` directory before attempting preload.

The minimum HIP chain that matters for this repo is:

- `libhsa-runtime64.so.1`
- `librocprofiler-register.so.0`
- `libamdhip64.so.7`
- `libhiprtc.so.7`

Preload these with `RTLD_GLOBAL` so that OpenMM's HIP plugin can resolve its dependencies when OpenMM loads plugins.

If a candidate root is incomplete or preload fails, log the failure and continue to the next candidate root.

### OpenMM behavior after bootstrap

The bootstrap must not change:

- `OPENMM_PLATFORM`
- `OPENMM_DEVICE_INDEX`
- `OPENMM_OPENCL_PLATFORM_INDEX`
- the platform fallback order in `simulation.py`

Once preload has succeeded, the existing OpenMM initialization path should behave as before, except that `HIP` is now allowed to register and initialize.

## Logging

Logging should be concise and actionable.

Recommended behavior:

- On successful ROCm preload: one `INFO` log naming the selected ROCm root.
- On detectable ROCm with preload failure: one `WARNING` log summarizing the failing root and reason.
- On hosts where no ROCm installation is found: no noisy warning. This is an expected condition on NVIDIA and CPU-only servers.

The existing per-platform logs in `simulation.py` should remain the primary source for why OpenMM ultimately selected `HIP`, `CUDA`, `OpenCL`, `CPU`, or `Reference`.

## Device Selection

No change is needed to the existing `OPENMM_DEVICE_INDEX` handling.

On the local AMD workstation, OpenMM can still be pinned explicitly with:

- `OPENMM_DEVICE_INDEX=0` for the discrete RX 6700 XT
- `OPENMM_DEVICE_INDEX=1` for the integrated Radeon GPU

This remains an operator choice, not automatic policy in the bootstrap.

## Testing Strategy

### Unit tests

Add unit tests for the bootstrap helper with heavy use of mocking so the tests do not require real GPU hardware.

Required test coverage:

1. non-Linux host -> no-op
2. no ROCm roots found -> no-op
3. candidate roots discovered in the intended priority order
4. required libraries present -> preload attempted in the required order
5. preload failure on one candidate -> next candidate is attempted
6. first successful candidate stops further probing

These tests should validate behavior and decision-making, not real ROCm execution.

### Regression test for fallback behavior

Add a focused regression test around the current non-fatal fallback path so the repo continues to:

- try platforms in the existing order
- skip unavailable platforms
- continue toward CPU fallback when higher-priority GPU platforms are unavailable

This test should use mocks around OpenMM platform lookup and initialization rather than real hardware.

### Manual verification

Manual verification for this design is still necessary because hardware plugin loading is environment-specific.

On the local AMD machine:

- verify OpenMM registers `HIP`
- verify the repo can still fall back correctly if `HIP` is unavailable or forced off

On a non-AMD host:

- verify the bootstrap is a no-op
- verify the repo still reaches `CUDA` or `CPU` through the current fallback path

## Documentation Changes

Update `README.md` with a short runtime note covering:

- the repo can self-bootstrap ROCm runtime visibility on Linux AMD hosts
- NVIDIA hosts continue to rely on normal CUDA availability
- CPU-only hosts continue to fall back automatically
- `OPENMM_DEVICE_INDEX` may be used to pin a specific GPU when multiple GPUs are visible

The README update should stay short and operational rather than becoming a full ROCm installation guide.

## Failure Handling

Failure to find or preload ROCm libraries must not terminate the program.

The correct failure model is:

1. bootstrap tries to help when appropriate
2. bootstrap logs a concise warning if it found ROCm but could not make it usable
3. existing OpenMM platform selection continues unchanged
4. repo falls through to the next platform according to the existing order

This matches the requested runtime behavior for mixed deployment targets.

## Non-Goals

- modifying system linker configuration
- changing `/opt/rocm` symlinks
- installing ROCm, CUDA, or OpenCL drivers
- making arbitrary third-party OpenMM scripts on the same machine inherit this behavior
- changing the repo's platform ranking or fallback semantics

## Risks

### Import-order sensitivity

If some future code imports OpenMM before importing `membrane_openmm`, the bootstrap cannot help that process. Keeping the initialization in `membrane_openmm/__init__.py` covers this repo's normal code paths but not arbitrary external scripts.

### ROCm version-specific filenames

The preload chain is based on the currently observed ROCm 7.2 runtime naming. The implementation should isolate library names in one place so future version-specific adjustments are straightforward.

### Overly noisy diagnostics

If bootstrap logs warnings on every non-AMD host, it will add noise in the server environments this repo targets. The implementation must distinguish "ROCm absent by design" from "ROCm present but misconfigured."

## Success Criteria

The design is successful when:

- this repo runs on the local AMD Linux workstation without needing a manual `LD_LIBRARY_PATH` export
- the existing fallback order remains unchanged
- the repo still runs on NVIDIA and CPU-only servers without ROCm-specific breakage
- failures in ROCm bootstrap remain non-fatal and fall through to the existing platform selection behavior
