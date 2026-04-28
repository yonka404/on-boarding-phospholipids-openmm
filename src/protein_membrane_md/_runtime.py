import ctypes
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_BOOTSTRAP_ATTEMPTED = False
_SKIP_BOOTSTRAP_ENV = "MEMBRANE_OPENMM_SKIP_BOOTSTRAP"
_HIP_LIBRARY_NAMES = ("libamdhip64.so.7", "libhiprtc.so.7")
_ROCM_ENV_VARS = ("ROCM_PATH", "HIP_PATH")


def bootstrap_openmm_runtime() -> None:
    global _BOOTSTRAP_ATTEMPTED

    if _BOOTSTRAP_ATTEMPTED:
        return
    _BOOTSTRAP_ATTEMPTED = True

    if os.environ.get(_SKIP_BOOTSTRAP_ENV) == "1":
        return
    if platform.system() != "Linux":
        return

    roots = discover_rocm_roots()
    if not roots:
        return

    failures: list[str] = []
    for root in roots:
        libraries = hip_runtime_libraries(root)
        if not libraries:
            failures.append(f"{root}: missing HIP runtime libraries")
            continue

        try:
            preload_shared_libraries(libraries)
        except OSError as exc:
            failures.append(f"{root}: {exc}")
            continue

        logger.info("Preloaded ROCm HIP runtime from %s", root)
        return

    logger.warning(
        "Detected ROCm installation(s) but could not preload HIP runtime: %s",
        "; ".join(failures),
    )


def discover_rocm_roots(
    environ: Mapping[str, str] | None = None,
    which=shutil.which,
    opt_root: Path = Path("/opt"),
) -> tuple[Path, ...]:
    environ = os.environ if environ is None else environ
    candidates: list[Path] = []

    for name in _ROCM_ENV_VARS:
        value = environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())

    hipconfig_path = which("hipconfig")
    if hipconfig_path:
        candidates.append(Path(hipconfig_path).expanduser().resolve().parent.parent)

    candidates.append(opt_root / "rocm")
    candidates.extend(_versioned_rocm_roots(opt_root))

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        roots.append(resolved)
        seen.add(resolved)

    return tuple(roots)


def hip_runtime_libraries(root: Path) -> tuple[Path, ...]:
    libraries = tuple((root / "lib" / name).resolve() for name in _HIP_LIBRARY_NAMES)
    if all(path.is_file() for path in libraries):
        return libraries
    return ()


def preload_shared_libraries(libraries: tuple[Path, ...]) -> None:
    for library in libraries:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _versioned_rocm_roots(opt_root: Path) -> tuple[Path, ...]:
    versioned = [path for path in opt_root.glob("rocm-*") if path.is_dir()]
    versioned.sort(key=_rocm_version_key, reverse=True)
    return tuple(versioned)


def _rocm_version_key(path: Path) -> tuple[int, ...]:
    parts = path.name.removeprefix("rocm-").split(".")
    numbers = [int(part) for part in parts if part.isdigit()]
    return tuple(numbers) or (0,)
