## Runtime Notes

- On Linux AMD hosts with ROCm installed, the package preloads the HIP runtime before OpenMM import so `HIP` can be used when available.
- On NVIDIA hosts, normal OpenMM `CUDA` selection remains unchanged.
- On CPU-only hosts, the existing fallback order still reaches `CPU`.
- Use `OPENMM_DEVICE_INDEX` to pin a specific GPU when multiple devices are visible.
