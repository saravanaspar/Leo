# Security Policy

## Supported version

Leo currently maintains the v1.0.0 code line. Security fixes should target the current `main` branch unless a maintainer documents another supported release line.

## Reporting a vulnerability

Please **do not open a public issue containing exploit details, secrets, or a working proof of concept**.

Preferred reporting path:

1. Open the repository's **Security** tab.
2. Use GitHub private vulnerability reporting / a private security advisory when available.
3. Include the affected commit/version, impact, reproduction conditions, and the smallest safe proof needed to validate the problem.

Repository security area:

<https://github.com/saravanaspar/Leo/security>

If private vulnerability reporting is not available, open a **minimal public issue requesting a private reporting channel** without including exploit details, secrets, or a proof of concept. A maintainer can then provide an appropriate private path.

## What to include

Useful reports contain:

- affected commit or version;
- operating system;
- Rust version;
- CPU/GPU backend involved;
- GPU model, driver, and CUDA version when relevant;
- exact reproduction steps;
- expected versus observed behavior;
- security impact;
- whether untrusted dataset/model/checkpoint/input content is required;
- whether the issue can corrupt artifacts, escape expected file paths, exhaust resources, or execute code.

## Scope notes

Leo reads local configuration, model/checkpoint, and dataset artifacts and dynamically loads NVIDIA CUDA/NVRTC libraries for GPU execution. Treat untrusted files, paths, shared caches, and runtime library environments as security-sensitive boundaries.

TinyStories and any other training corpus are external data sources. Their content and distribution terms are not security guarantees from Leo.

## Disclosure

Please allow maintainers time to reproduce and prepare a fix before public disclosure. Once a fix is available, the project may publish a security advisory describing affected versions, impact, and remediation.
