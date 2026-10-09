README
======

These models were moved out of the Rust monorepo on 2022-11-13. 
TTS is already accounted for here (under `tts/`), but we need to get W2L working again from this repository.

These are being removed to improve the build times of the Rust monorepo.

The idea is that these Docker images will build independently, move less frequently, and can be mounted into a 
k8s container as a volume or sidecar.

