# syntax=docker/dockerfile:1.7-labs
# Multi-stage build for vllm-jukebox.
#
# Builder: alpine + Go cross-compiles a fully-static CGO-disabled binary using
# BuildKit's automatic platform args (linux/amd64 + linux/arm64). Runtime:
# nvidia/cuda:12.6.0-base-ubuntu24.04 — ships nvidia-smi + libnvidia-ml so
# scheduler-mode admission control can query GPU VRAM without depending on
# the host's nvidia container-toolkit injection (CDI / nvidia-runtime as
# docker default). Adds ~170 MiB vs the prior distroless runtime — the
# tradeoff buys us reliable GPU introspection on hosts that don't auto-
# inject nvidia-smi. Jukebox itself doesn't need CUDA libs at runtime;
# it shells out to Docker/`vllm`/`llama-server` for actual inference.

FROM --platform=${BUILDPLATFORM} golang:1.25-alpine AS builder

ARG TARGETOS
ARG TARGETARCH

RUN apk add --no-cache ca-certificates git

WORKDIR /src

# Pre-download deps as a cached layer — re-runs only when go.mod/go.sum change.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags='-s -w' -o /out/jukebox ./cmd/jukebox

# Runtime: nvidia/cuda base. Ships nvidia-smi + libnvidia-ml. NVIDIA's
# 12.6.0-base-ubuntu24.04 tag publishes both linux/amd64 and linux/arm64
# variants (sbsa). We create our own non-root user (UID 65532, matching
# the prior distroless nonroot UID) since the cuda base ships only root.
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

# docker CLI for shell-out to host docker.sock — jukebox uses `docker stop`,
# `docker start`, `docker inspect` for evict_action: stop / wake-from-Stopped
# / redeploy-member flows (see internal/jukebox/sleep.go + redeploy.go).
# Distroless predecessor never had docker CLI either, which silently broke
# Phase 4 cold-load when the image switched to overnight-stage-j-* — the
# error 'exec: "docker": executable file not found in $PATH' surfaced at
# the first wake-from-Stopped request. docker.io package installs just
# the CLI binary; the dockerd daemon doesn't auto-start (no systemd in
# container), so this is CLI-only as desired.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
 && rm -rf /var/lib/apt/lists/*

# Non-root jukebox user, UID/GID 65532 — matches the distroless `nonroot`
# UID the prior image used, so any host bind-mount or NFS ACL keyed on
# 65532 stays correct after the base swap. nologin shell, no home write,
# no group additions.
RUN groupadd --system --gid 65532 jukebox \
 && useradd  --system --uid 65532 --gid 65532 \
             --no-create-home --home-dir /nonexistent \
             --shell /usr/sbin/nologin jukebox

COPY --from=builder /out/jukebox /jukebox

# Default server port from configs/example.yaml. Purely informational —
# real port comes from the YAML config passed via -config.
EXPOSE 8080

USER jukebox:jukebox

# In-binary HEALTHCHECK — the nvidia/cuda base does have a shell, but
# keeping the probe in the jukebox binary keeps the contract identical
# across base-image swaps and avoids depending on any package the cuda
# image happens to ship today. `jukebox health` probes 127.0.0.1:<port>/health
# and exits 0 / 1. Port resolution: -port flag > JUKEBOX_HEALTHCHECK_PORT env >
# JUKEBOX_CONFIG yaml's server.port > 8080 default.
# start-period gives the daemon time to load its default model (cold-load
# can take several minutes); retries soak transient swap/eviction blips.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["/jukebox", "health"]

ENTRYPOINT ["/jukebox"]
CMD []

LABEL org.opencontainers.image.source="https://github.com/intarweb/vllm-jukebox"
LABEL org.opencontainers.image.description="Server that multiplexes multiple LLM models through vLLM/llama.cpp backends with automatic model swapping, multi-GPU scheduling, and graceful request draining"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="vllm-jukebox"
