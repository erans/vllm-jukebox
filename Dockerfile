# syntax=docker/dockerfile:1.7-labs
# Multi-stage build for vllm-jukebox.
#
# Builder: alpine + Go cross-compiles a fully-static CGO-disabled binary using
# BuildKit's automatic platform args (linux/amd64 + linux/arm64). Runtime:
# distroless static-debian12 (nonroot UID 65532) — minimal surface, no shell,
# no package manager. Jukebox itself doesn't need CUDA — it shells out to
# Docker/`vllm`/`llama-server` on the host; this image is the orchestrator,
# not the inference runtime.

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

# Runtime: distroless static. Nonroot user is UID 65532.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /out/jukebox /jukebox

# Default server port from configs/example.yaml. Purely informational —
# real port comes from the YAML config passed via -config.
EXPOSE 8080

USER nonroot:nonroot

ENTRYPOINT ["/jukebox"]
CMD []

LABEL org.opencontainers.image.source="https://github.com/intarweb/vllm-jukebox"
LABEL org.opencontainers.image.description="Server that multiplexes multiple LLM models through vLLM/llama.cpp backends with automatic model swapping, multi-GPU scheduling, and graceful request draining"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="vllm-jukebox"
