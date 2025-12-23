.PHONY: build test clean smoke smoke-scheduler smoke-vllm fmt lint vet run help

# Binary name and paths
BINARY_NAME := jukebox
BINARY_DIR := bin
BINARY_PATH := $(BINARY_DIR)/$(BINARY_NAME)
CMD_PATH := ./cmd/jukebox

# Go commands
GO := go
GOFMT := gofmt
GOVET := $(GO) vet
GOTEST := $(GO) test
GOBUILD := $(GO) build

# Build flags
LDFLAGS := -s -w

## help: Show this help message
help:
	@echo "vLLM Jukebox - Available targets:"
	@echo ""
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed 's/^/ /'
	@echo ""

## build: Build the jukebox binary
build:
	@mkdir -p $(BINARY_DIR)
	$(GOBUILD) -ldflags "$(LDFLAGS)" -o $(BINARY_PATH) $(CMD_PATH)

## build-debug: Build with debug symbols
build-debug:
	@mkdir -p $(BINARY_DIR)
	$(GOBUILD) -o $(BINARY_PATH) $(CMD_PATH)

## test: Run all tests
test:
	$(GOTEST) -v ./...

## test-short: Run tests without verbose output
test-short:
	$(GOTEST) ./...

## test-cover: Run tests with coverage report
test-cover:
	$(GOTEST) -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

## clean: Remove build artifacts
clean:
	rm -rf $(BINARY_DIR)
	rm -f coverage.out coverage.html

## fmt: Format Go source files
fmt:
	$(GOFMT) -s -w .

## fmt-check: Check if code is formatted (for CI)
fmt-check:
	@test -z "$$($(GOFMT) -l .)" || (echo "Code is not formatted. Run 'make fmt'" && exit 1)

## vet: Run go vet
vet:
	$(GOVET) ./...

## lint: Run golangci-lint (requires golangci-lint installed)
lint:
	golangci-lint run ./...

## smoke: Run basic smoke test
smoke: build
	./scripts/smoke.sh

## smoke-scheduler: Run scheduler mode smoke test
smoke-scheduler: build
	./scripts/smoke_scheduler.sh

## smoke-vllm: Run real vLLM integration test (requires GPU)
smoke-vllm: build
	RUN_VLLM_SMOKE=1 ./scripts/smoke_vllm.sh

## run: Build and run with example config (swap mode)
run: build
	./$(BINARY_PATH) -config configs/example.yaml

## run-scheduler: Build and run with scheduler config
run-scheduler: build
	./$(BINARY_PATH) -config configs/scheduler_example.yaml

## deps: Download dependencies
deps:
	$(GO) mod download

## tidy: Tidy go.mod
tidy:
	$(GO) mod tidy

## check: Run fmt-check, vet, and tests
check: fmt-check vet test

## all: Build and run all checks
all: clean build check
