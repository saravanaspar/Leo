.PHONY: check build test release

check:
	./scripts/check.sh

build:
	cargo build --workspace

test:
	cargo test --workspace

release:
	cargo build --release -p leo-cli
