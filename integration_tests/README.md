# Integration tests

## Developing
Integration tests are gated behind the `integration-tests` feature.

To run integration tests, run
```sh
cargo run --example integration_tests
```

## Setup

The tests drive a real enforcer, bitcoind and electrs. They read the paths to
these binaries from environment variables. An example env file is provided
[here](/integration_tests/example.env).

Copy it to `integrationtests.env` in the repo root, and set the paths. The
tests read that file from the working directory or from a parent directory.

```sh
cargo run --example integration_tests
```

Pass a test name after `--` to run a single test.

An env file is optional. Variables that are already set in the environment
take precedence over `integrationtests.env`. To load a different env file, set
`TRUTHCOIN_INTEGRATION_TEST_ENV` to its path. The values in that file override
the environment.
