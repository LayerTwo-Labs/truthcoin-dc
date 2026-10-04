# Integration tests

## Developing
Integration tests are gated behind the `integration-tests` feature.

To run integration tests, run
```sh
cargo run --example integration_tests
```

## Setup

The tests drive a real enforcer, bitcoind and electrs. The quickest way to get
those in place is

```sh
./scripts/setup_integration_tests.sh
```

which fetches or builds the binaries and writes their paths to
`integrationtests.env` in the repo root. The tests read that file from the
working directory or from a parent directory. To write the file by hand, start
from the example env file [here](/integration_tests/example.env).

```sh
cargo run --example integration_tests
```

Pass a test name after `--` to run a single test.

An env file is optional. Variables that are already set in the environment
take precedence over `integrationtests.env`. To load a different env file, set
`TRUTHCOIN_INTEGRATION_TEST_ENV` to its path. The values in that file override
the environment.
