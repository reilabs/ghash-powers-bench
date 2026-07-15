# Binary-field PMULL benchmarks

This crate compares fixed-base exponentiation and full-width multiplication in
several binary fields. The optimized AArch64 paths use the generic `PMULL`
carry-less multiplication instruction; there is no GHASH-only instruction in
these benchmarks.

The implementations use field-specific, straight-line modular reduction and
dispatch on CPU features once per batch. The timed PMULL loops contain no
per-multiplication indirect dispatch.

## Fields

| CLI name | Field or modulus | Stored element | Main product |
|---|---|---:|---:|
| `b127` | GF(2^127), `x^127 + x + 1` | 16 bytes | 3 PMULL |
| `ghash128` | GF(2^128), `x^128 + x^7 + x^2 + x + 1` | 16 bytes | 3 PMULL |
| `b163` | GF(2^163), `x^163 + x^7 + x^6 + x^3 + 1` | 24 bytes | 6 PMULL |
| `b191` | GF(2^191), `x^191 + x^9 + 1` | 24 bytes | 6 PMULL |
| `sect193` | GF(2^193), `x^193 + x^15 + 1` | 32 bytes | 6 PMULL plus the 193rd-bit terms |
| `ghash2` | `K[v]/(v^2 + v + x^121)`, where `K` is GHASH | 32 bytes | 3 GHASH multiplications |

The degree-191 trinomial was selected from [Joerg Arndt's list of irreducible
trinomials over GF(2)](https://www.jjj.de/mathdata/all-trinomial-irredpoly-short.txt).

## What the window size means

The power benchmark computes many powers of one fixed generator. For a window
width `w`, a 128-bit exponent is split into

```text
ceil(128 / w)
```

windows. For every window, the precomputed table stores all `2^w` possible
contributions. A power then selects at most one table entry per window and
multiplies the selected entries together.

Increasing `w` has two opposing effects:

- It reduces the number of field multiplications per power.
- It doubles each window's table size for every extra bit, increasing cache and
  TLB pressure.

The total table size is

```text
ceil(128 / w) * 2^w * sizeof(field element)
```

Consequently, the best window is not determined solely by field degree. It
depends on element size, multiplication cost, cache hierarchy, batch size, and
the particular CPU. Precomputation itself is outside the timed section.

All power benchmarks in this repository use uniformly random exponents in
`[0, 2^128)`, even when the field is larger than 128 bits. Thus every field does
the same exponent-width workload. The multiplication benchmark is different:
its operands are uniformly random across the entire field, with only bits above
the canonical field representation cleared.

## Tuned windows on this machine

Windows from roughly 9 through 17 bits were swept, with neighboring candidates
confirmed at batches of 2^21 and 2^22 powers. These defaults were fastest on the
current AArch64 PMULL machine for large batches:

| Field | Window bits | Windows | Table size |
|---|---:|---:|---:|
| `b127` | 15 | 9 | 4.5 MiB |
| `ghash128` | 15 | 9 | 4.5 MiB |
| `ghash2` | 13 | 10 | 2.5 MiB |
| `b163` | 15 | 9 | 6.8 MiB |
| `b191` | 12 | 11 | 1.0 MiB |
| `sect193` | 12 | 11 | 1.4 MiB |

These are empirical choices, not portable constants. Retune them when changing
CPU, exponent width, representation, or workload size.

## Results on this machine

Measurement environment:

- AArch64 Linux VM exposing an Apple CPU, 8 logical cores
- CPU features include `aes` and `pmull`
- `rustc 1.97.0` targeting `aarch64-unknown-linux-gnu`
- Release profile: optimization level 3 with LTO enabled

The table below uses 2^22 operations and the best of 11 samples. Power timings
use the tuned windows above. `Relative` is elapsed time divided by the
corresponding GHASH timing, so smaller is faster.

| Field | ns/power | Power relative | ns/full-width mul | Mul relative |
|---|---:|---:|---:|---:|
| `b127` | 48.155 | 0.74x | 1.950 | 0.70x |
| `ghash128` | 65.424 | 1.00x | 2.783 | 1.00x |
| `b191` | 111.470 | 1.70x | 3.204 | 1.15x |
| `b163` | 129.923 | 1.99x | 4.398 | 1.58x |
| `sect193` | 132.800 | 2.03x | 4.064 | 1.46x |
| `ghash2` | 167.453 | 2.56x | 9.061 | 3.26x |

The direct multiplication benchmark is a bulk-throughput measurement over
independent random operands. Operand loads and result stores are included; RNG,
allocation, and checksum folding are outside the timed interval. The power
benchmark includes table lookups and field multiplications but excludes table
construction, random exponent generation, output allocation, and checksum
folding.

Some notable results:

- `b127` is about 26% faster than GHASH for fixed-base powers and 30% faster for
  multiplication. Both use three PMULLs, but `x^127 = x + 1` gives b127 a much
  cheaper reduction.
- `b191` is faster than the smaller `b163`. Both use six PMULLs, while b191 has
  a compact trinomial reducer and b163 has a pentanomial reducer.
- `sect193` also uses six PMULLs for its low 192 bits. Its lone fourth-limb bit
  is handled with masked XORs rather than another full limb multiplication.
- One `ghash2` multiplication performs three GHASH multiplications, which is
  reflected in its direct multiplication time.

Absolute timings can change with VM scheduling, thermal throttling, and CPU
frequency. Compare fields using measurements from the same run.

## Running the benchmarks

Build and run a fixed-base power benchmark:

```bash
cargo run --release -- --field b191 --min-log 20 --max-log 22 --samples 11
```

Override the precomputation window when tuning:

```bash
cargo run --release -- --field b191 --min-log 22 --max-log 22 \
  --samples 11 --window-bits 13
```

Run full-width random field multiplications:

```bash
cargo run --release -- --mul --field b191 \
  --min-log 20 --max-log 22 --samples 11
```

Supported field names are `b127`, `ghash128`, `ghash2`, `b163`, `b191`, and
`sect193`.

## Correctness checks

Run the test suite with:

```bash
cargo test
```

The tests compare optimized multiplication and reduction against a generic
polynomial reference, exercise random full-width operands, and check the
specialized PMULL batch kernels against reference table powering.
