use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::{
    env,
    hint::black_box,
    mem::size_of,
    sync::OnceLock,
    time::{Duration, Instant},
};

const DEFAULT_MIN_LOG: u32 = 16;
const DEFAULT_MAX_LOG: u32 = 24;
const DEFAULT_SAMPLES: usize = 1;
const DEFAULT_SEED: u64 = 0x4748_4153_485f_2026;
const DEFAULT_B127_WINDOW_BITS: usize = 15;
const DEFAULT_GHASH128_WINDOW_BITS: usize = 15;
const DEFAULT_GHASH2_WINDOW_BITS: usize = 13;
const DEFAULT_B163_WINDOW_BITS: usize = 15;
const DEFAULT_B191_WINDOW_BITS: usize = 12;
const DEFAULT_SECT193_WINDOW_BITS: usize = 12;
const MAX_WINDOW_BITS: usize = 20;
const MAX_FIELD_LIMBS: usize = 4;
const MAX_PRODUCT_LIMBS: usize = MAX_FIELD_LIMBS * 2;
const EXPONENT_BITS: usize = 128;
const GHASH2_DELTA_U_POWER: usize = 121;
#[cfg(test)]
const GHASH2_BITS: usize = 256;

static B127_TERMS: [usize; 2] = [0, 1];
static B163_TERMS: [usize; 4] = [0, 3, 6, 7];
// Degree-191 irreducible trinomial from
// https://www.jjj.de/mathdata/all-trinomial-irredpoly-short.txt
static B191_TERMS: [usize; 2] = [0, 9];
static SECT193_TERMS: [usize; 2] = [0, 15];

static B163_FIELD: BinaryFieldSpec = BinaryFieldSpec {
    name: "b163",
    description: "GF(2^163) / (u^163 + u^7 + u^6 + u^3 + 1)",
    bits: 163,
    terms: &B163_TERMS,
};

static B127_FIELD: BinaryFieldSpec = BinaryFieldSpec {
    name: "b127",
    description: "GF(2^127) / (u^127 + u + 1)",
    bits: 127,
    terms: &B127_TERMS,
};

static SECT193_FIELD: BinaryFieldSpec = BinaryFieldSpec {
    name: "sect193",
    description: "GF(2^193) / (u^193 + u^15 + 1)",
    bits: 193,
    terms: &SECT193_TERMS,
};

static B191_FIELD: BinaryFieldSpec = BinaryFieldSpec {
    name: "b191",
    description: "GF(2^191) / (u^191 + u^9 + 1)",
    bits: 191,
    terms: &B191_TERMS,
};

fn main() {
    let config = Config::parse_or_exit();

    if config.multiply {
        match config.field {
            FieldChoice::B127 => run_b127_multiply(config),
            FieldChoice::Ghash128 => run_ghash128_multiply(config),
            FieldChoice::Ghash2 => run_ghash2_multiply(config),
            FieldChoice::B163 => run_b163_multiply(config),
            FieldChoice::B191 => run_b191_multiply(config),
            FieldChoice::Sect193 => run_sect193_multiply(config),
        }
        return;
    }

    match config.field {
        FieldChoice::B127 => run_b127(config),
        FieldChoice::Ghash128 => run_ghash128(config),
        FieldChoice::Ghash2 => run_ghash2(config),
        FieldChoice::B163 => run_b163(config),
        FieldChoice::B191 => run_b191(config),
        FieldChoice::Sect193 => run_binary_field(config, &SECT193_FIELD),
    }
}

fn run_ghash128(config: Config) {
    let max_batch = 1usize << config.max_log;

    println!("GHASH random fixed-base powers benchmark");
    println!("field: GF(2^128) / (u^128 + u^7 + u^2 + u + 1)");
    println!("generator: u (primitive element), coefficient encoding 0x2");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = detected_backend();
    println!("multiplication backend: {}", backend.name);
    let compute = select_ghash_compute(backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );

    let table = FixedBaseTable::new(GhashElement::GENERATOR, config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_exponents(&mut rng, max_batch);
    let mut outputs = vec![GhashElement::ZERO; max_batch];

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );

    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let exponents = &exponents[..batch];
        let outputs = &mut outputs[..batch];

        let mut best = Duration::MAX;
        let mut checksum = GhashElement::ZERO;

        for _ in 0..config.samples {
            outputs.fill(GhashElement::ZERO);

            let start = Instant::now();
            compute(black_box(&table), black_box(exponents), black_box(outputs));
            let elapsed = start.elapsed();

            checksum = fold_checksum(black_box(outputs));
            black_box(checksum);
            best = best.min(elapsed);
        }

        let best_ms = best.as_secs_f64() * 1_000.0;
        let ns_per_elem = best.as_secs_f64() * 1_000_000_000.0 / batch as f64;
        println!(
            "{log:>8} {batch:>12} {best_ms:>14.3} {ns_per_elem:>14.3} 0x{:032x}",
            checksum.0
        );
    }
}

fn run_b127(config: Config) {
    let max_batch = 1usize << config.max_log;
    println!("Binary-field random fixed-base powers benchmark");
    println!("field: {}", B127_FIELD.description);
    println!("base: u (coefficient encoding 0x2)");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = b127_backend();
    println!("multiplication backend: {}", backend.name);
    let compute = select_b127_compute(backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );
    let table = FixedBaseTable::new(GhashElement::GENERATOR, config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_exponents(&mut rng, max_batch);
    let mut outputs = vec![GhashElement::ZERO; max_batch];
    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );
    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = GhashElement::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            compute(
                black_box(&table),
                black_box(&exponents[..batch]),
                black_box(&mut outputs[..batch]),
            );
            best = best.min(start.elapsed());
            checksum = fold_checksum(black_box(&outputs[..batch]));
            black_box(checksum);
        }
        println!(
            "{log:>8} {batch:>12} {:>14.3} {:>14.3} 0x{:032x}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / batch as f64,
            checksum.0
        );
    }
}

fn run_ghash2(config: Config) {
    let max_batch = 1usize << config.max_log;

    println!("GHASH quadratic-extension random fixed-base powers benchmark");
    println!("field: K[v] / (v^2 + v + u^121), where K = GF(2^128) / (u^128 + u^7 + u^2 + u + 1)");
    println!("generator: v (extension root)");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = detected_backend();
    println!("base-field multiplication backend: {}", backend.name);
    let compute = select_ghash2_compute(backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );

    let table = Ghash2FixedBaseTable::new(Ghash2Element::GENERATOR, config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_binary_exponents(&mut rng, EXPONENT_BITS, max_batch);
    let mut outputs = vec![Ghash2Element::ZERO; max_batch];

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );

    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let exponents = &exponents[..batch];
        let outputs = &mut outputs[..batch];

        let mut best = Duration::MAX;
        let mut checksum = Ghash2Element::ZERO;

        for _ in 0..config.samples {
            outputs.fill(Ghash2Element::ZERO);

            let start = Instant::now();
            compute(black_box(&table), black_box(exponents), black_box(outputs));
            let elapsed = start.elapsed();

            checksum = fold_ghash2_checksum(black_box(outputs));
            black_box(checksum);
            best = best.min(elapsed);
        }

        let best_ms = best.as_secs_f64() * 1_000.0;
        let ns_per_elem = best.as_secs_f64() * 1_000_000_000.0 / batch as f64;
        println!(
            "{log:>8} {batch:>12} {best_ms:>14.3} {ns_per_elem:>14.3} {}",
            format_ghash2_hex(checksum)
        );
    }
}

fn run_binary_field(config: Config, spec: &'static BinaryFieldSpec) {
    let max_batch = 1usize << config.max_log;

    println!("Binary-field random fixed-base powers benchmark");
    println!("field: {} ({})", spec.name, spec.description);
    println!("base: u (coefficient encoding bit 1)");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = BinaryMulBackend::detect(spec);
    println!("multiplication backend: {}", backend.name);
    let compute = select_binary_compute(spec, backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );

    let table = BinaryFixedBaseTable::new(spec, config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );

    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_binary_exponents(&mut rng, EXPONENT_BITS, max_batch);
    let mut outputs = vec![BinaryElement::ZERO; max_batch];

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );

    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let exponents = &exponents[..batch];
        let outputs = &mut outputs[..batch];

        let mut best = Duration::MAX;
        let mut checksum = BinaryElement::ZERO;

        for _ in 0..config.samples {
            outputs.fill(BinaryElement::ZERO);

            let start = Instant::now();
            compute(black_box(&table), black_box(exponents), black_box(outputs));
            let elapsed = start.elapsed();

            checksum = fold_binary_checksum(black_box(outputs));
            black_box(checksum);
            best = best.min(elapsed);
        }

        let best_ms = best.as_secs_f64() * 1_000.0;
        let ns_per_elem = best.as_secs_f64() * 1_000_000_000.0 / batch as f64;
        println!(
            "{log:>8} {batch:>12} {best_ms:>14.3} {ns_per_elem:>14.3} {}",
            format_binary_hex(checksum, spec.bits)
        );
    }
}

fn run_b163(config: Config) {
    let max_batch = 1usize << config.max_log;

    println!("Binary-field random fixed-base powers benchmark");
    println!("field: b163 ({})", B163_FIELD.description);
    println!("base: u (coefficient encoding bit 1)");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = B163MulBackend::detect();
    println!("multiplication backend: {}", backend.name);
    let compute = select_b163_compute(backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );

    let table = B163FixedBaseTable::new(config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );

    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_binary_exponents(&mut rng, EXPONENT_BITS, max_batch);
    let mut outputs = vec![B163Element::ZERO; max_batch];

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );

    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let exponents = &exponents[..batch];
        let outputs = &mut outputs[..batch];
        let mut best = Duration::MAX;
        let mut checksum = B163Element::ZERO;

        for _ in 0..config.samples {
            outputs.fill(B163Element::ZERO);
            let start = Instant::now();
            compute(black_box(&table), black_box(exponents), black_box(outputs));
            let elapsed = start.elapsed();
            checksum = outputs
                .iter()
                .copied()
                .fold(B163Element::ZERO, B163Element::add);
            black_box(checksum);
            best = best.min(elapsed);
        }

        let best_ms = best.as_secs_f64() * 1_000.0;
        let ns_per_elem = best.as_secs_f64() * 1_000_000_000.0 / batch as f64;
        println!(
            "{log:>8} {batch:>12} {best_ms:>14.3} {ns_per_elem:>14.3} {}",
            format_binary_hex(checksum.into_binary(), B163_FIELD.bits)
        );
    }
}

fn run_b191(config: Config) {
    let max_batch = 1usize << config.max_log;
    println!("Binary-field random fixed-base powers benchmark");
    println!("field: b191 ({})", B191_FIELD.description);
    println!("base: u (coefficient encoding bit 1)");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    let backend = b191_backend();
    println!("multiplication backend: {}", backend.name);
    let compute = select_b191_compute(backend);
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );
    let table = B163FixedBaseTable::new(config.window_bits, backend);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count(),
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_binary_exponents(&mut rng, EXPONENT_BITS, max_batch);
    let mut outputs = vec![B163Element::ZERO; max_batch];
    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );
    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = B163Element::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            compute(
                black_box(&table),
                black_box(&exponents[..batch]),
                black_box(&mut outputs[..batch]),
            );
            best = best.min(start.elapsed());
            checksum = outputs[..batch]
                .iter()
                .copied()
                .fold(B163Element::ZERO, B163Element::add);
            black_box(checksum);
        }
        println!(
            "{log:>8} {batch:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / batch as f64,
            format_binary_hex(checksum.into_binary(), B191_FIELD.bits)
        );
    }
}

fn run_ghash128_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = detected_backend();
    println!("Full-field multiplication benchmark");
    println!("field: GHASH GF(2^128)");
    println!("multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_exponents(&mut rng, max_batch);
    let rhs = random_exponents(&mut rng, max_batch);
    let mut outputs = vec![GhashElement::ZERO; max_batch];
    run_ghash_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn run_b127_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = b127_backend();
    println!("Full-field multiplication benchmark");
    println!("field: b127 GF(2^127)");
    println!("multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_b127_elements(&mut rng, max_batch);
    let rhs = random_b127_elements(&mut rng, max_batch);
    let mut outputs = vec![GhashElement::ZERO; max_batch];
    run_b127_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn run_ghash2_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = detected_backend();
    println!("Full-field multiplication benchmark");
    println!("field: GHASH quadratic extension (256 bits)");
    println!("base-field multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_ghash2_elements(&mut rng, max_batch);
    let rhs = random_ghash2_elements(&mut rng, max_batch);
    let mut outputs = vec![Ghash2Element::ZERO; max_batch];
    run_ghash2_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn run_b163_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = B163MulBackend::detect();
    println!("Full-field multiplication benchmark");
    println!("field: b163 GF(2^163)");
    println!("multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_b163_elements(&mut rng, max_batch);
    let rhs = random_b163_elements(&mut rng, max_batch);
    let mut outputs = vec![B163Element::ZERO; max_batch];
    run_b163_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn run_b191_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = b191_backend();
    println!("Full-field multiplication benchmark");
    println!("field: b191 GF(2^191)");
    println!("multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_b191_elements(&mut rng, max_batch);
    let rhs = random_b191_elements(&mut rng, max_batch);
    let mut outputs = vec![B163Element::ZERO; max_batch];
    run_b191_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn run_sect193_multiply(config: Config) {
    let max_batch = 1usize << config.max_log;
    let backend = BinaryMulBackend::detect(&SECT193_FIELD);
    println!("Full-field multiplication benchmark");
    println!("field: sect193 GF(2^193)");
    println!("multiplication backend: {}", backend.name);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs = random_binary_exponents(&mut rng, SECT193_FIELD.bits, max_batch);
    let rhs = random_binary_exponents(&mut rng, SECT193_FIELD.bits, max_batch);
    let mut outputs = vec![BinaryElement::ZERO; max_batch];
    run_sect193_mul_batches(config, backend, &lhs, &rhs, &mut outputs);
}

fn random_ghash2_elements(rng: &mut StdRng, len: usize) -> Vec<Ghash2Element> {
    (0..len)
        .map(|_| Ghash2Element {
            c0: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
            c1: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
        })
        .collect()
}

fn random_b127_elements(rng: &mut StdRng, len: usize) -> Vec<u128> {
    random_exponents(rng, len)
        .into_iter()
        .map(|value| value & (u128::MAX >> 1))
        .collect()
}

fn random_b163_elements(rng: &mut StdRng, len: usize) -> Vec<B163Element> {
    (0..len)
        .map(|_| B163Element {
            limbs: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64() & ((1u64 << 35) - 1),
            ],
        })
        .collect()
}

fn random_b191_elements(rng: &mut StdRng, len: usize) -> Vec<B163Element> {
    (0..len)
        .map(|_| B163Element {
            limbs: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64() & (u64::MAX >> 1),
            ],
        })
        .collect()
}

fn print_mul_header() {
    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/mul", "checksum"
    );
}

fn run_ghash_mul_batches(
    config: Config,
    backend: Backend,
    lhs: &[u128],
    rhs: &[u128],
    outputs: &mut [GhashElement],
) {
    let multiply = select_ghash_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = GhashElement::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = fold_checksum(black_box(&outputs[..n]));
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} 0x{:032x}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            checksum.0
        );
    }
}

fn run_b127_mul_batches(
    config: Config,
    backend: Backend,
    lhs: &[u128],
    rhs: &[u128],
    outputs: &mut [GhashElement],
) {
    let multiply = select_b127_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = GhashElement::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = fold_checksum(black_box(&outputs[..n]));
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} 0x{:032x}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            checksum.0
        );
    }
}

fn run_ghash2_mul_batches(
    config: Config,
    backend: Backend,
    lhs: &[Ghash2Element],
    rhs: &[Ghash2Element],
    outputs: &mut [Ghash2Element],
) {
    let multiply = select_ghash2_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = Ghash2Element::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = fold_ghash2_checksum(black_box(&outputs[..n]));
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            format_ghash2_hex(checksum)
        );
    }
}

fn run_b163_mul_batches(
    config: Config,
    backend: B163MulBackend,
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
) {
    let multiply = select_b163_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = B163Element::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = outputs[..n]
                .iter()
                .copied()
                .fold(B163Element::ZERO, B163Element::add);
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            format_binary_hex(checksum.into_binary(), B163_FIELD.bits)
        );
    }
}

fn run_b191_mul_batches(
    config: Config,
    backend: B163MulBackend,
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
) {
    let multiply = select_b191_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = B163Element::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = outputs[..n]
                .iter()
                .copied()
                .fold(B163Element::ZERO, B163Element::add);
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            format_binary_hex(checksum.into_binary(), B191_FIELD.bits)
        );
    }
}

fn run_sect193_mul_batches(
    config: Config,
    backend: BinaryMulBackend,
    lhs: &[BinaryElement],
    rhs: &[BinaryElement],
    outputs: &mut [BinaryElement],
) {
    let multiply = select_sect193_mul_batch();
    print_mul_header();
    for log in config.min_log..=config.max_log {
        let n = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = BinaryElement::ZERO;
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..n]),
                black_box(&rhs[..n]),
                black_box(&mut outputs[..n]),
                backend,
            );
            best = best.min(start.elapsed());
            checksum = fold_binary_checksum(black_box(&outputs[..n]));
            black_box(checksum);
        }
        println!(
            "{log:>8} {n:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / n as f64,
            format_binary_hex(checksum, SECT193_FIELD.bits)
        );
    }
}

fn compute_random_powers(table: &FixedBaseTable, exponents: &[u128], outputs: &mut [GhashElement]) {
    assert_eq!(exponents.len(), outputs.len());

    for (exponent, output) in exponents.iter().zip(outputs) {
        *output = table.pow(*exponent);
    }
}

fn compute_ghash2_random_powers(
    table: &Ghash2FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [Ghash2Element],
) {
    assert_eq!(exponents.len(), outputs.len());

    for (exponent, output) in exponents.iter().zip(outputs) {
        *output = table.pow(*exponent);
    }
}

fn random_exponents(rng: &mut StdRng, len: usize) -> Vec<u128> {
    (0..len)
        .map(|_| {
            let low = rng.next_u64() as u128;
            let high = rng.next_u64() as u128;
            low | (high << 64)
        })
        .collect()
}

fn fold_checksum(outputs: &[GhashElement]) -> GhashElement {
    outputs
        .iter()
        .copied()
        .fold(GhashElement::ZERO, |acc, value| acc.add(value))
}

fn fold_ghash2_checksum(outputs: &[Ghash2Element]) -> Ghash2Element {
    outputs
        .iter()
        .copied()
        .fold(Ghash2Element::ZERO, |acc, value| acc.add(value))
}

fn compute_binary_random_powers(
    table: &BinaryFixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [BinaryElement],
) {
    assert_eq!(exponents.len(), outputs.len());

    for (exponent, output) in exponents.iter().zip(outputs) {
        *output = table.pow(*exponent);
    }
}

fn compute_b163_random_powers(
    table: &B163FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [B163Element],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().zip(outputs) {
        *output = table.pow(*exponent);
    }
}

type GhashCompute = fn(&FixedBaseTable, &[u128], &mut [GhashElement]);
type Ghash2Compute = fn(&Ghash2FixedBaseTable, &[BinaryElement], &mut [Ghash2Element]);
type BinaryCompute = fn(&BinaryFixedBaseTable, &[BinaryElement], &mut [BinaryElement]);
type B163Compute = fn(&B163FixedBaseTable, &[BinaryElement], &mut [B163Element]);
type GhashMulBatch = fn(&[u128], &[u128], &mut [GhashElement], Backend);
type Ghash2MulBatch = fn(&[Ghash2Element], &[Ghash2Element], &mut [Ghash2Element], Backend);
type B163MulBatch = fn(&[B163Element], &[B163Element], &mut [B163Element], B163MulBackend);
type Sect193MulBatch =
    fn(&[BinaryElement], &[BinaryElement], &mut [BinaryElement], BinaryMulBackend);

fn select_ghash_mul_batch() -> GhashMulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_ghash_batch_pmull_checked;
    }
    multiply_ghash_batch
}

fn select_b127_mul_batch() -> GhashMulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_b127_batch_pmull_checked;
    }
    multiply_ghash_batch
}

fn select_ghash2_mul_batch() -> Ghash2MulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_ghash2_batch_pmull_checked;
    }
    multiply_ghash2_batch
}

fn select_b163_mul_batch() -> B163MulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_b163_batch_pmull_checked;
    }
    multiply_b163_batch
}

fn select_b191_mul_batch() -> B163MulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_b191_batch_pmull_checked;
    }
    multiply_b163_batch
}

fn select_sect193_mul_batch() -> Sect193MulBatch {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_sect193_batch_pmull_checked;
    }
    multiply_sect193_batch
}

fn multiply_ghash_batch(
    lhs: &[u128],
    rhs: &[u128],
    outputs: &mut [GhashElement],
    backend: Backend,
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = GhashElement((backend.mul)(*lhs, *rhs));
    }
}

fn multiply_ghash2_batch(
    lhs: &[Ghash2Element],
    rhs: &[Ghash2Element],
    outputs: &mut [Ghash2Element],
    backend: Backend,
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = lhs.mul(*rhs, backend.mul);
    }
}

fn multiply_b163_batch(
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
    backend: B163MulBackend,
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = (backend.mul)(*lhs, *rhs);
    }
}

fn multiply_sect193_batch(
    lhs: &[BinaryElement],
    rhs: &[BinaryElement],
    outputs: &mut [BinaryElement],
    backend: BinaryMulBackend,
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = (backend.mul)(*lhs, *rhs);
    }
}

#[cfg(target_arch = "aarch64")]
fn multiply_ghash_batch_pmull_checked(
    lhs: &[u128],
    rhs: &[u128],
    outputs: &mut [GhashElement],
    _backend: Backend,
) {
    unsafe { multiply_ghash_batch_pmull(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn multiply_b127_batch_pmull_checked(
    lhs: &[u128],
    rhs: &[u128],
    outputs: &mut [GhashElement],
    _backend: Backend,
) {
    unsafe { multiply_b127_batch_pmull(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn multiply_ghash2_batch_pmull_checked(
    lhs: &[Ghash2Element],
    rhs: &[Ghash2Element],
    outputs: &mut [Ghash2Element],
    _backend: Backend,
) {
    unsafe { multiply_ghash2_batch_pmull(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn multiply_b163_batch_pmull_checked(
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
    _backend: B163MulBackend,
) {
    unsafe { multiply_b163_batch_pmull(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn multiply_b191_batch_pmull_checked(
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
    _backend: B163MulBackend,
) {
    unsafe { multiply_b191_batch_pmull(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn multiply_sect193_batch_pmull_checked(
    lhs: &[BinaryElement],
    rhs: &[BinaryElement],
    outputs: &mut [BinaryElement],
    _backend: BinaryMulBackend,
) {
    unsafe { multiply_sect193_batch_pmull(lhs, rhs, outputs) }
}

fn select_ghash_compute(_backend: Backend) -> GhashCompute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_random_powers_pmull_checked;
    }
    compute_random_powers
}

fn select_b127_compute(_backend: Backend) -> GhashCompute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_b127_random_powers_pmull_checked;
    }
    compute_random_powers
}

fn select_ghash2_compute(_backend: Backend) -> Ghash2Compute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_ghash2_random_powers_pmull_checked;
    }
    compute_ghash2_random_powers
}

fn select_binary_compute(
    spec: &'static BinaryFieldSpec,
    _backend: BinaryMulBackend,
) -> BinaryCompute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return match spec.name {
            "sect193" => compute_sect193_random_powers_pmull_checked,
            _ => compute_binary_random_powers,
        };
    }
    compute_binary_random_powers
}

fn select_b163_compute(_backend: B163MulBackend) -> B163Compute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_b163_random_powers_pmull_checked;
    }
    compute_b163_random_powers
}

fn select_b191_compute(_backend: B163MulBackend) -> B163Compute {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_b191_random_powers_pmull_checked;
    }
    compute_b163_random_powers
}

#[cfg(target_arch = "aarch64")]
fn compute_random_powers_pmull_checked(
    table: &FixedBaseTable,
    exponents: &[u128],
    outputs: &mut [GhashElement],
) {
    // SAFETY: selected only after runtime PMULL detection.
    unsafe { compute_random_powers_pmull(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn compute_b127_random_powers_pmull_checked(
    table: &FixedBaseTable,
    exponents: &[u128],
    outputs: &mut [GhashElement],
) {
    unsafe { compute_b127_random_powers_pmull(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn compute_ghash2_random_powers_pmull_checked(
    table: &Ghash2FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [Ghash2Element],
) {
    // SAFETY: selected only after runtime PMULL detection.
    unsafe { compute_ghash2_random_powers_pmull(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn compute_b163_random_powers_pmull_checked(
    table: &B163FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [B163Element],
) {
    // SAFETY: selected only after runtime PMULL detection.
    unsafe { compute_b163_random_powers_pmull(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn compute_b191_random_powers_pmull_checked(
    table: &B163FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [B163Element],
) {
    unsafe { compute_b191_random_powers_pmull(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
fn compute_sect193_random_powers_pmull_checked(
    table: &BinaryFixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [BinaryElement],
) {
    // SAFETY: selected only after runtime PMULL detection.
    unsafe { compute_sect193_random_powers_pmull(table, exponents, outputs) }
}

fn random_binary_exponents(rng: &mut StdRng, bits: usize, len: usize) -> Vec<BinaryElement> {
    let limb_count = limb_count(bits);
    (0..len)
        .map(|_| {
            let mut limbs = [0u64; MAX_FIELD_LIMBS];
            for limb in limbs.iter_mut().take(limb_count) {
                *limb = rng.next_u64();
            }
            BinaryElement { limbs }.masked(bits)
        })
        .collect()
}

fn fold_binary_checksum(outputs: &[BinaryElement]) -> BinaryElement {
    outputs
        .iter()
        .copied()
        .fold(BinaryElement::ZERO, |acc, value| acc.add(value))
}

struct Config {
    field: FieldChoice,
    multiply: bool,
    min_log: u32,
    max_log: u32,
    samples: usize,
    seed: u64,
    window_bits: usize,
}

impl Config {
    fn parse_or_exit() -> Self {
        match Self::parse(env::args().skip(1)) {
            Ok(config) => config,
            Err(message) => {
                eprintln!("{message}");
                eprintln!();
                eprintln!(
                    "usage: cargo run --release --bin ghash-powers-bench -- [--field b127|ghash128|ghash2|b163|b191|sect193] [--mul] [--min-log N] [--max-log N] [--samples N] [--window-bits N] [--seed N]"
                );
                eprintln!(
                    "defaults: --field ghash128 --min-log 16 --max-log 24 --samples 1 --window-bits field-specific --seed 0x47484153485f2026"
                );
                std::process::exit(2);
            }
        }
    }

    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = Self {
            field: FieldChoice::Ghash128,
            multiply: false,
            min_log: DEFAULT_MIN_LOG,
            max_log: DEFAULT_MAX_LOG,
            samples: DEFAULT_SAMPLES,
            seed: DEFAULT_SEED,
            window_bits: DEFAULT_GHASH128_WINDOW_BITS,
        };
        let mut window_bits_was_set = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    return Err(String::from("GHASH random fixed-base powers benchmark"));
                }
                "--mul" => config.multiply = true,
                "--field" => {
                    config.field = parse_field(&next_arg(&mut args, "--field")?)?;
                }
                "--min-log" => {
                    config.min_log = parse_next(&mut args, "--min-log")?;
                }
                "--max-log" => {
                    config.max_log = parse_next(&mut args, "--max-log")?;
                }
                "--samples" => {
                    config.samples = parse_next(&mut args, "--samples")?;
                }
                "--window-bits" => {
                    config.window_bits = parse_next(&mut args, "--window-bits")?;
                    window_bits_was_set = true;
                }
                "--seed" => {
                    config.seed = parse_next(&mut args, "--seed")?;
                }
                _ => {
                    if let Some((name, value)) = arg.split_once('=') {
                        match name {
                            "--field" => config.field = parse_field(value)?,
                            "--min-log" => config.min_log = parse_value(value, name)?,
                            "--max-log" => config.max_log = parse_value(value, name)?,
                            "--samples" => config.samples = parse_value(value, name)?,
                            "--window-bits" => {
                                config.window_bits = parse_value(value, name)?;
                                window_bits_was_set = true;
                            }
                            "--seed" => config.seed = parse_value(value, name)?,
                            _ => return Err(format!("unknown argument: {arg}")),
                        }
                    } else {
                        return Err(format!("unknown argument: {arg}"));
                    }
                }
            }
        }

        if !window_bits_was_set {
            config.window_bits = default_window_bits(config.field);
        }

        if config.min_log > config.max_log {
            return Err(format!(
                "--min-log ({}) must be <= --max-log ({})",
                config.min_log, config.max_log
            ));
        }
        if config.max_log >= usize::BITS {
            return Err(format!(
                "--max-log ({}) is too large for this target",
                config.max_log
            ));
        }
        if config.samples == 0 {
            return Err(String::from("--samples must be at least 1"));
        }
        if !(1..=MAX_WINDOW_BITS).contains(&config.window_bits) {
            return Err(format!(
                "--window-bits must be in 1..={MAX_WINDOW_BITS}; larger values make the fixed-base table very large"
            ));
        }

        Ok(config)
    }
}

#[derive(Clone, Copy)]
enum FieldChoice {
    B127,
    Ghash128,
    Ghash2,
    B163,
    B191,
    Sect193,
}

fn default_window_bits(field: FieldChoice) -> usize {
    match field {
        FieldChoice::B127 => DEFAULT_B127_WINDOW_BITS,
        FieldChoice::Ghash128 => DEFAULT_GHASH128_WINDOW_BITS,
        FieldChoice::Ghash2 => DEFAULT_GHASH2_WINDOW_BITS,
        FieldChoice::B163 => DEFAULT_B163_WINDOW_BITS,
        FieldChoice::B191 => DEFAULT_B191_WINDOW_BITS,
        FieldChoice::Sect193 => DEFAULT_SECT193_WINDOW_BITS,
    }
}

fn parse_field(value: &str) -> Result<FieldChoice, String> {
    match value {
        "b127" | "127" => Ok(FieldChoice::B127),
        "ghash128" | "ghash" | "128" => Ok(FieldChoice::Ghash128),
        "ghash2" | "ghash256" | "qghash" | "256" => Ok(FieldChoice::Ghash2),
        "b163" | "163" | "nist163" => Ok(FieldChoice::B163),
        "b191" | "191" => Ok(FieldChoice::B191),
        "sect193" | "193" | "sec193" => Ok(FieldChoice::Sect193),
        _ => Err(format!(
            "unknown field {value:?}; expected b127, ghash128, ghash2, b163, b191, or sect193"
        )),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, name: &'static str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_next<T: ParseInteger>(
    args: &mut impl Iterator<Item = String>,
    name: &'static str,
) -> Result<T, String> {
    let value = next_arg(args, name)?;
    parse_value(&value, name)
}

fn parse_value<T: ParseInteger>(value: &str, name: &str) -> Result<T, String> {
    T::parse_integer(value).map_err(|message| format!("invalid {name} value {value:?}: {message}"))
}

trait ParseInteger: Sized {
    fn parse_integer(value: &str) -> Result<Self, String>;
}

impl ParseInteger for u32 {
    fn parse_integer(value: &str) -> Result<Self, String> {
        parse_u128(value).and_then(|value| {
            u32::try_from(value).map_err(|_| String::from("value does not fit in u32"))
        })
    }
}

impl ParseInteger for usize {
    fn parse_integer(value: &str) -> Result<Self, String> {
        parse_u128(value).and_then(|value| {
            usize::try_from(value).map_err(|_| String::from("value does not fit in usize"))
        })
    }
}

impl ParseInteger for u64 {
    fn parse_integer(value: &str) -> Result<Self, String> {
        parse_u128(value).and_then(|value| {
            u64::try_from(value).map_err(|_| String::from("value does not fit in u64"))
        })
    }
}

fn parse_u128(value: &str) -> Result<u128, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u128::from_str_radix(hex, 16).map_err(|error| error.to_string())
    } else {
        value.parse::<u128>().map_err(|error| error.to_string())
    }
}

struct FixedBaseTable {
    table: Vec<GhashElement>,
    mul: MulRaw,
    window_bits: usize,
    window_count: usize,
    window_mask: u128,
    window_size: usize,
}

impl FixedBaseTable {
    fn new(base: GhashElement, window_bits: usize, backend: Backend) -> Self {
        assert!((1..=MAX_WINDOW_BITS).contains(&window_bits));

        let window_count = 128usize.div_ceil(window_bits);
        let window_size = 1usize << window_bits;
        let window_mask = (1u128 << window_bits) - 1;
        let mut table = vec![GhashElement::ONE; window_count * window_size];
        let mut window_base = base;
        let mul = backend.mul;

        for window in 0..window_count {
            let offset = window * window_size;
            for value in 1..window_size {
                table[offset + value] =
                    GhashElement(mul(table[offset + value - 1].0, window_base.0));
            }

            for _ in 0..window_bits {
                window_base = GhashElement(mul(window_base.0, window_base.0));
            }
        }

        Self {
            table,
            mul,
            window_bits,
            window_count,
            window_mask,
            window_size,
        }
    }

    fn pow(&self, exponent: u128) -> GhashElement {
        let mut acc = GhashElement::ONE;
        let mul = self.mul;

        for window in 0..self.window_count {
            let shift = window * self.window_bits;
            let value = ((exponent >> shift) & self.window_mask) as usize;
            if value != 0 {
                acc = GhashElement(mul(acc.0, self.table[window * self.window_size + value].0));
            }
        }

        acc
    }

    fn window_count(&self) -> usize {
        self.window_count
    }

    fn table_bytes(&self) -> usize {
        self.table.len() * size_of::<GhashElement>()
    }
}

struct Ghash2FixedBaseTable {
    table: Vec<Ghash2Element>,
    mul: MulRaw,
    window_bits: usize,
    window_count: usize,
    window_mask: u64,
    window_size: usize,
}

impl Ghash2FixedBaseTable {
    fn new(base: Ghash2Element, window_bits: usize, backend: Backend) -> Self {
        assert!((1..=MAX_WINDOW_BITS).contains(&window_bits));

        let window_count = EXPONENT_BITS.div_ceil(window_bits);
        let window_size = 1usize << window_bits;
        let window_mask = (1u64 << window_bits) - 1;
        let mut table = vec![Ghash2Element::ONE; window_count * window_size];
        let mut window_base = base;
        let mul = backend.mul;

        for window in 0..window_count {
            let offset = window * window_size;
            for value in 1..window_size {
                table[offset + value] = table[offset + value - 1].mul(window_base, mul);
            }

            for _ in 0..window_bits {
                window_base = window_base.mul(window_base, mul);
            }
        }

        Self {
            table,
            mul,
            window_bits,
            window_count,
            window_mask,
            window_size,
        }
    }

    fn pow(&self, exponent: BinaryElement) -> Ghash2Element {
        let mut acc = Ghash2Element::ONE;

        for window in 0..self.window_count {
            let shift = window * self.window_bits;
            let value = exponent.window(shift, self.window_bits, self.window_mask);
            if value != 0 {
                acc = acc.mul(self.table[window * self.window_size + value], self.mul);
            }
        }

        acc
    }

    fn window_count(&self) -> usize {
        self.window_count
    }

    fn table_bytes(&self) -> usize {
        self.table.len() * size_of::<Ghash2Element>()
    }
}

struct BinaryFieldSpec {
    name: &'static str,
    description: &'static str,
    bits: usize,
    terms: &'static [usize],
}

struct BinaryFixedBaseTable {
    table: Vec<BinaryElement>,
    mul: BinaryMul,
    window_bits: usize,
    window_count: usize,
    window_mask: u64,
    window_size: usize,
}

impl BinaryFixedBaseTable {
    fn new(spec: &'static BinaryFieldSpec, window_bits: usize, backend: BinaryMulBackend) -> Self {
        assert!((1..=MAX_WINDOW_BITS).contains(&window_bits));

        let window_count = EXPONENT_BITS.div_ceil(window_bits);
        let window_size = 1usize << window_bits;
        let window_mask = (1u64 << window_bits) - 1;
        let mut table = vec![BinaryElement::ONE; window_count * window_size];
        let mut window_base = BinaryElement::U.masked(spec.bits);
        let mul = backend.mul;

        for window in 0..window_count {
            let offset = window * window_size;
            for value in 1..window_size {
                table[offset + value] = mul(table[offset + value - 1], window_base);
            }

            for _ in 0..window_bits {
                window_base = mul(window_base, window_base);
            }
        }

        Self {
            table,
            mul,
            window_bits,
            window_count,
            window_mask,
            window_size,
        }
    }

    fn pow(&self, exponent: BinaryElement) -> BinaryElement {
        let mut acc = BinaryElement::ONE;
        let mul = self.mul;

        for window in 0..self.window_count {
            let shift = window * self.window_bits;
            let value = exponent.window(shift, self.window_bits, self.window_mask);
            if value != 0 {
                acc = mul(acc, self.table[window * self.window_size + value]);
            }
        }

        acc
    }

    fn window_count(&self) -> usize {
        self.window_count
    }

    fn table_bytes(&self) -> usize {
        self.table.len() * size_of::<BinaryElement>()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct B163Element {
    limbs: [u64; 3],
}

impl B163Element {
    const ZERO: Self = Self { limbs: [0; 3] };
    const ONE: Self = Self { limbs: [1, 0, 0] };
    const U: Self = Self { limbs: [2, 0, 0] };

    fn add(self, rhs: Self) -> Self {
        Self {
            limbs: [
                self.limbs[0] ^ rhs.limbs[0],
                self.limbs[1] ^ rhs.limbs[1],
                self.limbs[2] ^ rhs.limbs[2],
            ],
        }
    }

    fn into_binary(self) -> BinaryElement {
        BinaryElement {
            limbs: [self.limbs[0], self.limbs[1], self.limbs[2], 0],
        }
    }

    fn from_binary(value: BinaryElement) -> Self {
        Self {
            limbs: [value.limbs[0], value.limbs[1], value.limbs[2]],
        }
    }
}

type B163Mul = fn(B163Element, B163Element) -> B163Element;

#[derive(Clone, Copy)]
struct B163MulBackend {
    name: &'static str,
    mul: B163Mul,
}

impl B163MulBackend {
    fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("aes") {
                return Self {
                    name: "aarch64-pmull-karatsuba",
                    mul: mul_b163_compact_pmull_checked,
                };
            }
        }

        Self {
            name: "portable",
            mul: mul_b163_compact_portable,
        }
    }
}

fn b191_backend() -> B163MulBackend {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return B163MulBackend {
            name: "aarch64-pmull-karatsuba",
            mul: mul_b191_compact_pmull_checked,
        };
    }
    B163MulBackend {
        name: "portable",
        mul: mul_b191_compact_portable,
    }
}

struct B163FixedBaseTable {
    table: Vec<B163Element>,
    mul: B163Mul,
    window_bits: usize,
    window_count: usize,
    window_mask: u64,
    window_size: usize,
}

impl B163FixedBaseTable {
    fn new(window_bits: usize, backend: B163MulBackend) -> Self {
        assert!((1..=MAX_WINDOW_BITS).contains(&window_bits));

        let window_count = EXPONENT_BITS.div_ceil(window_bits);
        let window_size = 1usize << window_bits;
        let window_mask = (1u64 << window_bits) - 1;
        let mut table = vec![B163Element::ONE; window_count * window_size];
        let mut window_base = B163Element::U;
        let mul = backend.mul;

        for window in 0..window_count {
            let offset = window * window_size;
            for value in 1..window_size {
                table[offset + value] = mul(table[offset + value - 1], window_base);
            }
            for _ in 0..window_bits {
                window_base = mul(window_base, window_base);
            }
        }

        Self {
            table,
            mul,
            window_bits,
            window_count,
            window_mask,
            window_size,
        }
    }

    fn pow(&self, exponent: BinaryElement) -> B163Element {
        let mut acc = B163Element::ONE;
        for window in 0..self.window_count {
            let shift = window * self.window_bits;
            let value = exponent.window(shift, self.window_bits, self.window_mask);
            if value != 0 {
                acc = (self.mul)(acc, self.table[window * self.window_size + value]);
            }
        }
        acc
    }

    fn window_count(&self) -> usize {
        self.window_count
    }

    fn table_bytes(&self) -> usize {
        self.table.len() * size_of::<B163Element>()
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_ghash_batch_pmull(lhs: &[u128], rhs: &[u128], outputs: &mut [GhashElement]) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = GhashElement(unsafe { mul_raw_pmull(*lhs, *rhs) });
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_b127_batch_pmull(lhs: &[u128], rhs: &[u128], outputs: &mut [GhashElement]) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = GhashElement(unsafe { mul_b127_pmull(*lhs, *rhs) });
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_ghash2_batch_pmull(
    lhs: &[Ghash2Element],
    rhs: &[Ghash2Element],
    outputs: &mut [Ghash2Element],
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = unsafe { mul_ghash2_pmull(*lhs, *rhs) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_b163_batch_pmull(
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = unsafe { mul_b163_compact_pmull(*lhs, *rhs) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_b191_batch_pmull(
    lhs: &[B163Element],
    rhs: &[B163Element],
    outputs: &mut [B163Element],
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = unsafe { mul_b191_compact_pmull(*lhs, *rhs) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_sect193_batch_pmull(
    lhs: &[BinaryElement],
    rhs: &[BinaryElement],
    outputs: &mut [BinaryElement],
) {
    for ((lhs, rhs), output) in lhs.iter().zip(rhs).zip(outputs) {
        *output = unsafe { mul_sect193_pmull(*lhs, *rhs) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_random_powers_pmull(
    table: &FixedBaseTable,
    exponents: &[u128],
    outputs: &mut [GhashElement],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = GhashElement::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = ((exponent >> shift) & table.window_mask) as usize;
            if value != 0 {
                acc = GhashElement(unsafe {
                    mul_raw_pmull(acc.0, table.table[window * table.window_size + value].0)
                });
            }
        }
        *output = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_b127_random_powers_pmull(
    table: &FixedBaseTable,
    exponents: &[u128],
    outputs: &mut [GhashElement],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = GhashElement::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = ((exponent >> shift) & table.window_mask) as usize;
            if value != 0 {
                acc = GhashElement(unsafe {
                    mul_b127_pmull(acc.0, table.table[window * table.window_size + value].0)
                });
            }
        }
        *output = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn mul_ghash2_pmull(lhs: Ghash2Element, rhs: Ghash2Element) -> Ghash2Element {
    let c0_product = GhashElement(unsafe { mul_raw_pmull(lhs.c0.0, rhs.c0.0) });
    let c1_product = GhashElement(unsafe { mul_raw_pmull(lhs.c1.0, rhs.c1.0) });
    let cross_plus_products =
        GhashElement(unsafe { mul_raw_pmull(lhs.c0.0 ^ lhs.c1.0, rhs.c0.0 ^ rhs.c1.0) });
    Ghash2Element {
        c0: c0_product.add(mul_by_delta(c1_product)),
        c1: cross_plus_products.add(c0_product),
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_ghash2_random_powers_pmull(
    table: &Ghash2FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [Ghash2Element],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = Ghash2Element::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = exponent.window(shift, table.window_bits, table.window_mask);
            if value != 0 {
                acc = unsafe {
                    mul_ghash2_pmull(acc, table.table[window * table.window_size + value])
                };
            }
        }
        *output = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_b163_random_powers_pmull(
    table: &B163FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [B163Element],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = B163Element::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = exponent.window(shift, table.window_bits, table.window_mask);
            if value != 0 {
                acc = unsafe {
                    mul_b163_compact_pmull(acc, table.table[window * table.window_size + value])
                };
            }
        }
        *output = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_b191_random_powers_pmull(
    table: &B163FixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [B163Element],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = B163Element::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = exponent.window(shift, table.window_bits, table.window_mask);
            if value != 0 {
                acc = unsafe {
                    mul_b191_compact_pmull(acc, table.table[window * table.window_size + value])
                };
            }
        }
        *output = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_sect193_random_powers_pmull(
    table: &BinaryFixedBaseTable,
    exponents: &[BinaryElement],
    outputs: &mut [BinaryElement],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = BinaryElement::ONE;
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = exponent.window(shift, table.window_bits, table.window_mask);
            if value != 0 {
                acc = unsafe {
                    mul_sect193_pmull(acc, table.table[window * table.window_size + value])
                };
            }
        }
        *output = acc;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BinaryElement {
    limbs: [u64; MAX_FIELD_LIMBS],
}

impl BinaryElement {
    const ZERO: Self = Self {
        limbs: [0; MAX_FIELD_LIMBS],
    };
    const ONE: Self = Self {
        limbs: {
            let mut limbs = [0; MAX_FIELD_LIMBS];
            limbs[0] = 1;
            limbs
        },
    };
    const U: Self = Self {
        limbs: {
            let mut limbs = [0; MAX_FIELD_LIMBS];
            limbs[0] = 2;
            limbs
        },
    };

    fn add(self, rhs: Self) -> Self {
        let mut limbs = [0u64; MAX_FIELD_LIMBS];
        for (out, (lhs, rhs)) in limbs.iter_mut().zip(self.limbs.into_iter().zip(rhs.limbs)) {
            *out = lhs ^ rhs;
        }
        Self { limbs }
    }

    fn masked(mut self, bits: usize) -> Self {
        clear_above(&mut self.limbs, bits);
        self
    }

    fn window(self, shift: usize, bits: usize, mask: u64) -> usize {
        let word = shift / 64;
        let bit = shift % 64;
        let mut value = self.limbs.get(word).copied().unwrap_or(0) >> bit;

        if bit != 0 && bits > 64 - bit {
            value |= self.limbs.get(word + 1).copied().unwrap_or(0) << (64 - bit);
        }

        (value & mask) as usize
    }
}

type BinaryMul = fn(BinaryElement, BinaryElement) -> BinaryElement;

#[derive(Clone, Copy)]
struct BinaryMulBackend {
    name: &'static str,
    mul: BinaryMul,
}

impl BinaryMulBackend {
    fn detect(spec: &BinaryFieldSpec) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("aes") {
                let mul = match spec.name {
                    "b163" => mul_b163_pmull_checked,
                    "sect193" => mul_sect193_pmull_checked,
                    _ => unreachable!("unsupported binary field"),
                };
                return Self {
                    name: "aarch64-pmull-karatsuba",
                    mul,
                };
            }
        }

        let mul = match spec.name {
            "b163" => mul_b163_portable,
            "sect193" => mul_sect193_portable,
            _ => unreachable!("unsupported binary field"),
        };
        Self {
            name: "portable",
            mul,
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_b163_pmull_checked(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    // SAFETY: `BinaryMulBackend::detect` only selects this function after checking PMULL availability.
    unsafe { mul_b163_pmull(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
fn mul_b163_compact_pmull_checked(lhs: B163Element, rhs: B163Element) -> B163Element {
    // SAFETY: `B163MulBackend::detect` only selects this function after checking PMULL availability.
    unsafe { mul_b163_compact_pmull(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
fn mul_b191_compact_pmull_checked(lhs: B163Element, rhs: B163Element) -> B163Element {
    unsafe { mul_b191_compact_pmull(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
fn mul_sect193_pmull_checked(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    // SAFETY: `BinaryMulBackend::detect` only selects this function after checking PMULL availability.
    unsafe { mul_sect193_pmull(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn mul_b163_pmull(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    reduce_b163_product(product_3limb_pmull(lhs, rhs))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_b163_compact_pmull(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(reduce_b163_product(product_3limb_pmull(
        lhs.into_binary(),
        rhs.into_binary(),
    )))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_b191_compact_pmull(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(reduce_b191_product(product_3limb_pmull(
        lhs.into_binary(),
        rhs.into_binary(),
    )))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_sect193_pmull(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    let mut product = product_3limb_pmull(lhs, rhs);
    let lhs_top = lhs.limbs[3] & 1;
    let rhs_top = rhs.limbs[3] & 1;
    let lhs_top_mask = 0u64.wrapping_sub(lhs_top);
    let rhs_top_mask = 0u64.wrapping_sub(rhs_top);

    // Fold the products involving the one-bit fourth limbs into the 384-bit
    // product without unpredictable branches.
    product[3] ^= (rhs.limbs[0] & lhs_top_mask) ^ (lhs.limbs[0] & rhs_top_mask);
    product[4] ^= (rhs.limbs[1] & lhs_top_mask) ^ (lhs.limbs[1] & rhs_top_mask);
    product[5] ^= (rhs.limbs[2] & lhs_top_mask) ^ (lhs.limbs[2] & rhs_top_mask);
    product[6] ^= lhs_top & rhs_top;

    reduce_sect193_product(product)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
fn product_3limb_pmull(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    use std::arch::aarch64::vmull_p64;

    let a0 = lhs.limbs[0];
    let a1 = lhs.limbs[1];
    let a2 = lhs.limbs[2];
    let b0 = rhs.limbs[0];
    let b1 = rhs.limbs[1];
    let b2 = rhs.limbs[2];

    let z0 = vmull_p64(a0, b0);
    let z1 = vmull_p64(a1, b1);
    let z2 = vmull_p64(a2, b2);
    let c1 = vmull_p64(a0 ^ a1, b0 ^ b1) ^ z0 ^ z1;
    let c2 = vmull_p64(a0 ^ a2, b0 ^ b2) ^ z0 ^ z2 ^ z1;
    let c3 = vmull_p64(a1 ^ a2, b1 ^ b2) ^ z1 ^ z2;

    let mut product = [0u64; MAX_PRODUCT_LIMBS];
    xor_product(&mut product, 0, z0);
    xor_product(&mut product, 1, c1);
    xor_product(&mut product, 2, c2);
    xor_product(&mut product, 3, c3);
    xor_product(&mut product, 4, z2);
    product
}

fn clmul64_portable(lhs: u64, rhs: u64) -> u128 {
    let mut product = 0u128;
    let mut multiplier = rhs;

    while multiplier != 0 {
        let bit = multiplier.trailing_zeros();
        product ^= (lhs as u128) << bit;
        multiplier &= multiplier - 1;
    }

    product
}

fn mul_b163_portable(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    mul_binary_generic(lhs, rhs, &B163_FIELD, clmul64_portable)
}

fn mul_b163_compact_portable(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(mul_b163_portable(lhs.into_binary(), rhs.into_binary()))
}

fn mul_b191_compact_portable(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(mul_binary_generic(
        lhs.into_binary(),
        rhs.into_binary(),
        &B191_FIELD,
        clmul64_portable,
    ))
}

fn mul_sect193_portable(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    mul_binary_generic(lhs, rhs, &SECT193_FIELD, clmul64_portable)
}

fn mul_binary_generic(
    lhs: BinaryElement,
    rhs: BinaryElement,
    spec: &BinaryFieldSpec,
    clmul: fn(u64, u64) -> u128,
) -> BinaryElement {
    let mut product = [0u64; MAX_PRODUCT_LIMBS];
    let limbs = limb_count(spec.bits);

    for i in 0..limbs {
        for j in 0..limbs {
            xor_product(&mut product, i + j, clmul(lhs.limbs[i], rhs.limbs[j]));
        }
    }

    reduce_binary_product(product, spec)
}

fn xor_product(product: &mut [u64; MAX_PRODUCT_LIMBS], limb: usize, value: u128) {
    product[limb] ^= value as u64;
    product[limb + 1] ^= (value >> 64) as u64;
}

fn reduce_sect193_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
    // For p(u) = u^193 + u^15 + 1, write the product as L + u^193 H.
    // The first fold is L + H + u^15 H. It can reach degree 206, so its
    // at-most-14-bit carry is folded once more with the same relation.
    let h0 = (product[3] >> 1) | (product[4] << 63);
    let h1 = (product[4] >> 1) | (product[5] << 63);
    let h2 = (product[5] >> 1) | (product[6] << 63);

    let t0 = product[0] ^ h0 ^ (h0 << 15);
    let t1 = product[1] ^ h1 ^ (h0 >> 49) ^ (h1 << 15);
    let t2 = product[2] ^ h2 ^ (h1 >> 49) ^ (h2 << 15);
    let t3 = (product[3] & 1) ^ (h2 >> 49);
    let carry = t3 >> 1;

    BinaryElement {
        limbs: [t0 ^ carry ^ (carry << 15), t1, t2, t3 & 1],
    }
}

fn reduce_b163_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
    // For p(u) = u^163 + u^7 + u^6 + u^3 + 1, write the product as
    // L + u^163 H and fold H at offsets 0, 3, 6, and 7. The result can
    // reach degree 168, so one small second fold completes the reduction.
    const LOW_MASK: u64 = (1u64 << 35) - 1;

    let h0 = (product[2] >> 35) | (product[3] << 29);
    let h1 = (product[3] >> 35) | (product[4] << 29);
    let h2 = (product[4] >> 35) | (product[5] << 29);

    let t0 = product[0] ^ h0 ^ (h0 << 3) ^ (h0 << 6) ^ (h0 << 7);
    let t1 =
        product[1] ^ h1 ^ (h0 >> 61) ^ (h1 << 3) ^ (h0 >> 58) ^ (h1 << 6) ^ (h0 >> 57) ^ (h1 << 7);
    let t2 = (product[2] & LOW_MASK)
        ^ h2
        ^ (h1 >> 61)
        ^ (h2 << 3)
        ^ (h1 >> 58)
        ^ (h2 << 6)
        ^ (h1 >> 57)
        ^ (h2 << 7);
    let carry = t2 >> 35;

    BinaryElement {
        limbs: [
            t0 ^ carry ^ (carry << 3) ^ (carry << 6) ^ (carry << 7),
            t1,
            t2 & LOW_MASK,
            0,
        ],
    }
}

fn reduce_b191_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
    // u^191 = u^9 + 1. After the first fold, only the top eight bits of
    // u^9 H can cross degree 190, so one small second fold completes it.
    let h0 = (product[2] >> 63) | (product[3] << 1);
    let h1 = (product[3] >> 63) | (product[4] << 1);
    let h2 = (product[4] >> 63) | (product[5] << 1);
    let carry = h2 >> 54;

    BinaryElement {
        limbs: [
            product[0] ^ h0 ^ (h0 << 9) ^ carry ^ (carry << 9),
            product[1] ^ h1 ^ (h0 >> 55) ^ (h1 << 9),
            ((product[2] & (u64::MAX >> 1)) ^ h2 ^ (h1 >> 55) ^ (h2 << 9)) & (u64::MAX >> 1),
            0,
        ],
    }
}

fn reduce_binary_product(
    product: [u64; MAX_PRODUCT_LIMBS],
    spec: &BinaryFieldSpec,
) -> BinaryElement {
    let mut result = low_product(product, spec.bits);
    let high = shifted_product(product, spec.bits);

    for term in spec.terms {
        xor_shifted(&mut result, &high, *term);
    }

    let carry = shifted_limbs(result, spec.bits);
    clear_above(&mut result, spec.bits);

    for term in spec.terms {
        xor_shifted(&mut result, &carry, *term);
    }

    BinaryElement { limbs: result }.masked(spec.bits)
}

fn low_product(product: [u64; MAX_PRODUCT_LIMBS], bits: usize) -> [u64; MAX_FIELD_LIMBS] {
    let mut out = [0u64; MAX_FIELD_LIMBS];
    let limbs = limb_count(bits);
    out[..limbs].copy_from_slice(&product[..limbs]);
    clear_above(&mut out, bits);
    out
}

fn shifted_product(product: [u64; MAX_PRODUCT_LIMBS], shift: usize) -> [u64; MAX_FIELD_LIMBS] {
    let mut out = [0u64; MAX_FIELD_LIMBS];
    let word_shift = shift / 64;
    let bit_shift = shift % 64;

    for (i, limb) in out.iter_mut().enumerate() {
        let source = word_shift + i;
        let low = product.get(source).copied().unwrap_or(0) >> bit_shift;
        let high = if bit_shift == 0 {
            0
        } else {
            product.get(source + 1).copied().unwrap_or(0) << (64 - bit_shift)
        };
        *limb = low | high;
    }

    out
}

fn shifted_limbs(limbs: [u64; MAX_FIELD_LIMBS], shift: usize) -> [u64; MAX_FIELD_LIMBS] {
    let mut out = [0u64; MAX_FIELD_LIMBS];
    let word_shift = shift / 64;
    let bit_shift = shift % 64;

    for (i, limb) in out.iter_mut().enumerate() {
        let source = word_shift + i;
        let low = limbs.get(source).copied().unwrap_or(0) >> bit_shift;
        let high = if bit_shift == 0 {
            0
        } else {
            limbs.get(source + 1).copied().unwrap_or(0) << (64 - bit_shift)
        };
        *limb = low | high;
    }

    out
}

fn xor_shifted(target: &mut [u64; MAX_FIELD_LIMBS], source: &[u64; MAX_FIELD_LIMBS], shift: usize) {
    let word_shift = shift / 64;
    let bit_shift = shift % 64;

    for (i, limb) in source.iter().copied().enumerate() {
        if limb == 0 {
            continue;
        }

        let target_word = i + word_shift;
        if target_word < MAX_FIELD_LIMBS {
            target[target_word] ^= limb << bit_shift;
        }
        if bit_shift != 0 && target_word + 1 < MAX_FIELD_LIMBS {
            target[target_word + 1] ^= limb >> (64 - bit_shift);
        }
    }
}

fn clear_above(limbs: &mut [u64; MAX_FIELD_LIMBS], bits: usize) {
    let full_limbs = bits / 64;
    let top_bits = bits % 64;

    if top_bits != 0 {
        limbs[full_limbs] &= (1u64 << top_bits) - 1;
        for limb in limbs.iter_mut().skip(full_limbs + 1) {
            *limb = 0;
        }
    } else {
        for limb in limbs.iter_mut().skip(full_limbs) {
            *limb = 0;
        }
    }
}

fn limb_count(bits: usize) -> usize {
    bits.div_ceil(64)
}

fn format_binary_hex(value: BinaryElement, bits: usize) -> String {
    let hex_digits = bits.div_ceil(4);
    let mut full = String::with_capacity(MAX_FIELD_LIMBS * 16 + 2);
    full.push_str("0x");
    for limb in value.limbs.iter().rev() {
        full.push_str(&format!("{limb:016x}"));
    }
    let split = full.len() - hex_digits;
    format!("0x{}", &full[split..])
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GhashElement(u128);

impl GhashElement {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const GENERATOR: Self = Self(2);

    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }

    #[cfg(test)]
    fn square(self) -> Self {
        self.mul(self)
    }

    #[cfg(test)]
    fn mul(self, rhs: Self) -> Self {
        Self((detected_backend().mul)(self.0, rhs.0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Ghash2Element {
    c0: GhashElement,
    c1: GhashElement,
}

impl Ghash2Element {
    const ZERO: Self = Self {
        c0: GhashElement::ZERO,
        c1: GhashElement::ZERO,
    };
    const ONE: Self = Self {
        c0: GhashElement::ONE,
        c1: GhashElement::ZERO,
    };
    const GENERATOR: Self = Self {
        c0: GhashElement::ZERO,
        c1: GhashElement::ONE,
    };

    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0.add(rhs.c0),
            c1: self.c1.add(rhs.c1),
        }
    }

    fn mul(self, rhs: Self, mul: MulRaw) -> Self {
        let c0_product = GhashElement(mul(self.c0.0, rhs.c0.0));
        let c1_product = GhashElement(mul(self.c1.0, rhs.c1.0));
        let cross_plus_products = GhashElement(mul(self.c0.0 ^ self.c1.0, rhs.c0.0 ^ rhs.c1.0));

        Self {
            c0: c0_product.add(mul_by_delta(c1_product)),
            c1: cross_plus_products.add(c0_product),
        }
    }
}

fn mul_by_delta(value: GhashElement) -> GhashElement {
    let low = value.0 << GHASH2_DELTA_U_POWER;
    let high = value.0 >> (128 - GHASH2_DELTA_U_POWER);

    GhashElement(reduce_product(low, high))
}

fn format_ghash2_hex(value: Ghash2Element) -> String {
    format!("0x{:032x}{:032x}", value.c1.0, value.c0.0)
}

type MulRaw = fn(u128, u128) -> u128;

#[derive(Clone, Copy)]
struct Backend {
    name: &'static str,
    mul: MulRaw,
}

fn detected_backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(Backend::detect)
}

fn b127_backend() -> Backend {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return Backend {
            name: "aarch64-pmull",
            mul: mul_b127_pmull_checked,
        };
    }
    Backend {
        name: "portable",
        mul: mul_b127_portable,
    }
}

impl Backend {
    fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("aes") {
                return Self {
                    name: "aarch64-pmull",
                    mul: mul_raw_pmull_checked,
                };
            }
        }

        Self {
            name: "portable",
            mul: mul_raw_portable,
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_raw_pmull_checked(lhs: u128, rhs: u128) -> u128 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `Backend::detect` only selects this function after checking PMULL availability.
        unsafe { mul_raw_pmull(lhs, rhs) }
    }
}

#[cfg(target_arch = "aarch64")]
fn mul_b127_pmull_checked(lhs: u128, rhs: u128) -> u128 {
    unsafe { mul_b127_pmull(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_b127_pmull(lhs: u128, rhs: u128) -> u128 {
    use std::arch::aarch64::vmull_p64;
    let a0 = lhs as u64;
    let a1 = (lhs >> 64) as u64;
    let b0 = rhs as u64;
    let b1 = (rhs >> 64) as u64;
    let z0 = vmull_p64(a0, b0);
    let z2 = vmull_p64(a1, b1);
    let z1 = vmull_p64(a0 ^ a1, b0 ^ b1) ^ z0 ^ z2;
    reduce_b127_product(z0 ^ (z1 << 64), z2 ^ (z1 >> 64))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_raw_pmull(lhs: u128, rhs: u128) -> u128 {
    use std::arch::aarch64::vmull_p64;

    let lhs_low = lhs as u64;
    let lhs_high = (lhs >> 64) as u64;
    let rhs_low = rhs as u64;
    let rhs_high = (rhs >> 64) as u64;

    let product_low = vmull_p64(lhs_low, rhs_low);
    let product_high = vmull_p64(lhs_high, rhs_high);
    let product_mid =
        vmull_p64(lhs_low ^ lhs_high, rhs_low ^ rhs_high) ^ product_low ^ product_high;

    let low = product_low ^ (product_mid << 64);
    let high = product_high ^ (product_mid >> 64);

    reduce_product(low, high)
}

fn mul_raw_portable(lhs: u128, rhs: u128) -> u128 {
    let mut low = 0u128;
    let mut high = 0u128;
    let mut multiplier = rhs;

    while multiplier != 0 {
        let bit = multiplier.trailing_zeros();
        if bit < 128 {
            low ^= lhs << bit;
            if bit != 0 {
                high ^= lhs >> (128 - bit);
            }
        }
        multiplier &= multiplier - 1;
    }

    reduce_product(low, high)
}

fn mul_b127_portable(lhs: u128, rhs: u128) -> u128 {
    let result = mul_binary_generic(
        BinaryElement {
            limbs: [lhs as u64, (lhs >> 64) as u64, 0, 0],
        },
        BinaryElement {
            limbs: [rhs as u64, (rhs >> 64) as u64, 0, 0],
        },
        &B127_FIELD,
        clmul64_portable,
    );
    (result.limbs[0] as u128) | ((result.limbs[1] as u128) << 64)
}

fn reduce_b127_product(low: u128, high: u128) -> u128 {
    let h = (low >> 127) | (high << 1);
    (low & (u128::MAX >> 1)) ^ h ^ (h << 1)
}

fn reduce_product(low: u128, high: u128) -> u128 {
    let folded_low = low ^ high ^ (high << 1) ^ (high << 2) ^ (high << 7);
    let folded_high = (high >> 127) ^ (high >> 126) ^ (high >> 121);

    folded_low ^ folded_high ^ (folded_high << 1) ^ (folded_high << 2) ^ (folded_high << 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP_ORDER: u128 = u128::MAX;
    const PRIME_FACTORS: [u128; 9] = [
        3,
        5,
        17,
        257,
        641,
        65_537,
        274_177,
        6_700_417,
        67_280_421_310_721,
    ];
    const GHASH2_GROUP_ORDER_PRIME_FACTORS: [u128; 11] = [
        3,
        5,
        17,
        257,
        641,
        65_537,
        274_177,
        6_700_417,
        67_280_421_310_721,
        59_649_589_127_497_217,
        5_704_689_200_685_129_054_721,
    ];

    #[test]
    fn multiplication_matches_nist_algorithm() {
        let cases = [
            (0, 0),
            (1, 1),
            (2, 2),
            (0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321),
            (
                0xfedc_ba09_8765_4321_0123_4567_89ab_cdef,
                0x1357_9bdf_2468_ace0_f0e1_d2c3_b4a5_9687,
            ),
            (u128::MAX, 0x8000_0000_0000_0000_0000_0000_0000_0000),
        ];

        for (lhs, rhs) in cases {
            let lhs = GhashElement(lhs);
            let rhs = GhashElement(rhs);
            assert_eq!(lhs.mul(rhs), mul_reference(lhs, rhs));
        }
    }

    #[test]
    fn chosen_generator_is_primitive() {
        for factor in PRIME_FACTORS {
            let exponent = GROUP_ORDER / factor;
            assert_ne!(pow(GhashElement::GENERATOR, exponent), GhashElement::ONE);
        }

        assert_eq!(pow(GhashElement::GENERATOR, GROUP_ORDER), GhashElement::ONE);
    }

    #[test]
    fn ghash2_delta_has_trace_one() {
        assert_eq!(
            ghash_trace(GhashElement(1u128 << GHASH2_DELTA_U_POWER)),
            GhashElement::ONE
        );
    }

    #[test]
    fn ghash2_delta_multiply_matches_general_multiply() {
        let delta = GhashElement(1u128 << GHASH2_DELTA_U_POWER);

        for value in [
            GhashElement::ZERO,
            GhashElement::ONE,
            GhashElement::GENERATOR,
            GhashElement(0x1234_5678_90ab_cdef_fedc_ba09_8765_4321),
            GhashElement(u128::MAX),
        ] {
            assert_eq!(mul_by_delta(value), value.mul(delta));
        }
    }

    #[test]
    fn chosen_ghash2_generator_is_primitive() {
        for factor in GHASH2_GROUP_ORDER_PRIME_FACTORS {
            let exponent = ghash2_group_order_divisor(factor);
            assert_ne!(
                pow_ghash2(Ghash2Element::GENERATOR, exponent),
                Ghash2Element::ONE,
                "factor={factor}"
            );
        }

        assert_eq!(
            pow_ghash2(
                Ghash2Element::GENERATOR,
                binary_from_limbs([u64::MAX; MAX_FIELD_LIMBS]),
            ),
            Ghash2Element::ONE
        );
    }

    #[test]
    fn fixed_base_table_matches_square_and_multiply() {
        let table = FixedBaseTable::new(GhashElement::GENERATOR, 16, detected_backend());
        let exponents = [
            0,
            1,
            2,
            17,
            0xffff,
            0x1_0000,
            0x1234_5678_90ab_cdef_fedc_ba09_8765_4321,
            u128::MAX,
        ];

        for exponent in exponents {
            assert_eq!(table.pow(exponent), pow(GhashElement::GENERATOR, exponent));
        }
    }

    #[test]
    fn ghash2_fixed_base_table_matches_square_and_multiply() {
        for window_bits in [7, 11] {
            let table = Ghash2FixedBaseTable::new(
                Ghash2Element::GENERATOR,
                window_bits,
                detected_backend(),
            );
            for exponent in [
                BinaryElement::ZERO,
                BinaryElement::ONE,
                BinaryElement::U,
                binary_from_limbs([0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321, 0, 0]),
                binary_from_limbs([u64::MAX, u64::MAX, 0, 0]),
            ] {
                assert_eq!(
                    table.pow(exponent),
                    pow_ghash2(Ghash2Element::GENERATOR, exponent),
                    "window_bits={window_bits}"
                );
            }
        }
    }

    #[test]
    fn supported_window_sizes_match_square_and_multiply() {
        for window_bits in [1, 4, 8, 12, 14, 16, 18, 20] {
            let table =
                FixedBaseTable::new(GhashElement::GENERATOR, window_bits, detected_backend());
            for exponent in [
                0,
                1,
                2,
                17,
                0x1234_5678_90ab_cdef_fedc_ba09_8765_4321,
                u128::MAX,
            ] {
                assert_eq!(
                    table.pow(exponent),
                    pow(GhashElement::GENERATOR, exponent),
                    "window_bits={window_bits}"
                );
            }
        }
    }

    #[test]
    fn binary_field_multiplication_matches_reference() {
        for spec in [&B163_FIELD, &SECT193_FIELD] {
            let optimized_mul = BinaryMulBackend::detect(spec).mul;
            let cases = [
                (BinaryElement::ZERO, BinaryElement::ZERO),
                (BinaryElement::ONE, BinaryElement::ONE),
                (BinaryElement::U, BinaryElement::U),
                (
                    binary_from_limbs([0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321, 7, 0]),
                    binary_from_limbs([0x1357_9bdf_2468_ace0, 0xf0e1_d2c3_b4a5_9687, 11, 0]),
                ),
                (
                    binary_from_limbs([u64::MAX, u64::MAX, u64::MAX, u64::MAX]).masked(spec.bits),
                    binary_from_limbs([
                        0xaaaa_aaaa_aaaa_aaaa,
                        0x5555_5555_5555_5555,
                        0x0123_4567_89ab_cdef,
                        0x0000_0000_0000_0001,
                    ])
                    .masked(spec.bits),
                ),
            ];

            for (lhs, rhs) in cases {
                let lhs = lhs.masked(spec.bits);
                let rhs = rhs.masked(spec.bits);
                let expected = mul_binary_reference(lhs, rhs, spec);
                assert_eq!(
                    mul_binary_generic(lhs, rhs, spec, clmul64_portable),
                    expected,
                    "portable field={}",
                    spec.name
                );
                assert_eq!(
                    optimized_mul(lhs, rhs),
                    expected,
                    "optimized field={}",
                    spec.name
                );
                if spec.name == "b163" {
                    let compact_mul = B163MulBackend::detect().mul;
                    assert_eq!(
                        compact_mul(B163Element::from_binary(lhs), B163Element::from_binary(rhs))
                            .into_binary(),
                        expected,
                        "compact optimized field=b163"
                    );
                }
            }
        }
    }

    #[test]
    fn specialized_binary_reductions_match_generic_reduction() {
        let mut rng = StdRng::seed_from_u64(0x1930_1930_1930_1930);

        for _ in 0..100_000 {
            let mut product = [0u64; MAX_PRODUCT_LIMBS];
            for limb in &mut product[..6] {
                *limb = rng.next_u64();
            }
            product[6] = rng.next_u64() & 1;

            assert_eq!(
                reduce_sect193_product(product),
                reduce_binary_product(product, &SECT193_FIELD)
            );

            product[5] &= (1u64 << 61) - 1;
            product[6] = 0;
            assert_eq!(
                reduce_b191_product(product),
                reduce_binary_product(product, &B191_FIELD)
            );

            product[5] &= 0x1f;
            assert_eq!(
                reduce_b163_product(product),
                reduce_binary_product(product, &B163_FIELD)
            );
        }
    }

    #[test]
    fn b191_multiplication_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x1910_1910_1910_1910);
        let mul = b191_backend().mul;
        for _ in 0..10_000 {
            let lhs = B163Element {
                limbs: [rng.next_u64(), rng.next_u64(), rng.next_u64() >> 1],
            };
            let rhs = B163Element {
                limbs: [rng.next_u64(), rng.next_u64(), rng.next_u64() >> 1],
            };
            assert_eq!(
                mul(lhs, rhs).into_binary(),
                mul_binary_reference(lhs.into_binary(), rhs.into_binary(), &B191_FIELD)
            );
        }
    }

    #[test]
    fn b127_multiplication_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x1270_1270_1270_1270);
        let mul = b127_backend().mul;
        for _ in 0..10_000 {
            let lhs =
                ((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)) & (u128::MAX >> 1);
            let rhs =
                ((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)) & (u128::MAX >> 1);
            let expected = mul_binary_reference(
                BinaryElement {
                    limbs: [lhs as u64, (lhs >> 64) as u64, 0, 0],
                },
                BinaryElement {
                    limbs: [rhs as u64, (rhs >> 64) as u64, 0, 0],
                },
                &B127_FIELD,
            );
            assert_eq!(
                mul(lhs, rhs),
                (expected.limbs[0] as u128) | ((expected.limbs[1] as u128) << 64)
            );
        }
    }

    #[test]
    fn binary_field_tables_match_reference_powering() {
        for spec in [&B163_FIELD, &SECT193_FIELD] {
            let backend = BinaryMulBackend::detect(spec);
            for window_bits in [11, 15] {
                let table = BinaryFixedBaseTable::new(spec, window_bits, backend);
                for exponent in [
                    BinaryElement::ZERO,
                    BinaryElement::ONE,
                    BinaryElement::U,
                    binary_from_limbs([0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321, 17, 0])
                        .masked(EXPONENT_BITS),
                    binary_from_limbs([u64::MAX, u64::MAX, u64::MAX, u64::MAX])
                        .masked(EXPONENT_BITS),
                ] {
                    assert_eq!(
                        table.pow(exponent),
                        pow_binary_reference(BinaryElement::U.masked(spec.bits), exponent, spec),
                        "field={}, window_bits={window_bits}",
                        spec.name
                    );
                }
            }
        }

        let backend = B163MulBackend::detect();
        for window_bits in [11, 15] {
            let table = B163FixedBaseTable::new(window_bits, backend);
            for exponent in [
                BinaryElement::ZERO,
                BinaryElement::ONE,
                BinaryElement::U,
                binary_from_limbs([0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321, 0, 0]),
                binary_from_limbs([u64::MAX, u64::MAX, 0, 0]),
            ] {
                assert_eq!(
                    table.pow(exponent).into_binary(),
                    pow_binary_reference(
                        BinaryElement::U.masked(B163_FIELD.bits),
                        exponent,
                        &B163_FIELD
                    ),
                    "compact field=b163, window_bits={window_bits}"
                );
            }
        }

        let backend = b191_backend();
        for window_bits in [11, 15] {
            let table = B163FixedBaseTable::new(window_bits, backend);
            for exponent in [
                BinaryElement::ZERO,
                BinaryElement::ONE,
                BinaryElement::U,
                binary_from_limbs([0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321, 0, 0]),
                binary_from_limbs([u64::MAX, u64::MAX, 0, 0]),
            ] {
                assert_eq!(
                    table.pow(exponent).into_binary(),
                    pow_binary_reference(BinaryElement::U, exponent, &B191_FIELD),
                    "compact field=b191, window_bits={window_bits}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn specialized_pmull_batch_kernels_match_table_powering() {
        if !std::arch::is_aarch64_feature_detected!("aes") {
            return;
        }

        let exponents_u128 = [0, 1, 2, 17, u128::MAX];
        let exponents_binary = exponents_u128.map(|value| BinaryElement {
            limbs: [value as u64, (value >> 64) as u64, 0, 0],
        });

        let ghash_table = FixedBaseTable::new(GhashElement::GENERATOR, 11, detected_backend());
        let mut ghash_outputs = [GhashElement::ZERO; 5];
        compute_random_powers_pmull_checked(&ghash_table, &exponents_u128, &mut ghash_outputs);
        for (output, exponent) in ghash_outputs.into_iter().zip(exponents_u128) {
            assert_eq!(output, ghash_table.pow(exponent));
        }

        let b127_table = FixedBaseTable::new(GhashElement::GENERATOR, 11, b127_backend());
        let mut b127_outputs = [GhashElement::ZERO; 5];
        compute_b127_random_powers_pmull_checked(&b127_table, &exponents_u128, &mut b127_outputs);
        for (output, exponent) in b127_outputs.into_iter().zip(exponents_u128) {
            assert_eq!(output, b127_table.pow(exponent));
        }

        let ghash2_table =
            Ghash2FixedBaseTable::new(Ghash2Element::GENERATOR, 11, detected_backend());
        let mut ghash2_outputs = [Ghash2Element::ZERO; 5];
        compute_ghash2_random_powers_pmull_checked(
            &ghash2_table,
            &exponents_binary,
            &mut ghash2_outputs,
        );
        for (output, exponent) in ghash2_outputs.into_iter().zip(exponents_binary) {
            assert_eq!(output, ghash2_table.pow(exponent));
        }

        let b163_table = B163FixedBaseTable::new(11, B163MulBackend::detect());
        let mut b163_outputs = [B163Element::ZERO; 5];
        compute_b163_random_powers_pmull_checked(&b163_table, &exponents_binary, &mut b163_outputs);
        for (output, exponent) in b163_outputs.into_iter().zip(exponents_binary) {
            assert_eq!(output, b163_table.pow(exponent));
        }

        let b191_table = B163FixedBaseTable::new(11, b191_backend());
        let mut b191_outputs = [B163Element::ZERO; 5];
        compute_b191_random_powers_pmull_checked(&b191_table, &exponents_binary, &mut b191_outputs);
        for (output, exponent) in b191_outputs.into_iter().zip(exponents_binary) {
            assert_eq!(output, b191_table.pow(exponent));
        }

        let sect193_backend = BinaryMulBackend::detect(&SECT193_FIELD);
        let sect193_table = BinaryFixedBaseTable::new(&SECT193_FIELD, 11, sect193_backend);
        let mut sect193_outputs = [BinaryElement::ZERO; 5];
        compute_sect193_random_powers_pmull_checked(
            &sect193_table,
            &exponents_binary,
            &mut sect193_outputs,
        );
        for (output, exponent) in sect193_outputs.into_iter().zip(exponents_binary) {
            assert_eq!(output, sect193_table.pow(exponent));
        }
    }

    fn pow(mut base: GhashElement, mut exponent: u128) -> GhashElement {
        let mut acc = GhashElement::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                acc = acc.mul(base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square();
            }
        }
        acc
    }

    fn pow_ghash2(mut base: Ghash2Element, exponent: BinaryElement) -> Ghash2Element {
        let mut acc = Ghash2Element::ONE;
        let mul = detected_backend().mul;

        for bit in 0..GHASH2_BITS {
            if bit_is_set(&exponent.limbs, bit) {
                acc = acc.mul(base, mul);
            }
            base = base.mul(base, mul);
        }

        acc
    }

    fn ghash_trace(mut value: GhashElement) -> GhashElement {
        let mut trace = GhashElement::ZERO;

        for _ in 0..128 {
            trace = trace.add(value);
            value = value.square();
        }

        trace
    }

    fn ghash2_group_order_divisor(divisor: u128) -> BinaryElement {
        let mut quotient = [0u64; MAX_FIELD_LIMBS];
        let mut remainder = 0u128;

        for bit in (0..GHASH2_BITS).rev() {
            remainder = (remainder << 1) | 1;
            if remainder >= divisor {
                remainder -= divisor;
                flip_bit(&mut quotient, bit);
            }
        }

        assert_eq!(remainder, 0);
        BinaryElement { limbs: quotient }
    }

    fn mul_reference(lhs: GhashElement, rhs: GhashElement) -> GhashElement {
        let mut product = 0u128;
        let mut base = lhs.0;
        let mut multiplier = rhs.0;

        while multiplier != 0 {
            if multiplier & 1 != 0 {
                product ^= base;
            }
            multiplier >>= 1;
            base = mul_by_u(base);
        }

        GhashElement(product)
    }

    fn mul_by_u(value: u128) -> u128 {
        let carry = value >> 127;
        let shifted = value << 1;
        if carry == 0 {
            shifted
        } else {
            shifted ^ 0b1000_0111
        }
    }

    fn binary_from_limbs(limbs: [u64; MAX_FIELD_LIMBS]) -> BinaryElement {
        BinaryElement { limbs }
    }

    fn pow_binary_reference(
        mut base: BinaryElement,
        exponent: BinaryElement,
        spec: &BinaryFieldSpec,
    ) -> BinaryElement {
        let mut acc = BinaryElement::ONE;

        for bit in 0..spec.bits {
            if bit_is_set(&exponent.limbs, bit) {
                acc = mul_binary_reference(acc, base, spec);
            }
            base = mul_binary_reference(base, base, spec);
        }

        acc
    }

    fn mul_binary_reference(
        lhs: BinaryElement,
        rhs: BinaryElement,
        spec: &BinaryFieldSpec,
    ) -> BinaryElement {
        let mut product = [0u64; MAX_PRODUCT_LIMBS];

        for bit in 0..spec.bits {
            if bit_is_set(&rhs.limbs, bit) {
                xor_shifted_product_reference(&mut product, &lhs.limbs, bit);
            }
        }

        reduce_product_reference(product, spec)
    }

    fn xor_shifted_product_reference(
        product: &mut [u64; MAX_PRODUCT_LIMBS],
        source: &[u64; MAX_FIELD_LIMBS],
        shift: usize,
    ) {
        let word_shift = shift / 64;
        let bit_shift = shift % 64;

        for (i, limb) in source.iter().copied().enumerate() {
            if limb == 0 {
                continue;
            }

            let target_word = i + word_shift;
            if target_word < MAX_PRODUCT_LIMBS {
                product[target_word] ^= limb << bit_shift;
            }
            if bit_shift != 0 && target_word + 1 < MAX_PRODUCT_LIMBS {
                product[target_word + 1] ^= limb >> (64 - bit_shift);
            }
        }
    }

    fn reduce_product_reference(
        mut product: [u64; MAX_PRODUCT_LIMBS],
        spec: &BinaryFieldSpec,
    ) -> BinaryElement {
        for bit in (spec.bits..(spec.bits * 2)).rev() {
            if bit_is_set(&product, bit) {
                flip_bit(&mut product, bit);
                for term in spec.terms {
                    flip_bit(&mut product, bit - spec.bits + *term);
                }
            }
        }

        BinaryElement {
            limbs: low_product(product, spec.bits),
        }
    }

    fn bit_is_set<const N: usize>(limbs: &[u64; N], bit: usize) -> bool {
        let word = bit / 64;
        let offset = bit % 64;
        limbs
            .get(word)
            .is_some_and(|limb| ((limb >> offset) & 1) != 0)
    }

    fn flip_bit<const N: usize>(limbs: &mut [u64; N], bit: usize) {
        let word = bit / 64;
        let offset = bit % 64;
        limbs[word] ^= 1u64 << offset;
    }
}
