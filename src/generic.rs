use super::*;
use std::{
    hint::black_box,
    marker::PhantomData,
    mem::size_of,
    time::{Duration, Instant},
};

pub(crate) trait Field: Sized + 'static {
    type Elem: Copy;

    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    const BITS: usize;

    fn zero() -> Self::Elem;
    fn one() -> Self::Elem;
    fn generator() -> Self::Elem;
    fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem;
    fn random(rng: &mut StdRng) -> Self::Elem;
    fn format(value: Self::Elem) -> String;
    fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem;

    #[cfg(target_arch = "aarch64")]
    unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem;
}

pub(crate) struct B127;
pub(crate) struct Ghash128;
pub(crate) struct Ghash2;
pub(crate) struct B163;
pub(crate) struct B191;
pub(crate) struct Sect193;
pub(crate) struct B256;

macro_rules! impl_u128_field {
    ($field:ty, $name:literal, $description:literal, $bits:expr, $portable:path, $pmull:path, $mask:expr) => {
        impl Field for $field {
            type Elem = GhashElement;

            const NAME: &'static str = $name;
            const DESCRIPTION: &'static str = $description;
            const BITS: usize = $bits;

            #[inline(always)]
            fn zero() -> Self::Elem {
                GhashElement::ZERO
            }

            #[inline(always)]
            fn one() -> Self::Elem {
                GhashElement::ONE
            }

            #[inline(always)]
            fn generator() -> Self::Elem {
                GhashElement::GENERATOR
            }

            #[inline(always)]
            fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                lhs.add(rhs)
            }

            fn random(rng: &mut StdRng) -> Self::Elem {
                let value = (rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64);
                GhashElement(value & $mask)
            }

            fn format(value: Self::Elem) -> String {
                format!("0x{:032x}", value.0)
            }

            #[inline(always)]
            fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                GhashElement($portable(lhs.0, rhs.0))
            }

            #[cfg(target_arch = "aarch64")]
            #[inline(always)]
            unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                GhashElement(unsafe { $pmull(lhs.0, rhs.0) })
            }
        }
    };
}

impl_u128_field!(
    B127,
    "b127",
    "GF(2^127) / (u^127 + u + 1)",
    127,
    mul_b127_portable,
    mul_b127_pmull,
    u128::MAX >> 1
);

impl_u128_field!(
    Ghash128,
    "ghash128",
    "GF(2^128) / (u^128 + u^7 + u^2 + u + 1)",
    128,
    mul_raw_portable,
    mul_raw_pmull,
    u128::MAX
);

macro_rules! impl_three_limb_field {
    ($field:ty, $name:literal, $description:literal, $bits:expr, $top_mask:expr, $portable:path, $pmull:path) => {
        impl Field for $field {
            type Elem = B163Element;

            const NAME: &'static str = $name;
            const DESCRIPTION: &'static str = $description;
            const BITS: usize = $bits;

            #[inline(always)]
            fn zero() -> Self::Elem {
                B163Element::ZERO
            }

            #[inline(always)]
            fn one() -> Self::Elem {
                B163Element::ONE
            }

            #[inline(always)]
            fn generator() -> Self::Elem {
                B163Element::U
            }

            #[inline(always)]
            fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                lhs.add(rhs)
            }

            fn random(rng: &mut StdRng) -> Self::Elem {
                B163Element {
                    limbs: [rng.next_u64(), rng.next_u64(), rng.next_u64() & $top_mask],
                }
            }

            fn format(value: Self::Elem) -> String {
                format_binary_hex(value.into_binary(), Self::BITS)
            }

            #[inline(always)]
            fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                $portable(lhs, rhs)
            }

            #[cfg(target_arch = "aarch64")]
            #[inline(always)]
            unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
                unsafe { $pmull(lhs, rhs) }
            }
        }
    };
}

impl_three_limb_field!(
    B163,
    "b163",
    "GF(2^163) / (u^163 + u^7 + u^6 + u^3 + 1)",
    163,
    (1u64 << 35) - 1,
    mul_b163_compact_portable,
    mul_b163_compact_pmull
);

impl_three_limb_field!(
    B191,
    "b191",
    "GF(2^191) / (u^191 + u^9 + 1)",
    191,
    u64::MAX >> 1,
    mul_b191_compact_portable,
    mul_b191_compact_pmull
);

impl Field for Sect193 {
    type Elem = BinaryElement;

    const NAME: &'static str = "sect193";
    const DESCRIPTION: &'static str = "GF(2^193) / (u^193 + u^15 + 1)";
    const BITS: usize = 193;

    #[inline(always)]
    fn zero() -> Self::Elem {
        BinaryElement::ZERO
    }

    #[inline(always)]
    fn one() -> Self::Elem {
        BinaryElement::ONE
    }

    #[inline(always)]
    fn generator() -> Self::Elem {
        BinaryElement::U
    }

    #[inline(always)]
    fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        lhs.add(rhs)
    }

    fn random(rng: &mut StdRng) -> Self::Elem {
        BinaryElement {
            limbs: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64() & 1,
            ],
        }
    }

    fn format(value: Self::Elem) -> String {
        format_binary_hex(value, Self::BITS)
    }

    #[inline(always)]
    fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        mul_sect193_portable(lhs, rhs)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        unsafe { mul_sect193_pmull(lhs, rhs) }
    }
}

impl Field for B256 {
    type Elem = BinaryElement;

    const NAME: &'static str = "b256";
    const DESCRIPTION: &'static str = "GF(2^256) / (u^256 + u^10 + u^5 + u^2 + 1)";
    const BITS: usize = 256;

    #[inline(always)]
    fn zero() -> Self::Elem {
        BinaryElement::ZERO
    }

    #[inline(always)]
    fn one() -> Self::Elem {
        BinaryElement::ONE
    }

    #[inline(always)]
    fn generator() -> Self::Elem {
        BinaryElement::U
    }

    #[inline(always)]
    fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        lhs.add(rhs)
    }

    fn random(rng: &mut StdRng) -> Self::Elem {
        BinaryElement {
            limbs: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ],
        }
    }

    fn format(value: Self::Elem) -> String {
        format_binary_hex(value, Self::BITS)
    }

    #[inline(always)]
    fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        mul_b256_portable(lhs, rhs)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        unsafe { mul_b256_pmull(lhs, rhs) }
    }
}

impl Field for Ghash2 {
    type Elem = Ghash2Element;

    const NAME: &'static str = "ghash2";
    const DESCRIPTION: &'static str = "K[v] / (v^2 + v + u^121), K = GF(2^128) GHASH";
    const BITS: usize = 256;

    #[inline(always)]
    fn zero() -> Self::Elem {
        Ghash2Element::ZERO
    }

    #[inline(always)]
    fn one() -> Self::Elem {
        Ghash2Element::ONE
    }

    #[inline(always)]
    fn generator() -> Self::Elem {
        Ghash2Element::GENERATOR
    }

    #[inline(always)]
    fn add(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        lhs.add(rhs)
    }

    fn random(rng: &mut StdRng) -> Self::Elem {
        Ghash2Element {
            c0: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
            c1: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
        }
    }

    fn format(value: Self::Elem) -> String {
        format_ghash2_hex(value)
    }

    #[inline(always)]
    fn mul_portable(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        lhs.mul(rhs, mul_raw_portable)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn mul_pmull(lhs: Self::Elem, rhs: Self::Elem) -> Self::Elem {
        unsafe { mul_ghash2_pmull(lhs, rhs) }
    }
}

pub(crate) struct FixedBaseTable<F: Field> {
    table: Vec<F::Elem>,
    window_bits: usize,
    window_count: usize,
    window_mask: u128,
    window_size: usize,
    _field: PhantomData<F>,
}

impl<F: Field> FixedBaseTable<F> {
    pub(crate) fn new(window_bits: usize) -> Self {
        assert!((1..=MAX_WINDOW_BITS).contains(&window_bits));
        let window_count = EXPONENT_BITS.div_ceil(window_bits);
        let window_size = 1usize << window_bits;
        let window_mask = (1u128 << window_bits) - 1;
        let mut table = vec![F::one(); window_count * window_size];
        let mut window_base = F::generator();
        let mul = selected_mul::<F>();

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
            window_bits,
            window_count,
            window_mask,
            window_size,
            _field: PhantomData,
        }
    }

    pub(crate) fn table_bytes(&self) -> usize {
        self.table.len() * size_of::<F::Elem>()
    }

    #[cfg(test)]
    pub(crate) fn pow_portable(&self, exponent: u128) -> F::Elem {
        pow_one::<F>(self, exponent, F::mul_portable)
    }
}

type Mul<F> = fn(<F as Field>::Elem, <F as Field>::Elem) -> <F as Field>::Elem;

fn selected_mul<F: Field>() -> Mul<F> {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return mul_pmull_checked::<F>;
    }
    F::mul_portable
}

#[cfg(target_arch = "aarch64")]
fn mul_pmull_checked<F: Field>(lhs: F::Elem, rhs: F::Elem) -> F::Elem {
    unsafe { mul_pmull::<F>(lhs, rhs) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn mul_pmull<F: Field>(lhs: F::Elem, rhs: F::Elem) -> F::Elem {
    unsafe { F::mul_pmull(lhs, rhs) }
}

fn pow_one<F: Field>(table: &FixedBaseTable<F>, exponent: u128, mul: Mul<F>) -> F::Elem {
    let mut acc = F::one();
    for window in 0..table.window_count {
        let shift = window * table.window_bits;
        let value = ((exponent >> shift) & table.window_mask) as usize;
        if value != 0 {
            acc = mul(acc, table.table[window * table.window_size + value]);
        }
    }
    acc
}

fn compute_powers_portable<F: Field>(
    table: &FixedBaseTable<F>,
    exponents: &[u128],
    outputs: &mut [F::Elem],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        *output = pow_one::<F>(table, exponent, F::mul_portable);
    }
}

#[cfg(target_arch = "aarch64")]
fn compute_powers_pmull_checked<F: Field>(
    table: &FixedBaseTable<F>,
    exponents: &[u128],
    outputs: &mut [F::Elem],
) {
    unsafe { compute_powers_pmull::<F>(table, exponents, outputs) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn compute_powers_pmull<F: Field>(
    table: &FixedBaseTable<F>,
    exponents: &[u128],
    outputs: &mut [F::Elem],
) {
    assert_eq!(exponents.len(), outputs.len());
    for (exponent, output) in exponents.iter().copied().zip(outputs) {
        let mut acc = F::one();
        for window in 0..table.window_count {
            let shift = window * table.window_bits;
            let value = ((exponent >> shift) & table.window_mask) as usize;
            if value != 0 {
                acc = unsafe { F::mul_pmull(acc, table.table[window * table.window_size + value]) };
            }
        }
        *output = acc;
    }
}

fn multiply_batch_portable<F: Field>(lhs: &[F::Elem], rhs: &[F::Elem], outputs: &mut [F::Elem]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), outputs.len());
    for ((lhs, rhs), output) in lhs.iter().copied().zip(rhs.iter().copied()).zip(outputs) {
        *output = F::mul_portable(lhs, rhs);
    }
}

#[cfg(target_arch = "aarch64")]
fn multiply_batch_pmull_checked<F: Field>(
    lhs: &[F::Elem],
    rhs: &[F::Elem],
    outputs: &mut [F::Elem],
) {
    unsafe { multiply_batch_pmull::<F>(lhs, rhs, outputs) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn multiply_batch_pmull<F: Field>(
    lhs: &[F::Elem],
    rhs: &[F::Elem],
    outputs: &mut [F::Elem],
) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), outputs.len());
    for ((lhs, rhs), output) in lhs.iter().copied().zip(rhs.iter().copied()).zip(outputs) {
        *output = unsafe { F::mul_pmull(lhs, rhs) };
    }
}

type Compute<F> = fn(&FixedBaseTable<F>, &[u128], &mut [<F as Field>::Elem]);
type Multiply<F> = fn(&[<F as Field>::Elem], &[<F as Field>::Elem], &mut [<F as Field>::Elem]);

fn selected_compute<F: Field>() -> Compute<F> {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return compute_powers_pmull_checked::<F>;
    }
    compute_powers_portable::<F>
}

fn selected_multiply<F: Field>() -> Multiply<F> {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return multiply_batch_pmull_checked::<F>;
    }
    multiply_batch_portable::<F>
}

pub(crate) fn run_power<F: Field>(config: Config) {
    let max_batch = 1usize << config.max_log;
    println!("Generic binary-field random fixed-base powers benchmark");
    println!("field: {} ({})", F::NAME, F::DESCRIPTION);
    println!("base: u");
    println!("exponents: random values in [0, 2^{EXPONENT_BITS})");
    println!("multiplication backend: {}", backend_name());
    println!(
        "batch logs: {}..={}, samples: {}, window bits: {}, seed: 0x{:016x}",
        config.min_log, config.max_log, config.samples, config.window_bits, config.seed
    );

    let table = FixedBaseTable::<F>::new(config.window_bits);
    println!(
        "precompute table: {} windows, {:.1} MiB",
        table.window_count,
        table.table_bytes() as f64 / (1024.0 * 1024.0)
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let exponents = random_exponents(&mut rng, max_batch);
    let mut outputs = vec![F::zero(); max_batch];
    let compute = selected_compute::<F>();

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/elem", "checksum"
    );
    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = F::zero();
        for _ in 0..config.samples {
            let start = Instant::now();
            compute(
                black_box(&table),
                black_box(&exponents[..batch]),
                black_box(&mut outputs[..batch]),
            );
            best = best.min(start.elapsed());
            checksum = outputs[..batch].iter().copied().fold(F::zero(), F::add);
            black_box(checksum);
        }
        println!(
            "{log:>8} {batch:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / batch as f64,
            F::format(checksum)
        );
    }
}

pub(crate) fn run_multiply<F: Field>(config: Config) {
    let max_batch = 1usize << config.max_log;
    println!("Generic full-field multiplication benchmark");
    println!("field: {} ({})", F::NAME, F::DESCRIPTION);
    println!("multiplication backend: {}", backend_name());
    let mut rng = StdRng::seed_from_u64(config.seed);
    let lhs: Vec<_> = (0..max_batch).map(|_| F::random(&mut rng)).collect();
    let rhs: Vec<_> = (0..max_batch).map(|_| F::random(&mut rng)).collect();
    let mut outputs = vec![F::zero(); max_batch];
    let multiply = selected_multiply::<F>();

    print_mul_header();
    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        let mut best = Duration::MAX;
        let mut checksum = F::zero();
        for _ in 0..config.samples {
            let start = Instant::now();
            multiply(
                black_box(&lhs[..batch]),
                black_box(&rhs[..batch]),
                black_box(&mut outputs[..batch]),
            );
            best = best.min(start.elapsed());
            checksum = outputs[..batch].iter().copied().fold(F::zero(), F::add);
            black_box(checksum);
        }
        println!(
            "{log:>8} {batch:>12} {:>14.3} {:>14.3} {}",
            best.as_secs_f64() * 1e3,
            best.as_secs_f64() * 1e9 / batch as f64,
            F::format(checksum)
        );
    }
}

fn backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return "aarch64-pmull (monomorphized)";
    }
    "portable (monomorphized)"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_field<F: Field>()
    where
        F::Elem: std::fmt::Debug + Eq,
    {
        let mut rng = StdRng::seed_from_u64(0x6765_6e65_7269_6321 ^ F::BITS as u64);
        for _ in 0..1_000 {
            let lhs = F::random(&mut rng);
            let rhs = F::random(&mut rng);
            let expected = F::mul_portable(lhs, rhs);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("aes") {
                assert_eq!(unsafe { mul_pmull::<F>(lhs, rhs) }, expected, "{}", F::NAME);
            }
        }

        let table = FixedBaseTable::<F>::new(11);
        for exponent in [0, 1, 2, 17, u128::MAX] {
            let expected = table.pow_portable(exponent);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("aes") {
                let mut output = [F::zero()];
                compute_powers_pmull_checked::<F>(&table, &[exponent], &mut output);
                assert_eq!(output[0], expected, "{} exponent={exponent}", F::NAME);
            }
        }
    }

    #[test]
    fn all_generic_fields_match_portable_arithmetic() {
        check_field::<B127>();
        check_field::<Ghash128>();
        check_field::<Ghash2>();
        check_field::<B163>();
        check_field::<B191>();
        check_field::<Sect193>();
        check_field::<B256>();
    }
}
