use rand::{RngCore, SeedableRng, rngs::StdRng};
use std::env;

mod generic;
#[cfg(test)]
mod iso_root;
mod isomorphism;

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
const DEFAULT_B256_WINDOW_BITS: usize = 13;
const MAX_WINDOW_BITS: usize = 20;
const MAX_FIELD_LIMBS: usize = 4;
const MAX_PRODUCT_LIMBS: usize = MAX_FIELD_LIMBS * 2;
const EXPONENT_BITS: usize = 128;
const GHASH2_DELTA_U_POWER: usize = 121;
#[cfg(test)]
const GHASH2_BITS: usize = 256;

fn main() {
    let config = Config::parse_or_exit();

    if config.isomorphism {
        isomorphism::run(config);
        return;
    }

    if config.multiply {
        match config.field {
            FieldChoice::B127 => generic::run_multiply::<generic::B127>(config),
            FieldChoice::Ghash128 => generic::run_multiply::<generic::Ghash128>(config),
            FieldChoice::Ghash2 => generic::run_multiply::<generic::Ghash2>(config),
            FieldChoice::B163 => generic::run_multiply::<generic::B163>(config),
            FieldChoice::B191 => generic::run_multiply::<generic::B191>(config),
            FieldChoice::Sect193 => generic::run_multiply::<generic::Sect193>(config),
            FieldChoice::B256 => generic::run_multiply::<generic::B256>(config),
        }
        return;
    }

    match config.field {
        FieldChoice::B127 => generic::run_power::<generic::B127>(config),
        FieldChoice::Ghash128 => generic::run_power::<generic::Ghash128>(config),
        FieldChoice::Ghash2 => generic::run_power::<generic::Ghash2>(config),
        FieldChoice::B163 => generic::run_power::<generic::B163>(config),
        FieldChoice::B191 => generic::run_power::<generic::B191>(config),
        FieldChoice::Sect193 => generic::run_power::<generic::Sect193>(config),
        FieldChoice::B256 => generic::run_power::<generic::B256>(config),
    }
}

fn random_exponents(rng: &mut StdRng, len: usize) -> Vec<u128> {
    (0..len)
        .map(|_| (rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64))
        .collect()
}

fn print_mul_header() {
    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>14} {:>14}",
        "log2(n)", "n", "best_ms", "ns/mul", "checksum"
    );
}

struct Config {
    field: FieldChoice,
    multiply: bool,
    isomorphism: bool,
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
                    "usage: cargo run --release --bin ghash-powers-bench -- [--field b127|ghash128|ghash2|b163|b191|sect193|b256] [--mul|--isomorphism] [--min-log N] [--max-log N] [--samples N] [--window-bits N] [--seed N]"
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
            isomorphism: false,
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
                    return Err(String::from("Binary-field benchmark"));
                }
                "--mul" => config.multiply = true,
                "--isomorphism" | "--iso" => config.isomorphism = true,
                "--field" => config.field = parse_field(&next_arg(&mut args, "--field")?)?,
                "--min-log" => config.min_log = parse_next(&mut args, "--min-log")?,
                "--max-log" => config.max_log = parse_next(&mut args, "--max-log")?,
                "--samples" => config.samples = parse_next(&mut args, "--samples")?,
                "--window-bits" => {
                    config.window_bits = parse_next(&mut args, "--window-bits")?;
                    window_bits_was_set = true;
                }
                "--seed" => config.seed = parse_next(&mut args, "--seed")?,
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
        if config.multiply && config.isomorphism {
            return Err(String::from(
                "--mul and --isomorphism are mutually exclusive",
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
    B256,
}

fn default_window_bits(field: FieldChoice) -> usize {
    match field {
        FieldChoice::B127 => DEFAULT_B127_WINDOW_BITS,
        FieldChoice::Ghash128 => DEFAULT_GHASH128_WINDOW_BITS,
        FieldChoice::Ghash2 => DEFAULT_GHASH2_WINDOW_BITS,
        FieldChoice::B163 => DEFAULT_B163_WINDOW_BITS,
        FieldChoice::B191 => DEFAULT_B191_WINDOW_BITS,
        FieldChoice::Sect193 => DEFAULT_SECT193_WINDOW_BITS,
        FieldChoice::B256 => DEFAULT_B256_WINDOW_BITS,
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
        "b256" | "binary256" => Ok(FieldChoice::B256),
        _ => Err(format!(
            "unknown field {value:?}; expected b127, ghash128, ghash2, b163, b191, sect193, or b256"
        )),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_next<T: ParseInteger>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T, String> {
    parse_value(&next_arg(args, name)?, name)
}

fn parse_value<T: ParseInteger>(value: &str, name: &str) -> Result<T, String> {
    T::parse_integer(value).map_err(|message| format!("invalid {name} value {value:?}: {message}"))
}

trait ParseInteger: Sized {
    fn parse_integer(value: &str) -> Result<Self, String>;
}

macro_rules! impl_parse_integer {
    ($type:ty) => {
        impl ParseInteger for $type {
            fn parse_integer(value: &str) -> Result<Self, String> {
                parse_u128(value).and_then(|value| {
                    Self::try_from(value)
                        .map_err(|_| format!("value does not fit in {}", stringify!($type)))
                })
            }
        }
    };
}

impl_parse_integer!(u32);
impl_parse_integer!(usize);
impl_parse_integer!(u64);

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BinaryElement {
    limbs: [u64; MAX_FIELD_LIMBS],
}

impl BinaryElement {
    const ZERO: Self = Self {
        limbs: [0; MAX_FIELD_LIMBS],
    };
    const ONE: Self = Self {
        limbs: [1, 0, 0, 0],
    };
    const U: Self = Self {
        limbs: [2, 0, 0, 0],
    };

    fn add(self, rhs: Self) -> Self {
        let mut limbs = [0u64; MAX_FIELD_LIMBS];
        for (out, (lhs, rhs)) in limbs.iter_mut().zip(self.limbs.into_iter().zip(rhs.limbs)) {
            *out = lhs ^ rhs;
        }
        Self { limbs }
    }
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

    fn mul(self, rhs: Self, mul: fn(u128, u128) -> u128) -> Self {
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
    GhashElement(reduce_ghash_product(low, high))
}

fn format_ghash2_hex(value: Ghash2Element) -> String {
    format!("0x{:032x}{:032x}", value.c1.0, value.c0.0)
}

fn format_binary_hex(value: BinaryElement, bits: usize) -> String {
    let hex_digits = bits.div_ceil(4);
    let mut full = String::with_capacity(MAX_FIELD_LIMBS * 16);
    for limb in value.limbs.iter().rev() {
        full.push_str(&format!("{limb:016x}"));
    }
    format!("0x{}", &full[full.len() - hex_digits..])
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

fn xor_product(product: &mut [u64; MAX_PRODUCT_LIMBS], limb: usize, value: u128) {
    product[limb] ^= value as u64;
    product[limb + 1] ^= (value >> 64) as u64;
}

fn product_3limb_portable(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    let mut product = [0u64; MAX_PRODUCT_LIMBS];
    for i in 0..3 {
        for j in 0..3 {
            xor_product(
                &mut product,
                i + j,
                clmul64_portable(lhs.limbs[i], rhs.limbs[j]),
            );
        }
    }
    product
}

fn product_4limb_portable(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    let mut product = [0u64; MAX_PRODUCT_LIMBS];
    for i in 0..MAX_FIELD_LIMBS {
        for j in 0..MAX_FIELD_LIMBS {
            xor_product(
                &mut product,
                i + j,
                clmul64_portable(lhs.limbs[i], rhs.limbs[j]),
            );
        }
    }
    product
}

fn product_sect193_portable(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    let mut product = product_3limb_portable(lhs, rhs);
    fold_sect193_top_products(&mut product, lhs, rhs);
    product
}

fn fold_sect193_top_products(
    product: &mut [u64; MAX_PRODUCT_LIMBS],
    lhs: BinaryElement,
    rhs: BinaryElement,
) {
    let lhs_top = lhs.limbs[3] & 1;
    let rhs_top = rhs.limbs[3] & 1;
    let lhs_top_mask = 0u64.wrapping_sub(lhs_top);
    let rhs_top_mask = 0u64.wrapping_sub(rhs_top);

    product[3] ^= (rhs.limbs[0] & lhs_top_mask) ^ (lhs.limbs[0] & rhs_top_mask);
    product[4] ^= (rhs.limbs[1] & lhs_top_mask) ^ (lhs.limbs[1] & rhs_top_mask);
    product[5] ^= (rhs.limbs[2] & lhs_top_mask) ^ (lhs.limbs[2] & rhs_top_mask);
    product[6] ^= lhs_top & rhs_top;
}

fn product_2limb_portable(lhs: u128, rhs: u128) -> (u128, u128) {
    let mut low = 0u128;
    let mut high = 0u128;
    let mut multiplier = rhs;
    while multiplier != 0 {
        let bit = multiplier.trailing_zeros();
        low ^= lhs << bit;
        if bit != 0 {
            high ^= lhs >> (128 - bit);
        }
        multiplier &= multiplier - 1;
    }
    (low, high)
}

fn mul_raw_portable(lhs: u128, rhs: u128) -> u128 {
    let (low, high) = product_2limb_portable(lhs, rhs);
    reduce_ghash_product(low, high)
}

fn mul_b127_portable(lhs: u128, rhs: u128) -> u128 {
    let (low, high) = product_2limb_portable(lhs, rhs);
    reduce_b127_product(low, high)
}

fn mul_b163_compact_portable(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(reduce_b163_product(product_3limb_portable(
        lhs.into_binary(),
        rhs.into_binary(),
    )))
}

fn mul_b191_compact_portable(lhs: B163Element, rhs: B163Element) -> B163Element {
    B163Element::from_binary(reduce_b191_product(product_3limb_portable(
        lhs.into_binary(),
        rhs.into_binary(),
    )))
}

fn mul_sect193_portable(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    reduce_sect193_product(product_sect193_portable(lhs, rhs))
}

fn mul_b256_portable(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    reduce_b256_product(product_4limb_portable(lhs, rhs))
}

fn reduce_b127_product(low: u128, high: u128) -> u128 {
    let h = (low >> 127) | (high << 1);
    (low & (u128::MAX >> 1)) ^ h ^ (h << 1)
}

fn reduce_ghash_product(low: u128, high: u128) -> u128 {
    let folded_low = low ^ high ^ (high << 1) ^ (high << 2) ^ (high << 7);
    let folded_high = (high >> 127) ^ (high >> 126) ^ (high >> 121);
    folded_low ^ folded_high ^ (folded_high << 1) ^ (folded_high << 2) ^ (folded_high << 7)
}

fn reduce_b163_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
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

fn reduce_sect193_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
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

fn reduce_b256_product(product: [u64; MAX_PRODUCT_LIMBS]) -> BinaryElement {
    // For p(u) = u^256 + u^10 + u^5 + u^2 + 1, fold the high half
    // at offsets 0, 2, 5, and 10. Only nine carry bits remain, so one
    // small second fold completes the reduction.
    let [h0, h1, h2, h3] = [product[4], product[5], product[6], product[7]];
    let t0 = product[0] ^ h0 ^ (h0 << 2) ^ (h0 << 5) ^ (h0 << 10);
    let t1 =
        product[1] ^ h1 ^ (h0 >> 62) ^ (h1 << 2) ^ (h0 >> 59) ^ (h1 << 5) ^ (h0 >> 54) ^ (h1 << 10);
    let t2 =
        product[2] ^ h2 ^ (h1 >> 62) ^ (h2 << 2) ^ (h1 >> 59) ^ (h2 << 5) ^ (h1 >> 54) ^ (h2 << 10);
    let t3 =
        product[3] ^ h3 ^ (h2 >> 62) ^ (h3 << 2) ^ (h2 >> 59) ^ (h3 << 5) ^ (h2 >> 54) ^ (h3 << 10);
    let carry = (h3 >> 62) ^ (h3 >> 59) ^ (h3 >> 54);

    BinaryElement {
        limbs: [
            t0 ^ carry ^ (carry << 2) ^ (carry << 5) ^ (carry << 10),
            t1,
            t2,
            t3,
        ],
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
    fold_sect193_top_products(&mut product, lhs, rhs);
    reduce_sect193_product(product)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_b256_pmull(lhs: BinaryElement, rhs: BinaryElement) -> BinaryElement {
    reduce_b256_product(product_4limb_pmull(lhs, rhs))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
fn product_3limb_pmull(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    use std::arch::aarch64::vmull_p64;

    let [a0, a1, a2, _] = lhs.limbs;
    let [b0, b1, b2, _] = rhs.limbs;
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
fn product_4limb_pmull(lhs: BinaryElement, rhs: BinaryElement) -> [u64; MAX_PRODUCT_LIMBS] {
    let lhs_low = lhs.limbs[0] as u128 | (lhs.limbs[1] as u128) << 64;
    let lhs_high = lhs.limbs[2] as u128 | (lhs.limbs[3] as u128) << 64;
    let rhs_low = rhs.limbs[0] as u128 | (rhs.limbs[1] as u128) << 64;
    let rhs_high = rhs.limbs[2] as u128 | (rhs.limbs[3] as u128) << 64;

    let (low0, high0) = product_2limb_pmull(lhs_low, rhs_low);
    let (low2, high2) = product_2limb_pmull(lhs_high, rhs_high);
    let (low1, high1) = product_2limb_pmull(lhs_low ^ lhs_high, rhs_low ^ rhs_high);
    let cross_low = low1 ^ low0 ^ low2;
    let cross_high = high1 ^ high0 ^ high2;

    let mut product = [0u64; MAX_PRODUCT_LIMBS];
    xor_product(&mut product, 0, low0);
    xor_product(&mut product, 2, high0);
    xor_product(&mut product, 2, cross_low);
    xor_product(&mut product, 4, cross_high);
    xor_product(&mut product, 4, low2);
    xor_product(&mut product, 6, high2);
    product
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_b127_pmull(lhs: u128, rhs: u128) -> u128 {
    let (low, high) = product_2limb_pmull(lhs, rhs);
    reduce_b127_product(low, high)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_raw_pmull(lhs: u128, rhs: u128) -> u128 {
    let (low, high) = product_2limb_pmull(lhs, rhs);
    reduce_ghash_product(low, high)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
fn product_2limb_pmull(lhs: u128, rhs: u128) -> (u128, u128) {
    use std::arch::aarch64::vmull_p64;

    let lhs_low = lhs as u64;
    let lhs_high = (lhs >> 64) as u64;
    let rhs_low = rhs as u64;
    let rhs_high = (rhs >> 64) as u64;
    let product_low = vmull_p64(lhs_low, rhs_low);
    let product_high = vmull_p64(lhs_high, rhs_high);
    let product_mid =
        vmull_p64(lhs_low ^ lhs_high, rhs_low ^ rhs_high) ^ product_low ^ product_high;

    (
        product_low ^ (product_mid << 64),
        product_high ^ (product_mid >> 64),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use generic::{Field, FixedBaseTable};

    const GHASH_GROUP_ORDER: u128 = u128::MAX;
    const GHASH_PRIME_FACTORS: [u128; 9] = [
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

    struct RefField {
        bits: usize,
        terms: &'static [usize],
    }

    const B127_REF: RefField = RefField {
        bits: 127,
        terms: &[0, 1],
    };
    const B163_REF: RefField = RefField {
        bits: 163,
        terms: &[0, 3, 6, 7],
    };
    const B191_REF: RefField = RefField {
        bits: 191,
        terms: &[0, 9],
    };
    const SECT193_REF: RefField = RefField {
        bits: 193,
        terms: &[0, 15],
    };
    const B256_REF: RefField = RefField {
        bits: 256,
        terms: &[0, 2, 5, 10],
    };

    #[test]
    fn ghash_multiplication_matches_bitwise_reference() {
        let cases = [
            (0, 0),
            (1, 1),
            (2, 2),
            (0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321),
            (
                0xfedc_ba09_8765_4321_0123_4567_89ab_cdef,
                0x1357_9bdf_2468_ace0_f0e1_d2c3_b4a5_9687,
            ),
            (u128::MAX, 1u128 << 127),
        ];
        for (lhs, rhs) in cases {
            assert_eq!(mul_raw_portable(lhs, rhs), mul_ghash_reference(lhs, rhs));
        }
    }

    #[test]
    fn specialized_binary_multiplication_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x6269_6e61_7279_7265);
        for _ in 0..10_000 {
            let lhs127 = random_binary(&mut rng, B127_REF.bits);
            let rhs127 = random_binary(&mut rng, B127_REF.bits);
            let lhs_u128 = lhs127.limbs[0] as u128 | (lhs127.limbs[1] as u128) << 64;
            let rhs_u128 = rhs127.limbs[0] as u128 | (rhs127.limbs[1] as u128) << 64;
            assert_eq!(
                binary_from_u128(mul_b127_portable(lhs_u128, rhs_u128)),
                mul_binary_reference(lhs127, rhs127, &B127_REF)
            );

            let lhs163 = random_binary(&mut rng, B163_REF.bits);
            let rhs163 = random_binary(&mut rng, B163_REF.bits);
            assert_eq!(
                mul_b163_compact_portable(
                    B163Element::from_binary(lhs163),
                    B163Element::from_binary(rhs163)
                )
                .into_binary(),
                mul_binary_reference(lhs163, rhs163, &B163_REF)
            );

            let lhs191 = random_binary(&mut rng, B191_REF.bits);
            let rhs191 = random_binary(&mut rng, B191_REF.bits);
            assert_eq!(
                mul_b191_compact_portable(
                    B163Element::from_binary(lhs191),
                    B163Element::from_binary(rhs191)
                )
                .into_binary(),
                mul_binary_reference(lhs191, rhs191, &B191_REF)
            );

            let lhs193 = random_binary(&mut rng, SECT193_REF.bits);
            let rhs193 = random_binary(&mut rng, SECT193_REF.bits);
            assert_eq!(
                mul_sect193_portable(lhs193, rhs193),
                mul_binary_reference(lhs193, rhs193, &SECT193_REF)
            );

            let lhs256 = random_binary(&mut rng, B256_REF.bits);
            let rhs256 = random_binary(&mut rng, B256_REF.bits);
            assert_eq!(
                mul_b256_portable(lhs256, rhs256),
                mul_binary_reference(lhs256, rhs256, &B256_REF)
            );
        }
    }

    #[test]
    fn specialized_reducers_match_reference() {
        let mut rng = StdRng::seed_from_u64(0x7265_6475_6365_7273);
        for _ in 0..100_000 {
            let lhs163 = random_binary(&mut rng, B163_REF.bits);
            let rhs163 = random_binary(&mut rng, B163_REF.bits);
            assert_eq!(
                reduce_b163_product(product_3limb_portable(lhs163, rhs163)),
                mul_binary_reference(lhs163, rhs163, &B163_REF)
            );

            let lhs191 = random_binary(&mut rng, B191_REF.bits);
            let rhs191 = random_binary(&mut rng, B191_REF.bits);
            assert_eq!(
                reduce_b191_product(product_3limb_portable(lhs191, rhs191)),
                mul_binary_reference(lhs191, rhs191, &B191_REF)
            );

            let lhs193 = random_binary(&mut rng, SECT193_REF.bits);
            let rhs193 = random_binary(&mut rng, SECT193_REF.bits);
            assert_eq!(
                reduce_sect193_product(product_sect193_portable(lhs193, rhs193)),
                mul_binary_reference(lhs193, rhs193, &SECT193_REF)
            );

            let lhs256 = random_binary(&mut rng, B256_REF.bits);
            let rhs256 = random_binary(&mut rng, B256_REF.bits);
            assert_eq!(
                reduce_b256_product(product_4limb_portable(lhs256, rhs256)),
                mul_binary_reference(lhs256, rhs256, &B256_REF)
            );
        }
    }

    #[test]
    fn b256_modulus_is_irreducible() {
        // Rabin's criterion. Since 256 has only the prime divisor 2, it is
        // enough to check gcd(x^(2^128) - x, f) = 1 and x^(2^256) = x.
        let mut frobenius = BinaryElement::U;
        for _ in 0..128 {
            frobenius = mul_b256_portable(frobenius, frobenius);
        }

        let modulus = [0x425, 0, 0, 0, 1];
        let mut difference = [0u64; 5];
        difference[..MAX_FIELD_LIMBS].copy_from_slice(&frobenius.add(BinaryElement::U).limbs);
        assert_eq!(polynomial_gcd(modulus, difference), [1, 0, 0, 0, 0]);

        for _ in 0..128 {
            frobenius = mul_b256_portable(frobenius, frobenius);
        }
        assert_eq!(frobenius, BinaryElement::U);
    }

    #[test]
    fn generic_tables_match_square_and_multiply() {
        check_generic_table::<generic::B127>();
        check_generic_table::<generic::Ghash128>();
        check_generic_table::<generic::Ghash2>();
        check_generic_table::<generic::B163>();
        check_generic_table::<generic::B191>();
        check_generic_table::<generic::Sect193>();
        check_generic_table::<generic::B256>();
    }

    #[test]
    fn ghash_generator_is_primitive() {
        for factor in GHASH_PRIME_FACTORS {
            assert_ne!(
                pow_ghash(GhashElement::GENERATOR, GHASH_GROUP_ORDER / factor),
                GhashElement::ONE
            );
        }
        assert_eq!(
            pow_ghash(GhashElement::GENERATOR, GHASH_GROUP_ORDER),
            GhashElement::ONE
        );
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
            assert_eq!(
                mul_by_delta(value),
                GhashElement(mul_raw_portable(value.0, delta.0))
            );
        }
    }

    #[test]
    fn ghash2_generator_is_primitive() {
        for factor in GHASH2_GROUP_ORDER_PRIME_FACTORS {
            assert_ne!(
                pow_ghash2(Ghash2Element::GENERATOR, ghash2_group_order_divisor(factor)),
                Ghash2Element::ONE,
                "factor={factor}"
            );
        }
        assert_eq!(
            pow_ghash2(
                Ghash2Element::GENERATOR,
                BinaryElement {
                    limbs: [u64::MAX; MAX_FIELD_LIMBS]
                }
            ),
            Ghash2Element::ONE
        );
    }

    fn check_generic_table<F: Field>()
    where
        F::Elem: std::fmt::Debug + Eq,
    {
        for window_bits in [1, 4, 8, 11, 16, 20] {
            let table = FixedBaseTable::<F>::new(window_bits);
            for exponent in [
                0,
                1,
                2,
                17,
                0x1234_5678_90ab_cdef_fedc_ba09_8765_4321,
                u128::MAX,
            ] {
                assert_eq!(
                    table.pow_portable(exponent),
                    pow_field::<F>(F::generator(), exponent),
                    "field={}, window_bits={window_bits}",
                    F::NAME
                );
            }
        }
    }

    fn pow_field<F: Field>(mut base: F::Elem, mut exponent: u128) -> F::Elem {
        let mut acc = F::one();
        while exponent != 0 {
            if exponent & 1 != 0 {
                acc = F::mul_portable(acc, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = F::mul_portable(base, base);
            }
        }
        acc
    }

    fn pow_ghash(mut base: GhashElement, mut exponent: u128) -> GhashElement {
        let mut acc = GhashElement::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                acc = GhashElement(mul_raw_portable(acc.0, base.0));
            }
            exponent >>= 1;
            if exponent != 0 {
                base = GhashElement(mul_raw_portable(base.0, base.0));
            }
        }
        acc
    }

    fn pow_ghash2(mut base: Ghash2Element, exponent: BinaryElement) -> Ghash2Element {
        let mut acc = Ghash2Element::ONE;
        for bit in 0..GHASH2_BITS {
            if bit_is_set(&exponent.limbs, bit) {
                acc = acc.mul(base, mul_raw_portable);
            }
            base = base.mul(base, mul_raw_portable);
        }
        acc
    }

    fn ghash_trace(mut value: GhashElement) -> GhashElement {
        let mut trace = GhashElement::ZERO;
        for _ in 0..128 {
            trace = trace.add(value);
            value = GhashElement(mul_raw_portable(value.0, value.0));
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

    fn mul_ghash_reference(lhs: u128, rhs: u128) -> u128 {
        let mut product = 0u128;
        let mut base = lhs;
        let mut multiplier = rhs;
        while multiplier != 0 {
            if multiplier & 1 != 0 {
                product ^= base;
            }
            multiplier >>= 1;
            let carry = base >> 127;
            base <<= 1;
            if carry != 0 {
                base ^= 0x87;
            }
        }
        product
    }

    fn random_binary(rng: &mut StdRng, bits: usize) -> BinaryElement {
        let mut limbs = [0u64; MAX_FIELD_LIMBS];
        for limb in limbs.iter_mut().take(bits.div_ceil(64)) {
            *limb = rng.next_u64();
        }
        mask_binary(BinaryElement { limbs }, bits)
    }

    fn binary_from_u128(value: u128) -> BinaryElement {
        BinaryElement {
            limbs: [value as u64, (value >> 64) as u64, 0, 0],
        }
    }

    fn mask_binary(mut value: BinaryElement, bits: usize) -> BinaryElement {
        let whole_limbs = bits / 64;
        let top_bits = bits % 64;
        if top_bits != 0 {
            value.limbs[whole_limbs] &= (1u64 << top_bits) - 1;
            value.limbs[whole_limbs + 1..].fill(0);
        } else {
            value.limbs[whole_limbs..].fill(0);
        }
        value
    }

    fn mul_binary_reference(
        lhs: BinaryElement,
        rhs: BinaryElement,
        field: &RefField,
    ) -> BinaryElement {
        let mut product = [0u64; MAX_PRODUCT_LIMBS];
        for bit in 0..field.bits {
            if bit_is_set(&rhs.limbs, bit) {
                xor_shifted_reference(&mut product, &lhs.limbs, bit);
            }
        }
        for bit in (field.bits..field.bits * 2).rev() {
            if bit_is_set(&product, bit) {
                flip_bit(&mut product, bit);
                for term in field.terms {
                    flip_bit(&mut product, bit - field.bits + term);
                }
            }
        }
        BinaryElement {
            limbs: [product[0], product[1], product[2], product[3]],
        }
    }

    fn xor_shifted_reference(
        product: &mut [u64; MAX_PRODUCT_LIMBS],
        source: &[u64; MAX_FIELD_LIMBS],
        shift: usize,
    ) {
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        for (i, limb) in source.iter().copied().enumerate() {
            let target = i + word_shift;
            if target < MAX_PRODUCT_LIMBS {
                product[target] ^= limb << bit_shift;
            }
            if bit_shift != 0 && target + 1 < MAX_PRODUCT_LIMBS {
                product[target + 1] ^= limb >> (64 - bit_shift);
            }
        }
    }

    fn bit_is_set<const N: usize>(limbs: &[u64; N], bit: usize) -> bool {
        limbs
            .get(bit / 64)
            .is_some_and(|limb| (limb >> (bit % 64)) & 1 != 0)
    }

    fn flip_bit<const N: usize>(limbs: &mut [u64; N], bit: usize) {
        limbs[bit / 64] ^= 1u64 << (bit % 64);
    }

    fn polynomial_gcd(mut lhs: [u64; 5], mut rhs: [u64; 5]) -> [u64; 5] {
        while rhs != [0; 5] {
            let remainder = polynomial_remainder(lhs, rhs);
            lhs = rhs;
            rhs = remainder;
        }
        lhs
    }

    fn polynomial_remainder(mut dividend: [u64; 5], divisor: [u64; 5]) -> [u64; 5] {
        let divisor_degree = polynomial_degree(&divisor).expect("nonzero divisor");
        while let Some(dividend_degree) = polynomial_degree(&dividend) {
            if dividend_degree < divisor_degree {
                break;
            }
            xor_shifted_polynomial(&mut dividend, &divisor, dividend_degree - divisor_degree);
        }
        dividend
    }

    fn polynomial_degree(polynomial: &[u64; 5]) -> Option<usize> {
        polynomial
            .iter()
            .rposition(|limb| *limb != 0)
            .map(|word| word * 64 + (63 - polynomial[word].leading_zeros() as usize))
    }

    fn xor_shifted_polynomial(target: &mut [u64; 5], source: &[u64; 5], shift: usize) {
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        for (source_word, limb) in source.iter().copied().enumerate() {
            let target_word = source_word + word_shift;
            if target_word < target.len() {
                target[target_word] ^= limb << bit_shift;
            }
            if bit_shift != 0 && target_word + 1 < target.len() {
                target[target_word + 1] ^= limb >> (64 - bit_shift);
            }
        }
    }
}
