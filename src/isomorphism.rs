use super::*;
use std::{
    hint::black_box,
    sync::OnceLock,
    time::{Duration, Instant},
};

type Bits256 = [u64; 4];
type Columns = [Bits256; 256];
type ConvertBatch = fn(&[BinaryElement], &mut [BinaryElement]);

const B256_ROOT_IN_GHASH2: Ghash2Element = Ghash2Element {
    c0: GhashElement(0x5b7c_0058_bb5d_8781_f467_4693_f5e2_9589),
    c1: GhashElement(0x00f9_d800_78d5_ce4c_90c5_d6be_218c_fa5d),
};

struct Tables {
    forward: Box<[Bits256]>,
    inverse: Box<[Bits256]>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

pub(crate) fn run(config: Config) {
    let max_batch = 1usize << config.max_log;
    let _ = tables();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let inputs: Vec<_> = (0..max_batch)
        .map(|_| BinaryElement {
            limbs: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ],
        })
        .collect();
    let mut outputs = vec![BinaryElement::ZERO; max_batch];

    println!("b256 <-> ghash2 field-isomorphism benchmark");
    println!("representations: 256-bit direct quotient <-> quadratic GHASH extension");
    println!("table setup and input generation: outside timed region");
    println!("samples: {}, seed: 0x{:016x}", config.samples, config.seed);
    println!();
    println!(
        "{:>8} {:>22} {:>14} {:>14} {:>66}",
        "log2(n)", "operation", "best_ms", "ns/elem", "checksum"
    );

    let operations: [(&str, ConvertBatch); 2] = [
        ("b256->ghash2", forward_batch),
        ("ghash2->b256", inverse_batch),
    ];

    for log in config.min_log..=config.max_log {
        let batch = 1usize << log;
        for (name, convert) in operations {
            let mut best = Duration::MAX;
            let mut checksum = BinaryElement::ZERO;
            for _ in 0..config.samples {
                let start = Instant::now();
                convert(
                    black_box(&inputs[..batch]),
                    black_box(&mut outputs[..batch]),
                );
                best = best.min(start.elapsed());
                checksum = outputs[..batch]
                    .iter()
                    .copied()
                    .fold(BinaryElement::ZERO, BinaryElement::add);
                black_box(checksum);
            }
            println!(
                "{log:>8} {name:>22} {:>14.3} {:>14.3} {}",
                best.as_secs_f64() * 1e3,
                best.as_secs_f64() * 1e9 / batch as f64,
                format_binary_hex(checksum, 256)
            );
        }
    }
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(Tables::new)
}

impl Tables {
    fn new() -> Self {
        let forward = forward_columns();
        let inverse = invert_columns(&forward);
        Self {
            forward: grouped_table(&forward, 5),
            inverse: grouped_table(&inverse, 5),
        }
    }
}

fn forward_columns() -> Columns {
    let mut columns = [[0u64; 4]; 256];
    let mut power = Ghash2Element::ONE;
    for column in &mut columns {
        *column = ghash2_bits(power);
        power = power.mul(B256_ROOT_IN_GHASH2, mul_raw_portable);
    }
    columns
}

fn invert_columns(columns: &Columns) -> Columns {
    let mut rows = vec![[0u64; 8]; 256];
    for (column_index, column) in columns.iter().enumerate() {
        for row_index in 0..256 {
            if column[row_index / 64] & (1u64 << (row_index % 64)) != 0 {
                rows[row_index][column_index / 64] |= 1u64 << (column_index % 64);
            }
        }
    }
    for (row_index, row) in rows.iter_mut().enumerate() {
        row[4 + row_index / 64] = 1u64 << (row_index % 64);
    }

    for pivot in 0..256 {
        let pivot_row = (pivot..256)
            .find(|row| rows[*row][pivot / 64] & (1u64 << (pivot % 64)) != 0)
            .expect("isomorphism matrix must be invertible");
        rows.swap(pivot, pivot_row);
        let pivot_value = rows[pivot];
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index != pivot && row[pivot / 64] & (1u64 << (pivot % 64)) != 0 {
                for (word, pivot_word) in row.iter_mut().zip(pivot_value) {
                    *word ^= pivot_word;
                }
            }
        }
    }

    let mut inverse = [[0u64; 4]; 256];
    for input_bit in 0..256 {
        for output_bit in 0..256 {
            if rows[output_bit][4 + input_bit / 64] & (1u64 << (input_bit % 64)) != 0 {
                inverse[input_bit][output_bit / 64] |= 1u64 << (output_bit % 64);
            }
        }
    }
    inverse
}

fn grouped_table(columns: &Columns, width: usize) -> Box<[Bits256]> {
    let group_count = 256usize.div_ceil(width);
    let group_size = 1usize << width;
    let mut table = vec![[0u64; 4]; group_count * group_size];
    for group in 0..group_count {
        let offset = group * group_size;
        for value in 1..group_size {
            let bit = value.trailing_zeros() as usize;
            let previous = value & (value - 1);
            let column_index = group * width + bit;
            table[offset + value] = if column_index < 256 {
                xor_bits(table[offset + previous], columns[column_index])
            } else {
                table[offset + previous]
            };
        }
    }
    table.into_boxed_slice()
}

#[inline(always)]
fn xor_bits(lhs: Bits256, rhs: Bits256) -> Bits256 {
    [
        lhs[0] ^ rhs[0],
        lhs[1] ^ rhs[1],
        lhs[2] ^ rhs[2],
        lhs[3] ^ rhs[3],
    ]
}

fn forward_batch(inputs: &[BinaryElement], outputs: &mut [BinaryElement]) {
    convert_5bit_batch(inputs, outputs, &tables().forward);
}

fn inverse_batch(inputs: &[BinaryElement], outputs: &mut [BinaryElement]) {
    convert_5bit_batch(inputs, outputs, &tables().inverse);
}

fn convert_5bit_batch(inputs: &[BinaryElement], outputs: &mut [BinaryElement], table: &[Bits256]) {
    assert_eq!(inputs.len(), outputs.len());
    for (input, output) in inputs.iter().copied().zip(outputs) {
        *output = apply_5bit(input, table);
    }
}

#[inline(always)]
fn apply_5bit(input: BinaryElement, table: &[Bits256]) -> BinaryElement {
    let low = input.limbs[0] as u128 | (input.limbs[1] as u128) << 64;
    let high = input.limbs[2] as u128 | (input.limbs[3] as u128) << 64;
    let mut accumulator0 = [0u64; 4];
    let mut accumulator1 = [0u64; 4];
    let mut accumulator2 = [0u64; 4];
    let mut accumulator3 = [0u64; 4];

    for group in (0..24).step_by(4) {
        accumulator0 = xor_bits(
            accumulator0,
            table_5bit(table, group, (low >> (group * 5)) as usize),
        );
        accumulator1 = xor_bits(
            accumulator1,
            table_5bit(table, group + 1, (low >> ((group + 1) * 5)) as usize),
        );
        accumulator2 = xor_bits(
            accumulator2,
            table_5bit(table, group + 2, (low >> ((group + 2) * 5)) as usize),
        );
        accumulator3 = xor_bits(
            accumulator3,
            table_5bit(table, group + 3, (low >> ((group + 3) * 5)) as usize),
        );
    }

    accumulator0 = xor_bits(accumulator0, table_5bit(table, 24, (low >> 120) as usize));
    accumulator1 = xor_bits(
        accumulator1,
        table_5bit(table, 25, ((low >> 125) | (high << 3)) as usize),
    );

    for group in (26..50).step_by(4) {
        let shift = group * 5 - 128;
        accumulator0 = xor_bits(
            accumulator0,
            table_5bit(table, group, (high >> shift) as usize),
        );
        accumulator1 = xor_bits(
            accumulator1,
            table_5bit(table, group + 1, (high >> (shift + 5)) as usize),
        );
        accumulator2 = xor_bits(
            accumulator2,
            table_5bit(table, group + 2, (high >> (shift + 10)) as usize),
        );
        accumulator3 = xor_bits(
            accumulator3,
            table_5bit(table, group + 3, (high >> (shift + 15)) as usize),
        );
    }
    accumulator2 = xor_bits(accumulator2, table_5bit(table, 50, (high >> 122) as usize));
    accumulator3 = xor_bits(accumulator3, table_5bit(table, 51, (high >> 127) as usize));

    BinaryElement {
        limbs: xor_bits(
            xor_bits(accumulator0, accumulator1),
            xor_bits(accumulator2, accumulator3),
        ),
    }
}

#[inline(always)]
fn table_5bit(table: &[Bits256], group: usize, value: usize) -> Bits256 {
    // SAFETY: the table contains 52 complete 32-entry groups. Callers mask
    // implicitly by converting only the low five bits of each shifted value.
    unsafe { *table.get_unchecked(group * 32 + (value & 31)) }
}

fn ghash2_bits(value: Ghash2Element) -> Bits256 {
    [
        value.c0.0 as u64,
        (value.c0.0 >> 64) as u64,
        value.c1.0 as u64,
        (value.c1.0 >> 64) as u64,
    ]
}

#[cfg(test)]
fn bits_ghash2(value: BinaryElement) -> Ghash2Element {
    Ghash2Element {
        c0: GhashElement(value.limbs[0] as u128 | (value.limbs[1] as u128) << 64),
        c1: GhashElement(value.limbs[2] as u128 | (value.limbs[3] as u128) << 64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chosen_root_satisfies_b256_modulus() {
        let alpha = B256_ROOT_IN_GHASH2;
        let mut powers = [Ghash2Element::ONE; 257];
        for index in 1..powers.len() {
            powers[index] = powers[index - 1].mul(alpha, mul_raw_portable);
        }
        let value = [0, 2, 5, 10, 256]
            .into_iter()
            .fold(Ghash2Element::ZERO, |sum, degree| sum.add(powers[degree]));
        assert_eq!(value, Ghash2Element::ZERO);
    }

    #[test]
    fn tables_are_inverse_field_isomorphisms() {
        let tables = tables();
        let forward = forward_columns();
        for bit in 0..256 {
            let mut basis = BinaryElement::ZERO;
            basis.limbs[bit / 64] = 1u64 << (bit % 64);
            assert_eq!(
                apply_5bit(apply_5bit(basis, &tables.forward), &tables.inverse),
                basis
            );
            assert_eq!(
                apply_columns(basis, &forward),
                apply_5bit(basis, &tables.forward)
            );
        }

        let mut rng = StdRng::seed_from_u64(0x6973_6f5f_7465_7374);
        for _ in 0..10_000 {
            let lhs = BinaryElement {
                limbs: [
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ],
            };
            let rhs = BinaryElement {
                limbs: [
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ],
            };
            let lhs_image = apply_5bit(lhs, &tables.forward);
            let rhs_image = apply_5bit(rhs, &tables.forward);
            let product_image =
                bits_ghash2(lhs_image).mul(bits_ghash2(rhs_image), mul_raw_portable);
            let expected = apply_5bit(mul_b256_portable(lhs, rhs), &tables.forward);
            assert_eq!(ghash2_bits(product_image), expected.limbs);
            assert_eq!(apply_5bit(lhs_image, &tables.inverse), lhs);
        }
    }

    fn apply_columns(input: BinaryElement, columns: &Columns) -> BinaryElement {
        let mut output = [0u64; 4];
        for (bit, column) in columns.iter().copied().enumerate() {
            if input.limbs[bit / 64] & (1u64 << (bit % 64)) != 0 {
                output = xor_bits(output, column);
            }
        }
        BinaryElement { limbs: output }
    }
}
