use super::*;

type Element = Ghash2Element;
type Polynomial = Vec<Element>;

#[test]
#[ignore = "offline isomorphism-constant generator"]
fn generate_b256_root_in_ghash2() {
    let mut modulus = vec![Element::ZERO; 257];
    for degree in [0, 2, 5, 10, 256] {
        modulus[degree] = Element::ONE;
    }

    let mut factor = modulus;
    let mut rng = StdRng::seed_from_u64(0x6973_6f6d_6f72_7068);
    while degree(&factor) > 1 {
        let old_degree = degree(&factor);
        loop {
            let candidate = random_polynomial(&mut rng, old_degree);
            let splitter = absolute_trace_mod(candidate, &factor);
            let divisor = gcd(factor.clone(), splitter);
            let divisor_degree = degree(&divisor);
            if divisor_degree != 0 && divisor_degree != old_degree {
                eprintln!("split degree {old_degree} -> {divisor_degree}");
                factor = divisor;
                break;
            }
        }
    }

    let mut root = factor[0];
    let mut conjugate = root;
    for _ in 1..256 {
        conjugate = field_mul(conjugate, conjugate);
        if (conjugate.c1.0, conjugate.c0.0) < (root.c1.0, root.c0.0) {
            root = conjugate;
        }
    }
    assert_eq!(evaluate_b256_modulus(root), Element::ZERO);
    println!(
        "root = Ghash2Element {{ c0: GhashElement(0x{:032x}), c1: GhashElement(0x{:032x}) }};",
        root.c0.0, root.c1.0
    );
}

fn random_polynomial(rng: &mut StdRng, max_degree: usize) -> Polynomial {
    let mut polynomial = Vec::with_capacity(max_degree);
    for _ in 0..max_degree {
        polynomial.push(Element {
            c0: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
            c1: GhashElement((rng.next_u64() as u128) | ((rng.next_u64() as u128) << 64)),
        });
    }
    trim(polynomial)
}

fn absolute_trace_mod(mut value: Polynomial, modulus: &[Element]) -> Polynomial {
    let mut trace = Vec::new();
    for _ in 0..256 {
        add_assign(&mut trace, &value);
        value = square_mod_monic(&value, modulus);
    }
    trim(trace)
}

fn square_mod_monic(value: &[Element], modulus: &[Element]) -> Polynomial {
    if value.is_empty() {
        return Vec::new();
    }

    let mut square = vec![Element::ZERO; value.len() * 2 - 1];
    for (index, coefficient) in value.iter().copied().enumerate() {
        square[index * 2] = field_mul(coefficient, coefficient);
    }
    remainder_monic(square, modulus)
}

fn gcd(mut lhs: Polynomial, mut rhs: Polynomial) -> Polynomial {
    while !rhs.is_empty() {
        let remainder = polynomial_remainder(lhs, &rhs);
        lhs = rhs;
        rhs = remainder;
    }
    make_monic(lhs)
}

fn polynomial_remainder(mut dividend: Polynomial, divisor: &[Element]) -> Polynomial {
    if dividend.len() < divisor.len() {
        return trim(dividend);
    }

    let inverse_lead = field_inverse(*divisor.last().expect("nonzero divisor"));
    while dividend.len() >= divisor.len() {
        let offset = dividend.len() - divisor.len();
        let factor = field_mul(*dividend.last().unwrap(), inverse_lead);
        for (target, coefficient) in dividend[offset..].iter_mut().zip(divisor.iter().copied()) {
            *target = target.add(field_mul(factor, coefficient));
        }
        trim_in_place(&mut dividend);
    }
    dividend
}

fn remainder_monic(mut dividend: Polynomial, modulus: &[Element]) -> Polynomial {
    while dividend.len() >= modulus.len() {
        let offset = dividend.len() - modulus.len();
        let factor = *dividend.last().unwrap();
        for (target, coefficient) in dividend[offset..modulus.len() + offset - 1]
            .iter_mut()
            .zip(modulus[..modulus.len() - 1].iter().copied())
        {
            *target = target.add(field_mul(factor, coefficient));
        }
        dividend.pop();
        trim_in_place(&mut dividend);
    }
    dividend
}

fn make_monic(mut polynomial: Polynomial) -> Polynomial {
    if let Some(lead) = polynomial.last().copied() {
        let inverse = field_inverse(lead);
        for coefficient in &mut polynomial {
            *coefficient = field_mul(*coefficient, inverse);
        }
    }
    trim(polynomial)
}

fn field_inverse(value: Element) -> Element {
    assert_ne!(value, Element::ZERO);
    let mut result = value;
    for _ in 1..255 {
        result = field_mul(field_mul(result, result), value);
    }
    field_mul(result, result)
}

fn field_mul(lhs: Element, rhs: Element) -> Element {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return unsafe { field_mul_pmull(lhs, rhs) };
    }
    lhs.mul(rhs, mul_raw_portable)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn field_mul_pmull(lhs: Element, rhs: Element) -> Element {
    unsafe { mul_ghash2_pmull(lhs, rhs) }
}

fn evaluate_b256_modulus(value: Element) -> Element {
    let mut powers = [Element::ONE; 257];
    for index in 1..powers.len() {
        powers[index] = field_mul(powers[index - 1], value);
    }
    [0, 2, 5, 10, 256]
        .into_iter()
        .fold(Element::ZERO, |sum, degree| sum.add(powers[degree]))
}

fn add_assign(lhs: &mut Polynomial, rhs: &[Element]) {
    lhs.resize(lhs.len().max(rhs.len()), Element::ZERO);
    for (target, value) in lhs.iter_mut().zip(rhs.iter().copied()) {
        *target = target.add(value);
    }
    trim_in_place(lhs);
}

fn degree(polynomial: &[Element]) -> usize {
    polynomial.len().saturating_sub(1)
}

fn trim(mut polynomial: Polynomial) -> Polynomial {
    trim_in_place(&mut polynomial);
    polynomial
}

fn trim_in_place(polynomial: &mut Polynomial) {
    while polynomial.last() == Some(&Element::ZERO) {
        polynomial.pop();
    }
}
