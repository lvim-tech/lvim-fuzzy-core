//! Scoring 16 candidates at once (AVX2): the scalar recurrence of the matcher
//! (src/matcher.rs) run in lockstep over 16 candidates of the same length,
//! one per 16-bit lane, with the same bonus rules. Only for the common hot
//! case — one plain fuzzy word over ASCII candidates that already passed the
//! prefilter — and only where the CPU has AVX2; the scalar path does the rest.
//! The scores are the scalar ones exactly (see the test).

use crate::matcher::{BONUS_CONSEC, BONUS_FIRST, GAP_INNER, GAP_TRAILING};

pub const LANES: usize = 16;
pub const MAX_NEEDLE: usize = 16;
/// Lines longer than this go to the scalar path (i16 lanes: the leading gap
/// alone could overflow far beyond).
pub const MAX_LEN: usize = 4000;
/// Candidates longer than this (bytes) are scored one by one: a tile of a few
/// long lines is mostly padding, and the scalar path skips the columns that
/// cannot match.
pub const TILE_MAX: usize = 256;

/// Whether the 16-lane path runs: the CPU has AVX2 and LVIM_FUZZY_NO_SIMD is
/// not set (for comparing, or to rule the path out when hunting a problem).
pub fn available() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("LVIM_FUZZY_NO_SIMD").is_some_and(|v| v != "0") {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Score 16 candidates of `len` characters, laid out column-major: for
/// character `j` of candidate `lane`, `keys[j * 16 + lane]` is the character
/// (as the needle compares: folded or not, normalised or not) and
/// `bonus[j * 16 + lane]` the bonus a match there earns. Shorter candidates
/// are padded with key 0 (never matched) and bonus 0; each padded character
/// costs the score GAP_TRAILING, which the caller adds back. A lane that
/// does not match gets a value below -16000.
///
/// # Safety
/// The CPU must support AVX2 ([`available`]); `keys` and `bonus` hold
/// `len * 16` entries; `needle` is 1..=16 characters, none 0; `len` <=
/// MAX_LEN.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn score_tile(
    keys: &[u16],
    bonus: &[i16],
    len: usize,
    needle: &[u16],
    out: &mut [i16; LANES],
) {
    use core::arch::x86_64::*;
    let m = needle.len();
    let last = m - 1;
    let neg = _mm256_set1_epi16(i16::MIN / 2);
    let mut d = [neg; MAX_NEEDLE];
    let mut mrow = [neg; MAX_NEEDLE];
    let s = |v: i32| _mm256_set1_epi16(v as i16);
    let vconsec = s(BONUS_CONSEC);
    let wants: [__m256i; MAX_NEEDLE] =
        std::array::from_fn(|i| s(*needle.get(i).unwrap_or(&0) as i32));
    let gaps: [__m256i; MAX_NEEDLE] =
        std::array::from_fn(|i| s(if i == last { GAP_TRAILING } else { GAP_INNER }));
    for j in 0..len {
        let c = _mm256_loadu_si256(keys.as_ptr().add(j * LANES) as *const __m256i);
        let b = _mm256_loadu_si256(bonus.as_ptr().add(j * LANES) as *const __m256i);
        let lead = s(-(j as i32)); // j * GAP_LEADING
        // rows from the last down: each cell reads the previous column of the row above
        for i in (0..m).rev() {
            let matched = _mm256_cmpeq_epi16(c, wants[i]);
            let (fresh, consec) = if i == 0 {
                (lead, neg)
            } else {
                (mrow[i - 1], d[i - 1])
            };
            let x = _mm256_adds_epi16(fresh, b);
            let y = _mm256_adds_epi16(_mm256_adds_epi16(consec, b), vconsec);
            let dval = _mm256_blendv_epi8(neg, _mm256_max_epi16(x, y), matched);
            mrow[i] = _mm256_max_epi16(dval, _mm256_adds_epi16(mrow[i], gaps[i]));
            d[i] = dval;
        }
    }
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, mrow[last]);
}

/// The rebased recursion over one window, 16 columns at a time (i16 lanes,
/// along the line). `fold` / `bonus`: the window's folded characters and
/// match bonuses, padded with zeros to a multiple of 16 (a 0 never matches);
/// `needle` 1..=16 characters, none 0; `rows` scratch of at least
/// 4 * (fold.len() + 16) values. Returns what `matcher::rebased_window`
/// returns (the best value of the last row; very low when nothing aligns).
///
/// Inner rows: M'[j] = max(D'[j], M'[j-1] - 1) is a prefix maximum of
/// D'[k] + k, minus j — four shift-and-max steps over the 16 lanes, the
/// previous chunk's last value carried in. The last row has no decay, so it
/// only needs the maximum of its D'.
///
/// # Safety
/// AVX2 ([`available`]); the slices as described.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn score_window(fold: &[i16], bonus: &[i16], needle: &[i16], rows: &mut [i16]) -> i32 {
    use core::arch::x86_64::*;
    let w = fold.len();
    let m = needle.len();
    let neg = _mm256_set1_epi16(i16::MIN / 2);
    let one = _mm256_set1_epi16(1);
    let c17 = _mm256_set1_epi16(17);
    let ramp = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    // shift the 16 lanes up by s, filling with `neg`
    let up = |x: __m256i, s: i32| -> __m256i {
        let z = _mm256_permute2x128_si256(x, neg, 0x02); // [neg.lo, x.lo]
        match s {
            1 => _mm256_alignr_epi8(x, z, 14),
            2 => _mm256_alignr_epi8(x, z, 12),
            4 => _mm256_alignr_epi8(x, z, 8),
            _ => z,
        }
    };
    // rows: previous D', previous M', current D', current M'; each with one
    // leading slot (column -1 = unreachable)
    let stride = w + 16;
    let (prev, cur) = rows.split_at_mut(2 * stride);
    let (pd, pm) = prev.split_at_mut(stride);
    let (cd, cm) = cur.split_at_mut(stride);
    let (mut pd, mut pm, mut cd, mut cm) = (pd, pm, cd, cm);
    let mut best = neg;
    for (i, &want) in needle.iter().enumerate() {
        let wantv = _mm256_set1_epi16(want);
        let last_row = i == m - 1;
        cd[0] = i16::MIN / 2;
        cm[0] = i16::MIN / 2;
        let mut carry = _mm256_set1_epi16(i16::MIN / 2);
        let mut c = 0;
        while c < w {
            let f = _mm256_loadu_si256(fold.as_ptr().add(c) as *const __m256i);
            let b = _mm256_loadu_si256(bonus.as_ptr().add(c) as *const __m256i);
            let eq = _mm256_cmpeq_epi16(f, wantv);
            let dval = if i == 0 {
                _mm256_blendv_epi8(neg, b, eq)
            } else {
                // column j reads row i-1 at column j-1: slot j of the shifted row
                let fp = _mm256_loadu_si256(pm.as_ptr().add(c) as *const __m256i);
                let cp = _mm256_loadu_si256(pd.as_ptr().add(c) as *const __m256i);
                let x = _mm256_adds_epi16(_mm256_adds_epi16(fp, b), one);
                let y = _mm256_adds_epi16(_mm256_adds_epi16(cp, b), c17);
                _mm256_blendv_epi8(neg, _mm256_max_epi16(x, y), eq)
            };
            if last_row {
                best = _mm256_max_epi16(best, dval);
            } else {
                _mm256_storeu_si256(cd.as_mut_ptr().add(c + 1) as *mut __m256i, dval);
                // M' = prefix max of (D' + lane) - lane, with the carry - 1 - lane
                let mut t = _mm256_adds_epi16(dval, ramp);
                t = _mm256_max_epi16(t, up(t, 1));
                t = _mm256_max_epi16(t, up(t, 2));
                t = _mm256_max_epi16(t, up(t, 4));
                t = _mm256_max_epi16(t, up(t, 8));
                t = _mm256_max_epi16(t, _mm256_subs_epi16(carry, one));
                let mv = _mm256_subs_epi16(t, ramp);
                _mm256_storeu_si256(cm.as_mut_ptr().add(c + 1) as *mut __m256i, mv);
                // the last lane, decayed over the 16 lanes, seeds the next chunk
                let last = _mm256_extract_epi16(mv, 15) as i16;
                carry = _mm256_set1_epi16(last);
            }
            c += 16;
        }
        std::mem::swap(&mut pd, &mut cd);
        std::mem::swap(&mut pm, &mut cm);
    }
    // the maximum over the 16 lanes
    let mut lanes = [0i16; 16];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, best);
    lanes.iter().map(|&v| v as i32).max().unwrap()
}

/// Lines per group of the inter-line kernel (one u8 lane per line).
pub const GROUP: usize = 32;
/// The widest line a group takes.
pub const GROUP_COLS: usize = 256;
/// What a reachable cell loses per needle row in the u8 kernel (the largest
/// gain of one step: bonus 18 + consecutive 16 + 1).
const GROUP_K: i32 = 35;
/// The bias of row 0.
const GROUP_B0: i32 = 202; // B0 + 18 + 35 <= 255: the sum before the row's drop
/// The same for the u16 lanes (m <= 16: B0 >= 128 + 15 * 35).
const GROUP16_B0: i32 = 1024;

/// The widest line the u8 group kernel scores exactly for a needle of `m`.
pub fn group_width(m: usize) -> usize {
    if m == 0 || m > 7 {
        return 0;
    }
    ((GROUP_B0 - GROUP_K * (m as i32 - 1)).max(0) as usize).min(GROUP_COLS)
}

/// What to subtract from a lane of `group_score` to get the rebased best.
pub fn group_bias(m: usize) -> i32 {
    GROUP_B0 - GROUP_K * (m as i32 - 1)
}

#[cfg(target_arch = "x86_64")]
pub type V = core::arch::x86_64::__m256i;

/// Whether ASCII byte `c` is a word boundary for the bonus rules (the one
/// definition: the byte tables of the search and the group kernel use it).
pub const fn is_boundary(c: u8) -> bool {
    matches!(
        c,
        b'/' | b'_'
            | b'-'
            | b'.'
            | b' '
            | b':'
            | b','
            | b';'
            | b'|'
            | b'\t'
            | b'\n'
            | 0x0b
            | 0x0c
            | b'\r'
    )
}

/// For the boundary test by nibbles: per low nibble, the set of high nibbles
/// (bits 0..7) that make a boundary with it.
static BOUND_LO: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut c = 0u8;
    while c < 128 {
        if is_boundary(c) {
            t[(c & 15) as usize] |= 1 << (c >> 4);
        }
        c += 1;
    }
    t
};

static GROUP_MASK: [u8; 64] = {
    let mut m = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        m[i] = 0xFF;
        i += 1;
    }
    m
};

/// Column-major copy of 32 lines: cols[j] holds byte j of every line (0 past
/// a line's end), for j < width rounded up to 32.
///
/// # Safety
/// AVX2; every pointer readable for `width` rounded up to 32 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_transpose(
    ptrs: &[*const u8; GROUP],
    lens: &[u16; GROUP],
    width: usize,
    cols: &mut [V; GROUP_COLS],
) {
    use core::arch::x86_64::*;
    let blocks = width.div_ceil(32);
    for b in 0..blocks {
        let mut half = [[_mm256_setzero_si256(); 16]; 2];
        for h in 0..2 {
            let mut cur = [_mm256_setzero_si256(); 16];
            for r in 0..16 {
                let line = h * 16 + r;
                let rem = (lens[line] as isize - (b * 32) as isize).clamp(0, 32) as usize;
                let v = _mm256_loadu_si256(ptrs[line].add(b * 32) as *const __m256i);
                let mask = _mm256_loadu_si256(GROUP_MASK.as_ptr().add(32 - rem) as *const __m256i);
                cur[r] = _mm256_and_si256(v, mask);
            }
            // four in-lane interleaves: bytes, words, doublewords, quadwords
            let mut nxt = [_mm256_setzero_si256(); 16];
            for g in 0..8 {
                nxt[2 * g] = _mm256_unpacklo_epi8(cur[2 * g], cur[2 * g + 1]);
                nxt[2 * g + 1] = _mm256_unpackhi_epi8(cur[2 * g], cur[2 * g + 1]);
            }
            // groups of 2 vectors (2 column blocks), 8 row groups
            for g in 0..4 {
                for q in 0..2 {
                    cur[4 * g + 2 * q] = _mm256_unpacklo_epi16(nxt[4 * g + q], nxt[4 * g + 2 + q]);
                    cur[4 * g + 2 * q + 1] =
                        _mm256_unpackhi_epi16(nxt[4 * g + q], nxt[4 * g + 2 + q]);
                }
            }
            for g in 0..2 {
                for q in 0..4 {
                    nxt[8 * g + 2 * q] = _mm256_unpacklo_epi32(cur[8 * g + q], cur[8 * g + 4 + q]);
                    nxt[8 * g + 2 * q + 1] =
                        _mm256_unpackhi_epi32(cur[8 * g + q], cur[8 * g + 4 + q]);
                }
            }
            for q in 0..8 {
                cur[2 * q] = _mm256_unpacklo_epi64(nxt[q], nxt[8 + q]);
                cur[2 * q + 1] = _mm256_unpackhi_epi64(nxt[q], nxt[8 + q]);
            }
            half[h] = cur;
        }
        for q in 0..16 {
            cols[b * 32 + q] = _mm256_permute2x128_si256(half[0][q], half[1][q], 0x20);
            cols[b * 32 + 16 + q] = _mm256_permute2x128_si256(half[0][q], half[1][q], 0x31);
        }
    }
}

/// Which of the 32 lines hold `needle` (1..=16 bytes, none 0) as a
/// subsequence: one counter per lane, three operations per column.
///
/// # Safety
/// AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_subseq(cols: &[V; GROUP_COLS], width: usize, needle: &[u8], fold: bool) -> u32 {
    use core::arch::x86_64::*;
    let mut t = [0u8; 16];
    t[..needle.len()].copy_from_slice(needle);
    let tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(t.as_ptr() as *const __m128i));
    let ka = _mm256_set1_epi8((0x80 - 0x41) as i8);
    let t26 = _mm256_set1_epi8(-128 + 26);
    let c20 = _mm256_set1_epi8(0x20);
    let mut p = _mm256_setzero_si256();
    for j in 0..width {
        let mut c = cols[j];
        if fold {
            let upper = _mm256_cmpgt_epi8(t26, _mm256_add_epi8(c, ka));
            c = _mm256_or_si256(c, _mm256_and_si256(upper, c20));
        }
        let eq = _mm256_cmpeq_epi8(c, _mm256_shuffle_epi8(tab, p));
        p = _mm256_subs_epi8(p, eq);
    }
    _mm256_movemask_epi8(_mm256_cmpgt_epi8(
        p,
        _mm256_set1_epi8(needle.len() as i8 - 1),
    )) as u32
}

/// Per column: the folded bytes (when `fold` is given) and the match bonus,
/// by the matcher's rule for ASCII.
///
/// # Safety
/// AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_classify(
    cols: &[V; GROUP_COLS],
    width: usize,
    path: bool,
    fold: Option<&mut [V; GROUP_COLS]>,
    bonus: &mut [V; GROUP_COLS],
) {
    use core::arch::x86_64::*;
    // bound(c) by two nibble lookups: the low-nibble table holds the set of
    // high nibbles for which (high, low) is a boundary; the high table one
    // bit per high nibble
    let lo_tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(BOUND_LO.as_ptr() as *const __m128i));
    let hi_tab = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0,
        0, 0, 0, 0, 0, 0,
    );
    let nib = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let s = |v: i32| _mm256_set1_epi8(v as i8);
    let (ka, kl, kd) = (s(0x80 - 0x41), s(0x80 - 0x61), s(0x80 - 0x30));
    let (t26, t10) = (s(-128 + 26), s(-128 + 10));
    let (c7, c14, c16, c2, c20) = (s(7), s(14), s(16), s(2), s(0x20));
    let slash = s(b'/' as i32);
    let mut p_lower = zero;
    let mut p_digit = zero;
    let mut p_bound = zero; // 16 / 18 where the previous byte is a boundary
    let mut fold = fold;
    for j in 0..width {
        let c = cols[j];
        let upper = _mm256_cmpgt_epi8(t26, _mm256_add_epi8(c, ka));
        let lower = _mm256_cmpgt_epi8(t26, _mm256_add_epi8(c, kl));
        let digit = _mm256_cmpgt_epi8(t10, _mm256_add_epi8(c, kd));
        let lo = _mm256_shuffle_epi8(lo_tab, _mm256_and_si256(c, nib));
        let hi = _mm256_shuffle_epi8(hi_tab, _mm256_and_si256(_mm256_srli_epi16(c, 4), nib));
        let nonbound = _mm256_cmpeq_epi8(_mm256_and_si256(lo, hi), zero);
        let b7 = _mm256_and_si256(_mm256_andnot_si256(p_digit, digit), c7);
        let b14 = _mm256_and_si256(_mm256_and_si256(p_lower, upper), c14);
        bonus[j] = _mm256_max_epu8(_mm256_max_epu8(b7, b14), p_bound);
        if let Some(f) = fold.as_deref_mut() {
            f[j] = _mm256_or_si256(c, _mm256_and_si256(upper, c20));
        }
        p_lower = lower;
        p_digit = digit;
        p_bound = _mm256_andnot_si256(nonbound, c16);
        if path {
            p_bound = _mm256_add_epi8(p_bound, _mm256_and_si256(_mm256_cmpeq_epi8(c, slash), c2));
        }
    }
    bonus[0] = s(BONUS_FIRST);
}

/// The rebased recursion for 32 lines at once, one u8 lane per line, column
/// by column. Returns the best value of the last row per lane, biased by
/// `group_bias(M)`; 0 = no alignment. Exact for width <= group_width(M).
///
/// # Safety
/// AVX2; `needle` holds M bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_score<const M: usize>(
    keys: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    width: usize,
    needle: &[u8],
) -> V {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let s = |v: i32| _mm256_set1_epi8(v as i8);
    let wants: [V; M] = std::array::from_fn(|i| s(needle[i] as i32));
    let (one, c17, k, b0) = (s(1), s(17), s(GROUP_K), s(GROUP_B0));
    let mut d = [zero; M];
    let mut mm = [zero; M];
    let mut best = zero;
    for j in 0..width {
        let f = keys[j];
        let b = bonus[j];
        let b1 = _mm256_adds_epu8(b, one);
        let b17 = _mm256_adds_epu8(b, c17);
        let mut i = M - 1;
        while i > 0 {
            let eq = _mm256_cmpeq_epi8(f, wants[i]);
            let x = _mm256_max_epu8(
                _mm256_adds_epu8(mm[i - 1], b1),
                _mm256_adds_epu8(d[i - 1], b17),
            );
            let dv = _mm256_and_si256(_mm256_subs_epu8(x, k), eq);
            d[i] = dv;
            if i == M - 1 {
                best = _mm256_max_epu8(best, dv);
            } else {
                mm[i] = _mm256_max_epu8(dv, _mm256_subs_epu8(mm[i], one));
            }
            i -= 1;
        }
        let eq = _mm256_cmpeq_epi8(f, wants[0]);
        let dv = _mm256_and_si256(_mm256_adds_epu8(b, b0), eq);
        d[0] = dv;
        if M == 1 {
            best = _mm256_max_epu8(best, dv);
        } else {
            mm[0] = _mm256_max_epu8(dv, _mm256_subs_epu8(mm[0], one));
        }
    }
    best
}

/// `group_score` with u16 lanes: the 16 lines of half `h` (0 / 1) of the
/// group. Any needle up to 16, any width up to GROUP_COLS. Biased by
/// `group16_bias(M)`.
///
/// # Safety
/// As `group_score`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_score16<const M: usize>(
    keys: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    width: usize,
    needle: &[u8],
    out: &mut [u16; GROUP],
) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let s = |v: i32| _mm256_set1_epi16(v as i16);
    let wants: [V; M] = std::array::from_fn(|i| s(needle[i] as i32));
    let (one, c17, k, b0) = (s(1), s(17), s(GROUP_K), s(GROUP16_B0));
    for h in 0..2 {
        let mut d = [zero; M];
        let mut mm = [zero; M];
        let mut best = zero;
        for j in 0..width {
            let (f, b) = if h == 0 {
                (
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(keys[j])),
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bonus[j])),
                )
            } else {
                (
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(keys[j], 1)),
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bonus[j], 1)),
                )
            };
            let b1 = _mm256_add_epi16(b, one);
            let b17 = _mm256_add_epi16(b, c17);
            let mut i = M - 1;
            while i > 0 {
                let eq = _mm256_cmpeq_epi16(f, wants[i]);
                let x = _mm256_max_epu16(
                    _mm256_add_epi16(mm[i - 1], b1),
                    _mm256_add_epi16(d[i - 1], b17),
                );
                let dv = _mm256_and_si256(_mm256_subs_epu16(x, k), eq);
                d[i] = dv;
                if i == M - 1 {
                    best = _mm256_max_epu16(best, dv);
                } else {
                    mm[i] = _mm256_max_epu16(dv, _mm256_subs_epu16(mm[i], one));
                }
                i -= 1;
            }
            let eq = _mm256_cmpeq_epi16(f, wants[0]);
            let dv = _mm256_and_si256(_mm256_add_epi16(b, b0), eq);
            d[0] = dv;
            if M == 1 {
                best = _mm256_max_epu16(best, dv);
            } else {
                mm[0] = _mm256_max_epu16(dv, _mm256_subs_epu16(mm[0], one));
            }
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(16 * h) as *mut __m256i, best);
    }
}

/// What to subtract from a lane of `group_score16`.
pub fn group16_bias(m: usize) -> i32 {
    GROUP16_B0 - GROUP_K * (m as i32 - 1)
}

/// Score a needle of 1..=16 bytes over the group: u8 lanes when the width
/// allows, u16 lanes otherwise. `best`: the rebased best per lane (the score
/// is that minus the line length - 1); `ok` is cleared for lanes without an
/// alignment.
///
/// # Safety
/// As `group_score`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_best(
    keys: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    width: usize,
    needle: &[u8],
    best: &mut [i32; GROUP],
    ok: &mut [bool; GROUP],
) {
    let m = needle.len();
    if width <= group_width(m) {
        let mut out = [0u8; GROUP];
        group_score_any(keys, bonus, width, needle, &mut out);
        let bias = group_bias(m);
        for k in 0..GROUP {
            ok[k] &= out[k] != 0;
            best[k] = out[k] as i32 - bias;
        }
        return;
    }
    let mut out = [0u16; GROUP];
    macro_rules! go { ($($n:literal)*) => { match m { $($n => group_score16::<$n>(keys, bonus, width, needle, &mut out),)* _ => unreachable!() } } }
    go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    let bias = group16_bias(m);
    for k in 0..GROUP {
        ok[k] &= out[k] != 0;
        best[k] = out[k] as i32 - bias;
    }
}

/// `group_score` for a needle of 1..=7 bytes, the lanes written to `out`.
///
/// # Safety
/// As `group_score`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_score_any(
    keys: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    width: usize,
    needle: &[u8],
    out: &mut [u8; GROUP],
) {
    use core::arch::x86_64::*;
    let v = match needle.len() {
        1 => group_score::<1>(keys, bonus, width, needle),
        2 => group_score::<2>(keys, bonus, width, needle),
        3 => group_score::<3>(keys, bonus, width, needle),
        4 => group_score::<4>(keys, bonus, width, needle),
        5 => group_score::<5>(keys, bonus, width, needle),
        6 => group_score::<6>(keys, bonus, width, needle),
        7 => group_score::<7>(keys, bonus, width, needle),
        _ => unreachable!(),
    };
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
}

/// Lanes where byte `x` is in [lo, hi]: lo shifted to -128, one signed compare.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn within8(x: V, lo: i32, hi: i32) -> V {
    use core::arch::x86_64::*;
    _mm256_cmpgt_epi8(
        _mm256_set1_epi8((-128 + hi - lo + 1) as i8),
        _mm256_add_epi8(x, _mm256_set1_epi8((0x80 - lo) as i8)),
    )
}

/// Half `h` of a byte vector widened to 16 bits: by value, or by sign (a
/// mask stays a mask).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn widen(v: V, h: usize, signed: bool) -> V {
    use core::arch::x86_64::*;
    let half = if h == 0 {
        _mm256_castsi256_si128(v)
    } else {
        _mm256_extracti128_si256(v, 1)
    };
    if signed {
        _mm256_cvtepi8_epi16(half)
    } else {
        _mm256_cvtepu8_epi16(half)
    }
}

/// `utf8_eq` on 16-bit lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn utf8_eq16(c: V, nx: V, n: &[V; 4], two: bool) -> V {
    use core::arch::x86_64::*;
    if two {
        _mm256_or_si256(
            _mm256_and_si256(_mm256_cmpeq_epi16(c, n[0]), _mm256_cmpeq_epi16(nx, n[1])),
            _mm256_and_si256(_mm256_cmpeq_epi16(c, n[2]), _mm256_cmpeq_epi16(nx, n[3])),
        )
    } else {
        _mm256_or_si256(_mm256_cmpeq_epi16(c, n[0]), _mm256_cmpeq_epi16(c, n[2]))
    }
}

/// A needle character for the UTF-8 group kernel: its bytes (lead, second;
/// second 0 for ASCII) and those of its other case (the same when the word
/// is case-sensitive or the character has no case).
pub type Utf8Char = [u8; 4];

/// The kernel's form of needle character `c` (as the pattern holds it:
/// folded unless the word is case-sensitive): ASCII or basic Cyrillic, else
/// None (the word goes to the per-line paths).
pub fn utf8_char(c: char, case_sensitive: bool) -> Option<Utf8Char> {
    if c == '\0' {
        return None;
    }
    if c.is_ascii() {
        let b = c as u8;
        let other = if case_sensitive {
            b
        } else {
            b.to_ascii_uppercase()
        };
        return Some([b, 0, other, 0]);
    }
    if !matches!(c, 'А'..='я' | 'Ё' | 'ё') {
        return None;
    }
    let other = if case_sensitive {
        c
    } else {
        match c {
            'а'..='я' => char::from_u32(c as u32 - 0x20).unwrap(),
            'ё' => 'Ё',
            _ => c, // an upper-case letter in a folded word cannot occur
        }
    };
    let mut a = [0u8; 4];
    let mut b = [0u8; 4];
    c.encode_utf8(&mut a);
    other.encode_utf8(&mut b);
    Some([a[0], a[1], b[0], b[1]])
}

/// Per column of a UTF-8 group: the match bonus (at the column where a
/// character starts, by the classes of that character and the previous one),
/// whether the column continues a character, and per lane the number of
/// characters. `cols[width]` must be zero (the kernel reads the next byte).
///
/// # Safety
/// AVX2; width < GROUP_COLS.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_classify_utf8(
    cols: &[V; GROUP_COLS],
    width: usize,
    path: bool,
    bonus: &mut [V; GROUP_COLS],
    cont: &mut [V; GROUP_COLS],
    chars: &mut [u8; GROUP],
) {
    use core::arch::x86_64::*;
    let lo_tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(BOUND_LO.as_ptr() as *const __m128i));
    let hi_tab = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0,
        0, 0, 0, 0, 0, 0,
    );
    let nib = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let s = |v: i32| _mm256_set1_epi8(v as i8);
    let (c7, c14, c16, c2) = (s(7), s(14), s(16), s(2));
    let (d0, d1, c81, c91, top2, c80) = (s(0xD0), s(0xD1), s(0x81), s(0x91), s(0xC0), s(0x80));
    let slash = s(b'/' as i32);
    let mut p_lower = zero;
    let mut p_digit = zero;
    let mut p_bound = zero;
    let mut count = zero;
    for j in 0..width {
        let c = cols[j];
        let nx = cols[j + 1];
        let is_cont = _mm256_cmpeq_epi8(_mm256_and_si256(c, top2), c80);
        let upper_a = within8(c, 0x41, 0x5A);
        let lower_a = within8(c, 0x61, 0x7A);
        let digit = within8(c, 0x30, 0x39);
        let is_d0 = _mm256_cmpeq_epi8(c, d0);
        let is_d1 = _mm256_cmpeq_epi8(c, d1);
        // А-Я: D0 90-AF, Ё: D0 81; а-п: D0 B0-BF, р-я: D1 80-8F, ё: D1 91
        let upper_c = _mm256_and_si256(
            is_d0,
            _mm256_or_si256(within8(nx, 0x90, 0xAF), _mm256_cmpeq_epi8(nx, c81)),
        );
        let lower_c = _mm256_or_si256(
            _mm256_and_si256(is_d0, within8(nx, 0xB0, 0xBF)),
            _mm256_and_si256(
                is_d1,
                _mm256_or_si256(within8(nx, 0x80, 0x8F), _mm256_cmpeq_epi8(nx, c91)),
            ),
        );
        let upper = _mm256_or_si256(upper_a, upper_c);
        let lower = _mm256_or_si256(lower_a, lower_c);
        let lo = _mm256_shuffle_epi8(lo_tab, _mm256_and_si256(c, nib));
        let hi = _mm256_shuffle_epi8(hi_tab, _mm256_and_si256(_mm256_srli_epi16(c, 4), nib));
        let bound = _mm256_xor_si256(
            _mm256_cmpeq_epi8(_mm256_and_si256(lo, hi), zero),
            _mm256_set1_epi8(-1),
        );
        let b7 = _mm256_and_si256(_mm256_andnot_si256(p_digit, digit), c7);
        let b14 = _mm256_and_si256(_mm256_and_si256(p_lower, upper), c14);
        bonus[j] = _mm256_max_epu8(_mm256_max_epu8(b7, b14), p_bound);
        cont[j] = is_cont;
        // the classes of the previous character move only where one starts
        let mut nb = _mm256_and_si256(bound, c16);
        if path {
            nb = _mm256_add_epi8(nb, _mm256_and_si256(_mm256_cmpeq_epi8(c, slash), c2));
        }
        p_lower = _mm256_blendv_epi8(lower, p_lower, is_cont);
        p_digit = _mm256_blendv_epi8(digit, p_digit, is_cont);
        p_bound = _mm256_blendv_epi8(nb, p_bound, is_cont);
        // a character starts at a non-continuation byte that is not padding
        let starts = _mm256_andnot_si256(
            is_cont,
            _mm256_xor_si256(_mm256_cmpeq_epi8(c, zero), _mm256_set1_epi8(-1)),
        );
        count = _mm256_sub_epi8(count, starts);
    }
    bonus[0] = s(BONUS_FIRST);
    _mm256_storeu_si256(chars.as_mut_ptr() as *mut __m256i, count);
}

/// Where needle character `n` matches at a column (its byte `c`, the next
/// byte `nx`): ASCII by one byte, two-byte characters by both, either case.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn utf8_eq(c: V, nx: V, n: &[V; 4], two: bool) -> V {
    use core::arch::x86_64::*;
    if two {
        _mm256_or_si256(
            _mm256_and_si256(_mm256_cmpeq_epi8(c, n[0]), _mm256_cmpeq_epi8(nx, n[1])),
            _mm256_and_si256(_mm256_cmpeq_epi8(c, n[2]), _mm256_cmpeq_epi8(nx, n[3])),
        )
    } else {
        _mm256_or_si256(_mm256_cmpeq_epi8(c, n[0]), _mm256_cmpeq_epi8(c, n[2]))
    }
}

/// `group_score` over UTF-8 lines (u8 lanes): a continuation column leaves
/// every row as it was (no match, no gap), so the columns that count are the
/// characters. Biased by `group_bias(M)`; 0 = no alignment.
///
/// # Safety
/// AVX2; `cols[width]` zero; `needle` holds M characters.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_score_utf8<const M: usize>(
    cols: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    cont: &[V; GROUP_COLS],
    width: usize,
    needle: &[Utf8Char],
) -> V {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let s = |v: u8| _mm256_set1_epi8(v as i8);
    let wants: [[V; 4]; M] = std::array::from_fn(|i| needle[i].map(s));
    let two: [bool; M] = std::array::from_fn(|i| needle[i][1] != 0);
    let (one, c17, k, b0) = (s(1), s(17), s(GROUP_K as u8), s(GROUP_B0 as u8));
    let mut d = [zero; M];
    let mut mm = [zero; M];
    let mut best = zero;
    for j in 0..width {
        let (c, nx, b, ct) = (cols[j], cols[j + 1], bonus[j], cont[j]);
        let b1 = _mm256_adds_epu8(b, one);
        let b17 = _mm256_adds_epu8(b, c17);
        let mut i = M - 1;
        while i > 0 {
            let eq = utf8_eq(c, nx, &wants[i], two[i]);
            let x = _mm256_max_epu8(
                _mm256_adds_epu8(mm[i - 1], b1),
                _mm256_adds_epu8(d[i - 1], b17),
            );
            let dv = _mm256_and_si256(_mm256_subs_epu8(x, k), eq);
            if i == M - 1 {
                best = _mm256_max_epu8(best, dv);
            } else {
                mm[i] = _mm256_blendv_epi8(
                    _mm256_max_epu8(dv, _mm256_subs_epu8(mm[i], one)),
                    mm[i],
                    ct,
                );
            }
            d[i] = _mm256_blendv_epi8(dv, d[i], ct);
            i -= 1;
        }
        let eq = utf8_eq(c, nx, &wants[0], two[0]);
        let dv = _mm256_and_si256(_mm256_adds_epu8(b, b0), eq);
        if M == 1 {
            best = _mm256_max_epu8(best, dv);
        } else {
            mm[0] =
                _mm256_blendv_epi8(_mm256_max_epu8(dv, _mm256_subs_epu8(mm[0], one)), mm[0], ct);
        }
        d[0] = _mm256_blendv_epi8(dv, d[0], ct);
    }
    best
}

/// `group_score_utf8` with u16 lanes (any width < GROUP_COLS, M <= 16); the
/// two halves of the group one after the other. Biased by `group16_bias(M)`.
///
/// # Safety
/// As `group_score_utf8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_score16_utf8<const M: usize>(
    cols: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    cont: &[V; GROUP_COLS],
    width: usize,
    needle: &[Utf8Char],
    out: &mut [u16; GROUP],
) {
    use core::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let s = |v: i32| _mm256_set1_epi16(v as i16);
    let wants: [[V; 4]; M] = std::array::from_fn(|i| needle[i].map(|b| s(b as i32)));
    let two: [bool; M] = std::array::from_fn(|i| needle[i][1] != 0);
    let (one, c17, k, b0) = (s(1), s(17), s(GROUP_K), s(GROUP16_B0));
    for h in 0..2 {
        let mut d = [zero; M];
        let mut mm = [zero; M];
        let mut best = zero;
        for j in 0..width {
            let (c, nx, b) = (
                widen(cols[j], h, false),
                widen(cols[j + 1], h, false),
                widen(bonus[j], h, false),
            );
            // the mask widened by sign: 0xFF -> 0xFFFF
            let ct = widen(cont[j], h, true);
            let b1 = _mm256_add_epi16(b, one);
            let b17 = _mm256_add_epi16(b, c17);
            let mut i = M - 1;
            while i > 0 {
                let eq = utf8_eq16(c, nx, &wants[i], two[i]);
                let x = _mm256_max_epu16(
                    _mm256_add_epi16(mm[i - 1], b1),
                    _mm256_add_epi16(d[i - 1], b17),
                );
                let dv = _mm256_and_si256(_mm256_subs_epu16(x, k), eq);
                if i == M - 1 {
                    best = _mm256_max_epu16(best, dv);
                } else {
                    mm[i] = _mm256_blendv_epi8(
                        _mm256_max_epu16(dv, _mm256_subs_epu16(mm[i], one)),
                        mm[i],
                        ct,
                    );
                }
                d[i] = _mm256_blendv_epi8(dv, d[i], ct);
                i -= 1;
            }
            let eq = utf8_eq16(c, nx, &wants[0], two[0]);
            let dv = _mm256_and_si256(_mm256_add_epi16(b, b0), eq);
            if M == 1 {
                best = _mm256_max_epu16(best, dv);
            } else {
                mm[0] = _mm256_blendv_epi8(
                    _mm256_max_epu16(dv, _mm256_subs_epu16(mm[0], one)),
                    mm[0],
                    ct,
                );
            }
            d[0] = _mm256_blendv_epi8(dv, d[0], ct);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(16 * h) as *mut __m256i, best);
    }
}

/// `group_best` for UTF-8 lines: the rebased best per lane, `ok` cleared
/// where the needle does not align.
///
/// # Safety
/// As `group_score_utf8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_best_utf8(
    cols: &[V; GROUP_COLS],
    bonus: &[V; GROUP_COLS],
    cont: &[V; GROUP_COLS],
    width: usize,
    needle: &[Utf8Char],
    best: &mut [i32; GROUP],
    ok: &mut [bool; GROUP],
) {
    use core::arch::x86_64::*;
    let m = needle.len();
    if width <= group_width(m) {
        let v = match m {
            1 => group_score_utf8::<1>(cols, bonus, cont, width, needle),
            2 => group_score_utf8::<2>(cols, bonus, cont, width, needle),
            3 => group_score_utf8::<3>(cols, bonus, cont, width, needle),
            4 => group_score_utf8::<4>(cols, bonus, cont, width, needle),
            5 => group_score_utf8::<5>(cols, bonus, cont, width, needle),
            6 => group_score_utf8::<6>(cols, bonus, cont, width, needle),
            7 => group_score_utf8::<7>(cols, bonus, cont, width, needle),
            _ => unreachable!(),
        };
        let mut out = [0u8; GROUP];
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
        let bias = group_bias(m);
        for k in 0..GROUP {
            ok[k] &= out[k] != 0;
            best[k] = out[k] as i32 - bias;
        }
        return;
    }
    let mut out = [0u16; GROUP];
    macro_rules! go { ($($n:literal)*) => { match m { $($n => group_score16_utf8::<$n>(cols, bonus, cont, width, needle, &mut out),)* _ => unreachable!() } } }
    go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    let bias = group16_bias(m);
    for k in 0..GROUP {
        ok[k] &= out[k] != 0;
        best[k] = out[k] as i32 - bias;
    }
}

/// Which of the 32 UTF-8 lines hold `needle` as a subsequence (one counter
/// per lane; the needle characters looked up by the counter).
///
/// # Safety
/// AVX2; `cols[width]` zero; needle 1..=16 characters.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn group_subseq_utf8(cols: &[V; GROUP_COLS], width: usize, needle: &[Utf8Char]) -> u32 {
    use core::arch::x86_64::*;
    let mut t = [[0u8; 16]; 4];
    for (i, ch) in needle.iter().enumerate() {
        for b in 0..4 {
            t[b][i] = ch[b];
        }
    }
    let tab =
        |b: usize| _mm256_broadcastsi128_si256(_mm_loadu_si128(t[b].as_ptr() as *const __m128i));
    let (l1, s1, l2, s2) = (tab(0), tab(1), tab(2), tab(3));
    let zero = _mm256_setzero_si256();
    let mut p = zero;
    for j in 0..width {
        let (c, nx) = (cols[j], cols[j + 1]);
        let (a, b, x, y) = (
            _mm256_shuffle_epi8(l1, p),
            _mm256_shuffle_epi8(s1, p),
            _mm256_shuffle_epi8(l2, p),
            _mm256_shuffle_epi8(s2, p),
        );
        // the second byte matters only for a two-byte character (second != 0)
        let second1 = _mm256_or_si256(_mm256_cmpeq_epi8(b, zero), _mm256_cmpeq_epi8(nx, b));
        let second2 = _mm256_or_si256(_mm256_cmpeq_epi8(y, zero), _mm256_cmpeq_epi8(nx, y));
        let eq = _mm256_or_si256(
            _mm256_and_si256(_mm256_cmpeq_epi8(c, a), second1),
            _mm256_and_si256(_mm256_cmpeq_epi8(c, x), second2),
        );
        p = _mm256_subs_epi8(p, eq);
    }
    _mm256_movemask_epi8(_mm256_cmpgt_epi8(
        p,
        _mm256_set1_epi8(needle.len() as i8 - 1),
    )) as u32
}

/// For each 64 bytes of `hay` (out.len() == hay.len().div_ceil(64)), the
/// mask of the bytes equal to `a` or `b`; bits past the end are 0. `tail`:
/// the last `hay.len() % 32` bytes, then zeros.
///
/// # Safety
/// AVX2 ([`available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn eq2_masks(hay: &[u8], tail: &[u8; 32], a: u8, b: u8, out: &mut [u64]) {
    use core::arch::x86_64::*;
    let va = _mm256_set1_epi8(a as i8);
    let vb = _mm256_set1_epi8(b as i8);
    let n = hay.len();
    let full = n / 32; // whole 32-byte chunks; `tail` holds the rest, then
    // zeros (a, b are never 0: needles hold no NUL)
    for (k, o) in out.iter_mut().enumerate() {
        let mut word = 0u64;
        for h in 0..2 {
            let chunk = 2 * k + h;
            let p = if chunk < full {
                hay.as_ptr().add(chunk * 32)
            } else if chunk == full && n % 32 != 0 {
                tail.as_ptr()
            } else {
                break;
            };
            let v = _mm256_loadu_si256(p as *const __m256i);
            let eq = _mm256_or_si256(_mm256_cmpeq_epi8(v, va), _mm256_cmpeq_epi8(v, vb));
            word |= (_mm256_movemask_epi8(eq) as u32 as u64) << (32 * h);
        }
        *o = word;
    }
}

/// Fill lane `lane` of a tile with `text`: keys (folded and as-is, each only
/// when `fold` / `raw` is given) and bonuses, by the matcher's rules
/// (normalised first when `normalize`). Returns the number of characters.
#[allow(clippy::too_many_arguments)]
pub fn fill_lane(
    text: &str,
    lane: usize,
    normalize: bool,
    path: bool,
    mut raw: Option<&mut [u16]>,
    mut fold: Option<&mut [u16]>,
    bonus: &mut [i16],
) -> usize {
    use crate::matcher::{char_bonus, fold_char, normalize_char};
    let mut prev = '\0';
    let mut j = 0;
    let key = |c: char| {
        if (c as u32) < 0xFFFF {
            c as u16
        } else {
            0xFFFF
        }
    };
    for c in text.chars() {
        let c = if normalize { normalize_char(c) } else { c };
        let at = j * LANES + lane;
        if let Some(r) = raw.as_deref_mut() {
            r[at] = key(c);
        }
        if let Some(f) = fold.as_deref_mut() {
            f[at] = key(fold_char(c));
        }
        bonus[at] = if j == 0 {
            BONUS_FIRST as i16
        } else {
            char_bonus(prev, c, path) as i16
        };
        prev = c;
        j += 1;
    }
    j
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::matcher::{CaseMode, MatchConfig, Pattern, Scheme};

    #[test]
    fn simd_scores_like_the_scalar_path() {
        if !available() {
            eprintln!("no AVX2: skipped");
            return;
        }
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha: Vec<char> = "abcdeABCDE/_-. :,;|\t0129xyzабвАБВéÉ".chars().collect();
        let mut checked = 0;
        for &(q, path) in &[
            ("ab", false),
            ("abc", true),
            ("Ab", false),
            ("a0", false),
            ("x/y", true),
            ("eDc", false),
            ("абв", false),
            ("Бв", false),
            ("e", false),
            ("abcdeabcdeabcdea", false),
        ] {
            let cfg = MatchConfig {
                case: CaseMode::Smart,
                extended: true,
                exact: false,
                normalize: true,
                scheme: if path { Scheme::Path } else { Scheme::Default },
                typos: 0,
                exact_window: crate::matcher::EXACT_WINDOW,
            };
            let pat = Pattern::parse(q, &cfg);
            let (words, p, _) = pat.plain_words().expect("plain words");
            let [(w, cs)] = words.as_slice() else {
                panic!()
            };
            let needle: Vec<u16> = w.iter().map(|&c| c as u16).collect();
            for maxlen in [1usize, 2, 5, 17, 40, 130] {
                for _ in 0..40 {
                    let texts: Vec<String> = (0..LANES)
                        .map(|_| {
                            (0..1 + rnd(maxlen))
                                .map(|_| alpha[rnd(alpha.len())])
                                .collect()
                        })
                        .collect();
                    let lens: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();
                    let width = (*lens.iter().max().unwrap()).div_ceil(8) * 8;
                    let mut keys = vec![0u16; width * LANES];
                    let mut bonus = vec![0i16; width * LANES];
                    for (lane, t) in texts.iter().enumerate() {
                        if *cs {
                            fill_lane(t, lane, true, p, Some(&mut keys), None, &mut bonus);
                        } else {
                            fill_lane(t, lane, true, p, None, Some(&mut keys), &mut bonus);
                        }
                    }
                    let mut out = [0i16; LANES];
                    unsafe { score_tile(&keys, &bonus, width, &needle, &mut out) };
                    for (lane, t) in texts.iter().enumerate() {
                        let scalar = pat.score(t).map(|s| s.score);
                        let pad = (width - lens[lane]) as i32;
                        let simd = (out[lane] as i32 > -16000).then_some(out[lane] as i32 + pad);
                        assert_eq!(simd, scalar, "query {q:?} on {t:?}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 30_000);
    }

    #[test]
    fn group_kernel_like_the_scalar_path() {
        if !available() {
            return;
        }
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha: Vec<u8> = b"abcdeABCDE/_-. :,;|\t\x0b\x0c0129xyz".to_vec();
        let mut checked = 0;
        let mut matched = 0;
        for &(q, path) in &[
            ("ab", false),
            ("abc", true),
            ("Ab", false),
            ("a0", false),
            ("x/y", true),
            ("eDc", false),
            ("e", false),
            ("abcdea", false),
            ("a.b_c9d", true),
            ("abcdeabcdeabcdea", false),
            ("ab/cd_e.a", true),
            ("aaaa", false),
            ("/", true),
            ("ABC", false),
            ("a/b", true),
            ("1", false),
        ] {
            let cfg = MatchConfig {
                case: CaseMode::Smart,
                extended: true,
                exact: false,
                normalize: true,
                scheme: if path { Scheme::Path } else { Scheme::Default },
                typos: 0,
                exact_window: crate::matcher::EXACT_WINDOW,
            };
            let pat = Pattern::parse(q, &cfg);
            let (words, p, _) = pat.plain_words().expect("plain words");
            let [(w, cs)] = words.as_slice() else {
                panic!()
            };
            let needle: Vec<u8> = w.iter().map(|&c| c as u8).collect();
            for maxlen in [
                1usize, 2, 5, 17, 27, 28, 33, 62, 63, 64, 97, 98, 128, 132, 133, 167, 168, 200,
                202, 203, 255, 256,
            ] {
                for _ in 0..60 {
                    let texts: Vec<Vec<u8>> = (0..GROUP)
                        .map(|_| {
                            let mut t: Vec<u8> = (0..rnd(maxlen + 1))
                                .map(|_| alpha[rnd(alpha.len())])
                                .collect();
                            let n = t.len();
                            t.resize(GROUP_COLS + 2, b'a'); // what follows a line in the store: other lines
                            t[GROUP_COLS] = (n >> 8) as u8;
                            t[GROUP_COLS + 1] = n as u8;
                            t
                        })
                        .collect();
                    let lens: [u16; GROUP] = std::array::from_fn(|k| {
                        (texts[k][GROUP_COLS] as u16) << 8 | texts[k][GROUP_COLS + 1] as u16
                    });
                    let ptrs: [*const u8; GROUP] = std::array::from_fn(|k| texts[k].as_ptr());
                    let width = *lens.iter().max().unwrap() as usize;
                    let mut best = [0i32; GROUP];
                    let mut ok = [true; GROUP];
                    unsafe {
                        let z = core::arch::x86_64::_mm256_setzero_si256();
                        let mut cols = [z; GROUP_COLS];
                        let mut fold = [z; GROUP_COLS];
                        let mut bonus = [z; GROUP_COLS];
                        group_transpose(&ptrs, &lens, width, &mut cols);
                        group_classify(&cols, width, p, Some(&mut fold), &mut bonus);
                        group_best(
                            if *cs { &cols } else { &fold },
                            &bonus,
                            width,
                            &needle,
                            &mut best,
                            &mut ok,
                        );
                        let sub = group_subseq(&cols, width, &needle, !*cs);
                        for k in 0..GROUP {
                            assert_eq!(sub >> k & 1 != 0, ok[k], "subseq {q:?}");
                        }
                    }
                    for k in 0..GROUP {
                        let n = lens[k] as usize;
                        let t = std::str::from_utf8(&texts[k][..n]).unwrap();
                        let scalar = pat.score(t).map(|s| s.score);
                        let got = ok[k].then(|| best[k] - (n as i32 - 1));
                        assert_eq!(got, scalar, "query {q:?} on {t:?}");
                        checked += 1;
                        matched += scalar.is_some() as usize;
                    }
                }
            }
        }
        assert!(checked > 100_000 && matched > 20_000, "{checked} {matched}");
    }

    #[test]
    fn group_width_is_safe() {
        // at the widest width the u8 kernel takes for each needle length, it
        // agrees with the u16 kernel (which has room to spare) on every lane
        if !available() {
            return;
        }
        let mut seed = 0x0DDB_1A5E_5BAD_5EEDu64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha = b"aaab/_A1 ";
        for m in 1..=7usize {
            let width = group_width(m);
            if width == 0 {
                continue;
            }
            for _ in 0..400 {
                let needle: Vec<u8> = (0..m).map(|_| b"ab/1"[rnd(4)]).collect();
                let texts: Vec<Vec<u8>> = (0..GROUP)
                    .map(|_| {
                        let mut t: Vec<u8> = (0..width).map(|_| alpha[rnd(alpha.len())]).collect();
                        t.resize(GROUP_COLS, 0);
                        t
                    })
                    .collect();
                let lens = [width as u16; GROUP];
                let ptrs: [*const u8; GROUP] = std::array::from_fn(|k| texts[k].as_ptr());
                unsafe {
                    let z = core::arch::x86_64::_mm256_setzero_si256();
                    let mut cols = [z; GROUP_COLS];
                    let mut bonus = [z; GROUP_COLS];
                    group_transpose(&ptrs, &lens, width, &mut cols);
                    group_classify(&cols, width, true, None, &mut bonus);
                    let mut narrow = [0u8; GROUP];
                    group_score_any(&cols, &bonus, width, &needle, &mut narrow);
                    let mut wide = [0u16; GROUP];
                    macro_rules! go { ($($n:literal)*) => { match m { $($n => group_score16::<$n>(&cols, &bonus, width, &needle, &mut wide),)* _ => unreachable!() } } }
                    go!(1 2 3 4 5 6 7);
                    for k in 0..GROUP {
                        let a = (narrow[k] != 0).then(|| narrow[k] as i32 - group_bias(m));
                        let b = (wide[k] != 0).then(|| wide[k] as i32 - group16_bias(m));
                        assert_eq!(a, b, "m {m} width {width}");
                    }
                }
            }
        }
    }

    #[test]
    fn utf8_group_kernel_like_the_scalar_path() {
        if !available() {
            return;
        }
        let mut seed = 0xC0FF_EE12_3456_789Bu64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha: Vec<char> = "abAB/_- .1тТрРсСёЁяЯаА→─—中😀".chars().collect();
        let mut checked = 0;
        let mut matched = 0;
        for &(q, path) in &[
            ("тр", false),
            ("Тр", false),
            ("ё", false),
            ("Ё", false),
            ("а", false),
            ("ab", false),
            ("тa", true),
            ("т/р", true),
            ("яяя", false),
            ("тртртртртртртртр", false),
            ("тсрёяа", false),
            ("a1", false),
        ] {
            let cfg = MatchConfig {
                case: CaseMode::Smart,
                extended: true,
                exact: false,
                normalize: true,
                scheme: if path { Scheme::Path } else { Scheme::Default },
                typos: 0,
                exact_window: crate::matcher::EXACT_WINDOW,
            };
            let pat = Pattern::parse(q, &cfg);
            let (words, p, _) = pat.plain_words().expect("plain words");
            let [(w, cs)] = words.as_slice() else {
                panic!()
            };
            let needle: Vec<Utf8Char> = w.iter().map(|&c| utf8_char(c, *cs).unwrap()).collect();
            for maxchars in [1usize, 3, 16, 40, 60, 100] {
                for _ in 0..40 {
                    let texts: Vec<String> = (0..GROUP)
                        .map(|_| {
                            loop {
                                let t: String = (0..rnd(maxchars + 1))
                                    .map(|_| alpha[rnd(alpha.len())])
                                    .collect();
                                if t.len() < GROUP_COLS {
                                    break t;
                                }
                            }
                        })
                        .collect();
                    let bufs: Vec<Vec<u8>> = texts
                        .iter()
                        .map(|t| {
                            let mut b = t.as_bytes().to_vec();
                            b.resize(GROUP_COLS + 32, 0xD0); // what follows: other lines
                            b
                        })
                        .collect();
                    let lens: [u16; GROUP] = std::array::from_fn(|k| texts[k].len() as u16);
                    let ptrs: [*const u8; GROUP] = std::array::from_fn(|k| bufs[k].as_ptr());
                    let width = *lens.iter().max().unwrap() as usize;
                    let mut best = [0i32; GROUP];
                    let mut ok = [true; GROUP];
                    let mut chars = [0u8; GROUP];
                    unsafe {
                        let z = core::arch::x86_64::_mm256_setzero_si256();
                        let mut cols = [z; GROUP_COLS];
                        let mut bonus = [z; GROUP_COLS];
                        let mut cont = [z; GROUP_COLS];
                        group_transpose(&ptrs, &lens, width, &mut cols);
                        cols[width] = z;
                        group_classify_utf8(&cols, width, p, &mut bonus, &mut cont, &mut chars);
                        group_best_utf8(&cols, &bonus, &cont, width, &needle, &mut best, &mut ok);
                        let sub = group_subseq_utf8(&cols, width, &needle);
                        for k in 0..GROUP {
                            assert_eq!(sub >> k & 1 != 0, ok[k], "subseq {q:?} on {:?}", texts[k]);
                        }
                    }
                    for k in 0..GROUP {
                        let t = &texts[k];
                        assert_eq!(chars[k] as usize, t.chars().count(), "{t:?}");
                        let scalar = pat.score(t).map(|s| s.score);
                        let got = ok[k].then(|| best[k] - (chars[k] as i32 - 1));
                        assert_eq!(got, scalar, "query {q:?} on {t:?}");
                        checked += 1;
                        matched += scalar.is_some() as usize;
                    }
                }
            }
        }
        assert!(checked > 50_000 && matched > 5_000, "{checked} {matched}");
    }

    #[test]
    fn window_kernel_like_the_reference() {
        if !available() {
            return;
        }
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut checked = 0;
        for &w in &[
            1usize, 2, 15, 16, 17, 31, 32, 33, 64, 65, 100, 255, 256, 600,
        ] {
            for m in [1usize, 2, 3, 4, 7, 16] {
                for _ in 0..60 {
                    let alpha = 1 + rnd(4);
                    let fold: Vec<i16> = (0..w).map(|_| 1 + rnd(alpha) as i16).collect();
                    let bonus: Vec<i16> = (0..w).map(|_| [0, 0, 7, 14, 16, 18][rnd(6)]).collect();
                    let needle: Vec<i16> = (0..m).map(|_| 1 + rnd(alpha) as i16).collect();
                    let want = crate::matcher::rebased_window(&fold, &bonus, &needle);
                    let padded = w.div_ceil(16) * 16;
                    let mut f = fold.clone();
                    f.resize(padded, 0);
                    let mut b = bonus.clone();
                    b.resize(padded, 0);
                    let mut rows = vec![0i16; 4 * (padded + 16)];
                    let got = unsafe { score_window(&f, &b, &needle, &mut rows) };
                    if want > -10000 {
                        assert_eq!(got, want, "w {w} m {m}");
                    } else {
                        assert!(got < -10000, "w {w} m {m}: {got}");
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 4000);
    }
}
