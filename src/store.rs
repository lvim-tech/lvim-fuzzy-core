//! The candidates: one byte buffer holding every text, with an offset table
//! and, per candidate, what the search reads before it scores anything — the
//! presence signature (an O(1) prefilter), whether it is ASCII, and the shape
//! of a grep result (`path:line[:col]:text`). No string per candidate.

/// 32-bit presence signature of one byte: a–z (case-folded) one bit each,
/// digits one bit, the remaining bytes spread over four bits.
#[inline(always)]
pub fn sig_byte(b: u8) -> u32 {
    let l = b.to_ascii_lowercase();
    if l.is_ascii_lowercase() {
        1u32 << (l - b'a')
    } else if l.is_ascii_digit() {
        1u32 << 26
    } else {
        1u32 << (27 + (l as u32 & 3))
    }
}

/// The signature of a whole text. Non-ASCII text gets every bit: a letter
/// with a diacritic may match its base letter (normalisation), so nothing can
/// be ruled out from its bytes.
pub fn signature(text: &[u8]) -> u32 {
    let (sig, high, _) = scan(text);
    if high { u32::MAX } else { sig }
}

/// Per byte: its signature bit (bits 0..31) and whether it is an upper-case
/// ASCII letter (bit 32), for one pass without branches.
static BYTE_INFO: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut b = 0usize;
    while b < 256 {
        let c = b as u8;
        let l = c.to_ascii_lowercase();
        let sig: u32 = if l.is_ascii_lowercase() {
            1u32 << (l - b'a')
        } else if l.is_ascii_digit() {
            1u32 << 26
        } else {
            1u32 << (27 + (l as u32 & 3))
        };
        t[b] = sig as u64 | (c.is_ascii_uppercase() as u64) << 32;
        b += 1;
    }
    t
};

/// One pass over `text`: the ASCII signature, whether any byte is not
/// ASCII, whether an upper-case ASCII letter occurs.
#[inline]
fn scan(text: &[u8]) -> (u32, bool, bool) {
    let mut acc = 0u64;
    let mut or = 0u8;
    for &b in text {
        acc |= BYTE_INFO[b as usize];
        or |= b;
    }
    (acc as u32, or & 0x80 != 0, acc >> 32 != 0)
}

const NO_GREP: u32 = u32::MAX;

/// Above this many bytes a candidate counts as long (see `Store::long`).
pub const LONG: usize = 4096;

/// `path:line[:col]:text` → (end of the path, start of the text), byte offsets.
pub fn grep_shape(s: &str) -> Option<(u32, u32)> {
    let b = s.as_bytes();
    // the colons only (memchr), each followed by digits and a colon?
    for i in memchr::memchr_iter(b':', b) {
        if i >= 1 {
            let d0 = i + 1;
            let mut j = d0;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > d0 && j < b.len() && b[j] == b':' {
                let mut text = j + 1;
                let mut k = text;
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                if k > text && k < b.len() && b[k] == b':' {
                    text = k + 1;
                }
                return Some((i as u32, text as u32));
            }
        }
    }
    None
}

/// What the store keeps of one candidate besides its bytes.
struct Meta {
    sig: u32,
    flags: u8,
    grep: (u32, u32),
    trimmed: u32,
    long: bool,
}

fn meta(s: &str) -> Meta {
    let t = s.as_bytes();
    let (sig, high, ascii_upper) = scan(t);
    let ascii = !high;
    // bit 1: an upper-case letter (ASCII; any, on a non-ASCII line)
    let upper = ascii_upper || (high && s.chars().any(char::is_uppercase));
    // bit 2: non-ASCII, but every character one the UTF-8 group kernel
    // takes (and no NUL: the kernel's padding)
    let simple = high
        && !t.contains(&0)
        && s.chars()
            .all(|c| c.is_ascii() || crate::matcher::simple_utf8_char(c));
    let (trimmed, trimmed_bytes) = if ascii {
        // str::trim's white space within ASCII: 0x09..=0x0D and the space
        let ws = |b: &u8| matches!(b, 0x09..=0x0d | b' ');
        let a = t.iter().position(|b| !ws(b)).unwrap_or(t.len());
        let z = t.iter().rposition(|b| !ws(b)).map_or(a, |e| e + 1);
        (z - a, z - a)
    } else {
        let t = s.trim();
        (t.chars().count(), t.len())
    };
    Meta {
        sig: if high { u32::MAX } else { sig },
        flags: ascii as u8 | (upper as u8) << 1 | (simple as u8) << 2,
        grep: grep_shape(s).unwrap_or((NO_GREP, NO_GREP)),
        trimmed: trimmed as u32,
        long: trimmed_bytes > LONG,
    }
}

#[derive(Clone, Default)]
pub struct Store {
    data: Vec<u8>,
    /// End offset of each candidate (its start is the previous end).
    ends: Vec<u64>,
    sigs: Vec<u32>,
    /// bit 0: ASCII; bit 1: has an upper-case ASCII letter.
    flags: Vec<u8>,
    /// grep shape (path end, text start); NO_GREP when not a grep result.
    grep: Vec<(u32, u32)>,
    /// The length tiebreak: characters without the outer white space.
    trimmed: Vec<u32>,
    /// The candidates longer than LONG bytes, in index order (scored as tasks
    /// of their own, so one huge line does not hold a whole block back).
    long: Vec<u32>,
    /// A record cut at the end of the last chunk, waiting for the rest.
    pending: Vec<u8>,
    /// Skip empty records (a picker's behaviour; fzf keeps them).
    pub skip_empty: bool,
}

impl Store {
    pub fn new() -> Store {
        Store::default()
    }

    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Bytes held (texts only), for memory reporting.
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    pub fn reserve(&mut self, items: usize, bytes: usize) {
        self.data.reserve(bytes);
        self.ends.reserve(items);
        self.sigs.reserve(items);
        self.flags.reserve(items);
        self.grep.reserve(items);
        self.trimmed.reserve(items);
    }

    /// Add one candidate. Invalid UTF-8 is stored with replacement characters.
    pub fn push(&mut self, text: &[u8]) {
        if text.is_empty() && self.skip_empty {
            return;
        }
        match std::str::from_utf8(text) {
            Ok(_) => self.data.extend_from_slice(text),
            Err(_) => self
                .data
                .extend_from_slice(String::from_utf8_lossy(text).as_bytes()),
        }
        let start = self.ends.last().copied().unwrap_or(0) as usize;
        // SAFETY: what was just appended is valid UTF-8 (checked or replaced).
        let m = meta(unsafe { std::str::from_utf8_unchecked(&self.data[start..]) });
        self.add_meta(m);
        self.ends.push(self.data.len() as u64);
    }

    fn add_meta(&mut self, m: Meta) {
        self.sigs.push(m.sig);
        self.flags.push(m.flags);
        self.grep.push(m.grep);
        self.trimmed.push(m.trimmed);
        if m.long {
            self.long.push(self.ends.len() as u32);
        }
    }

    /// Add many candidates at once (the same as `push` for each), the
    /// per-candidate work and the copy spread over every core.
    pub fn extend_par(&mut self, texts: &[&str]) {
        use rayon::prelude::*;
        let kept: Vec<&str>;
        let texts = if self.skip_empty {
            kept = texts.iter().copied().filter(|t| !t.is_empty()).collect();
            &kept[..]
        } else {
            texts
        };
        if texts.is_empty() {
            return;
        }
        let base = self.data.len();
        let first = self.ends.len();
        // the end of each, then the bytes copied in parallel to their places
        let mut ends: Vec<u64> = Vec::with_capacity(texts.len());
        let mut at = base as u64;
        for t in texts {
            at += t.len() as u64;
            ends.push(at);
        }
        let total = at as usize - base;
        self.data.reserve(total);
        {
            let spare = &mut self.data.spare_capacity_mut()[..total];
            // whole runs of texts per task, each to its own slice
            let mut slices = Vec::new();
            let mut rest = spare;
            for chunk in texts.chunks(4096) {
                let len: usize = chunk.iter().map(|t| t.len()).sum();
                let (head, tail) = rest.split_at_mut(len);
                slices.push((head, chunk));
                rest = tail;
            }
            slices.into_par_iter().for_each(|(dst, chunk)| {
                let mut o = 0;
                for t in chunk {
                    // SAFETY: dst holds exactly the bytes of this run of texts;
                    // u8 and MaybeUninit<u8> have the same layout.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            t.as_ptr(),
                            dst.as_mut_ptr().add(o) as *mut u8,
                            t.len(),
                        )
                    };
                    o += t.len();
                }
            });
        }
        // SAFETY: the `total` bytes after the old end were written above.
        unsafe { self.data.set_len(base + total) };
        self.ends.extend_from_slice(&ends);
        // the metadata written in parallel straight to its places (no
        // intermediate list: fresh memory costs a page fault per 4 KiB)
        let n = texts.len();
        self.sigs.resize(first + n, 0);
        self.flags.resize(first + n, 0);
        self.grep.resize(first + n, (NO_GREP, NO_GREP));
        self.trimmed.resize(first + n, 0);
        let long: Vec<u32> = self.sigs[first..]
            .par_iter_mut()
            .zip(&mut self.flags[first..])
            .zip(&mut self.grep[first..])
            .zip(&mut self.trimmed[first..])
            .zip(texts)
            .enumerate()
            .filter_map(|(k, ((((sig, flags), grep), trimmed), t))| {
                let m = meta(t);
                *sig = m.sig;
                *flags = m.flags;
                *grep = m.grep;
                *trimmed = m.trimmed;
                m.long.then_some((first + k) as u32)
            })
            .collect();
        self.long.extend(long);
    }

    /// Add a chunk of a stream of `sep`-separated records (a process's
    /// output): a record cut at the end waits for the next chunk or `flush`,
    /// so any chunking gives the same candidates. A CR before a newline is
    /// dropped.
    pub fn append(&mut self, chunk: &[u8], sep: u8) {
        let mut from = 0;
        for p in memchr::memchr_iter(sep, chunk) {
            let mut rec = &chunk[from..p];
            if !self.pending.is_empty() {
                let mut whole = std::mem::take(&mut self.pending);
                whole.extend_from_slice(rec);
                self.push_record(&whole, sep);
                from = p + 1;
                continue;
            }
            if sep == b'\n' && rec.last() == Some(&b'\r') {
                rec = &rec[..rec.len() - 1];
            }
            self.push(rec);
            from = p + 1;
        }
        self.pending.extend_from_slice(&chunk[from..]);
    }

    fn push_record(&mut self, rec: &[u8], sep: u8) {
        let rec = if sep == b'\n' && rec.last() == Some(&b'\r') {
            &rec[..rec.len() - 1]
        } else {
            rec
        };
        self.push(rec);
    }

    /// The end of the stream: a last record without a separator.
    pub fn flush(&mut self, sep: u8) {
        if !self.pending.is_empty() {
            let rec = std::mem::take(&mut self.pending);
            self.push_record(&rec, sep);
        }
    }

    pub fn clear(&mut self) {
        let skip = self.skip_empty;
        *self = Store {
            skip_empty: skip,
            ..Store::default()
        };
    }

    /// Add every candidate of `other` after these (a part built elsewhere,
    /// e.g. on a reading thread). Taken as it is when this store is empty.
    pub fn absorb(&mut self, other: Store) {
        if self.is_empty() && self.pending.is_empty() {
            let skip = self.skip_empty;
            *self = other;
            self.skip_empty = skip;
            return;
        }
        let base = self.data.len() as u64;
        let first = self.ends.len() as u32;
        self.data.extend_from_slice(&other.data);
        self.ends.extend(other.ends.iter().map(|e| e + base));
        self.sigs.extend_from_slice(&other.sigs);
        self.flags.extend_from_slice(&other.flags);
        self.grep.extend_from_slice(&other.grep);
        self.trimmed.extend_from_slice(&other.trimmed);
        self.long.extend(other.long.iter().map(|i| i + first));
    }

    /// Keep only the last `n` candidates (fzf's --tail); indices start at 0 again.
    pub fn keep_last(&mut self, n: usize) {
        if self.len() <= n {
            return;
        }
        let drop = self.len() - n;
        let cut = self.ends[drop - 1];
        self.data.drain(..cut as usize);
        self.ends.drain(..drop);
        for e in &mut self.ends {
            *e -= cut;
        }
        self.sigs.drain(..drop);
        self.flags.drain(..drop);
        self.grep.drain(..drop);
        self.trimmed.drain(..drop);
        self.long.retain_mut(|i| {
            *i = i.wrapping_sub(drop as u32);
            (*i as usize) < n
        });
    }

    #[inline]
    pub fn text(&self, i: usize) -> &str {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        let end = self.ends[i] as usize;
        // SAFETY: only valid UTF-8 is ever stored (see push).
        unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) }
    }

    /// The raw tables, for the search's block loop: the data, the line ends,
    /// the signatures, the flags.
    #[inline]
    pub fn tables(&self) -> (&[u8], &[u64], &[u32], &[u8]) {
        (&self.data, &self.ends, &self.sigs, &self.flags)
    }

    /// Non-ASCII, every character simple for the UTF-8 group kernel (see
    /// `matcher::simple_utf8_char`).
    #[inline]
    pub fn is_simple_utf8(&self, i: usize) -> bool {
        self.flags[i] & 4 != 0
    }

    #[inline]
    pub fn sig(&self, i: usize) -> u32 {
        self.sigs[i]
    }

    #[inline]
    pub fn is_ascii(&self, i: usize) -> bool {
        self.flags[i] & 1 != 0
    }

    /// Has an upper-case letter: ASCII on an ASCII line, any on another.
    #[inline]
    pub fn has_upper(&self, i: usize) -> bool {
        self.flags[i] & 2 != 0
    }

    /// The candidates longer than [`LONG`] bytes, in index order.
    pub fn long(&self) -> &[u32] {
        &self.long
    }

    /// Characters without the outer white space (the length tiebreak).
    /// The trimmed lengths of all lines (see `trimmed_len`).
    #[inline]
    pub fn trimmed_table(&self) -> &[u32] {
        &self.trimmed
    }

    #[inline]
    pub fn trimmed_len(&self, i: usize) -> u32 {
        self.trimmed[i]
    }

    /// (end of the path, start of the text) for a grep result.
    #[inline]
    pub fn grep(&self, i: usize) -> Option<(u32, u32)> {
        let g = self.grep[i];
        (g.0 != NO_GREP).then_some(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_par_is_push_for_each() {
        let texts: Vec<String> = (0..20_000)
            .map(|i| match i % 7 {
                0 => String::new(),
                1 => format!("  src/main_{i}.rs \x0b\x0c "),
                2 => format!("търсене Ёж {i}"),
                3 => format!("a/b.rs:{i}:fn x"),
                4 => "x".repeat(LONG + 1 + i % 3),
                5 => format!("café {i}"),
                _ => format!("Plain Line {i}"),
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        for skip in [false, true] {
            let mut a = Store::new();
            a.skip_empty = skip;
            a.push(b"before");
            for t in &refs {
                a.push(t.as_bytes());
            }
            let mut b = Store::new();
            b.skip_empty = skip;
            b.push(b"before");
            b.extend_par(&refs[..5000]);
            b.extend_par(&refs[5000..]);
            assert_eq!(a.len(), b.len());
            assert_eq!(a.data, b.data);
            assert_eq!(a.ends, b.ends);
            assert_eq!(a.sigs, b.sigs);
            assert_eq!(a.flags, b.flags);
            assert_eq!(a.grep, b.grep);
            assert_eq!(a.trimmed, b.trimmed);
            assert_eq!(a.long, b.long);
        }
    }

    #[test]
    fn absorb_is_push_for_each() {
        let texts: Vec<String> = (0..3000)
            .map(|i| match i % 4 {
                0 => format!("a/b.rs:{i}:x"),
                1 => "y".repeat(LONG + 2),
                2 => format!("ёж {i}"),
                _ => String::new(),
            })
            .collect();
        let mut a = Store::new();
        for t in &texts {
            a.push(t.as_bytes());
        }
        let mut b = Store::new();
        for part in texts.chunks(700) {
            let mut p = Store::new();
            for t in part {
                p.push(t.as_bytes());
            }
            b.absorb(p);
        }
        assert_eq!(a.data, b.data);
        assert_eq!(a.ends, b.ends);
        assert_eq!(a.sigs, b.sigs);
        assert_eq!(a.flags, b.flags);
        assert_eq!(a.grep, b.grep);
        assert_eq!(a.trimmed, b.trimmed);
        assert_eq!(a.long, b.long);
    }

    #[test]
    fn simple_utf8_lines() {
        let mut s = Store::new();
        for t in [
            "ascii only",
            "търсене Ёж ёлка",
            "a → b ─ c — 中",
            "café",
            "a\u{a0}b",
            "Ђак",
            "αβγ",
            "😀 ок",
            "a\0б",
        ] {
            s.push(t.as_bytes());
        }
        let got: Vec<bool> = (0..s.len()).map(|i| s.is_simple_utf8(i)).collect();
        assert_eq!(
            got,
            [false, true, true, false, false, false, false, true, false]
        );
    }

    #[test]
    fn chunking_never_changes_the_records() {
        let input = b"src/a.rs:12:fn x\r\nb\n\nc\xffd\nlast";
        let mut whole = Store::new();
        whole.append(input, b'\n');
        whole.flush(b'\n');
        for cut in 0..input.len() {
            let mut s = Store::new();
            s.append(&input[..cut], b'\n');
            s.append(&input[cut..], b'\n');
            s.flush(b'\n');
            assert_eq!(s.len(), whole.len());
            for i in 0..s.len() {
                assert_eq!(s.text(i), whole.text(i), "cut at {cut}");
            }
        }
        assert_eq!(whole.text(0), "src/a.rs:12:fn x");
        assert_eq!(whole.grep(0), Some((8, 12)));
        assert_eq!(whole.text(2), "");
        assert_eq!(whole.text(3), "c\u{fffd}d");
        assert!(!whole.is_ascii(3));
        assert_eq!(whole.sig(3), u32::MAX);
        let mut skip = Store {
            skip_empty: true,
            ..Store::new()
        };
        skip.append(input, b'\n');
        skip.flush(b'\n');
        assert_eq!(skip.len(), whole.len() - 1);
    }

    #[test]
    fn keep_last() {
        let mut s = Store::new();
        for i in 0..10 {
            s.push(format!("item {i}").as_bytes());
        }
        s.keep_last(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.text(0), "item 7");
        assert_eq!(s.text(2), "item 9");
    }
}
