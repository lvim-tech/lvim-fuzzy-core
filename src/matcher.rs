//! Pattern parsing (fzf extended-search syntax) and scoring.
//!
//! The fuzzy scorer is the lvim-fuzzy model: an affine-gap local alignment
//! with first-char, word-boundary, camelCase and consecutive-run bonuses and
//! leading / inner / trailing gap penalties. Here it runs over Unicode scalar
//! values (the Neovim side is byte based) and can also recover the matched
//! positions for highlighting.

use std::cell::RefCell;

pub const BONUS_FIRST: i32 = 18;
pub const BONUS_BOUNDARY: i32 = 16;
pub const BONUS_CAMEL: i32 = 14;
pub const BONUS_CONSEC: i32 = 16;
pub const GAP_LEADING: i32 = -1;
pub const GAP_INNER: i32 = -2;
pub const GAP_TRAILING: i32 = -1;
const NEG: i32 = i32::MIN / 4;
/// Above this many cells (needle × line) the highlight positions are found
/// greedily instead of with full matrices (the score stays exact).
const MAX_CELLS: usize = 1 << 20;

/// The widest window (bytes or characters from the first possible match to
/// the last) scored exactly. A wider one — minified code, a huge log line —
/// is matched greedily in linear time: the leftmost occurrence, tightened
/// from its end backwards, scored by the same rules (docs/scoring.md).
pub const EXACT_WINDOW: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseMode {
    Smart,
    Ignore,
    Respect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Default,
    Path,
    History,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tiebreak {
    Length,
    Chunk,
    Pathname,
    Begin,
    End,
    Index,
}

#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub case: CaseMode,
    pub extended: bool,
    pub exact: bool,
    pub normalize: bool,
    pub scheme: Scheme,
    /// Fuzzy terms may miss this many of their characters (--typos); TYPOS_AUTO
    /// = by the term's length. 0 = fzf's behaviour.
    pub typos: u8,
    /// The widest match window scored exactly; a wider one is matched
    /// greedily in linear time. 0 = always exact. Default [`EXACT_WINDOW`].
    pub exact_window: usize,
}

/// The kinds of literal term (see `Pattern::single_literal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Literal {
    /// `^x`: at the start (after leading white space)
    Prefix,
    /// `x$`: at the end (before trailing white space)
    Suffix,
    /// `'x`: anywhere
    Exact,
}

/// `--typos=auto`: none up to 4 characters, 1 up to 8, 2 beyond.
pub const TYPOS_AUTO: u8 = u8::MAX;
/// Each missing character costs this much, so every exact match ranks before
/// any match with a typo of the same shape.
pub const TYPO_PENALTY: i32 = -40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Fuzzy,
    Exact,
    ExactBoundary,
    Prefix,
    Suffix,
    Equal,
}

#[derive(Clone, Debug)]
struct Term {
    kind: Kind,
    inverse: bool,
    text: Vec<char>,
    case_sensitive: bool,
    /// The term as bytes when it is all ASCII (for the quick pre-check);
    /// None = it cannot occur in ASCII text at all.
    ascii: Option<Box<[u8]>>,
    /// Each character of the term in UTF-8 with its other case (same length),
    /// for the pre-check on non-ASCII text; None = no safe pre-check (an ASCII
    /// letter that normalisation may match an accented letter for, or a case
    /// pair of different lengths).
    utf8: Option<Box<[Utf8Char]>>,
    /// How many characters this (fuzzy) term may miss.
    typos: u8,
}

/// One character of a term for the UTF-8 pre-check: its bytes and, when the
/// match ignores case, the bytes of the other case.
#[derive(Clone, Debug)]
struct Utf8Char {
    a: Box<[u8]>,
    b: Option<Box<[u8]>>,
}

/// Whether the ASCII `needle` occurs in `hay` in order. memchr jumps to each
/// next character (both cases when `hay` is not folded) instead of looking at
/// every byte.
#[inline]
fn subsequence(needle: &[u8], hay: &[u8], case_sensitive: bool) -> bool {
    let mut at = 0;
    for &c in needle {
        let rest = &hay[at.min(hay.len())..];
        let found = if case_sensitive || !c.is_ascii_lowercase() {
            memchr::memchr(c, rest)
        } else {
            memchr::memchr2(c, c.to_ascii_uppercase(), rest)
        };
        match found {
            Some(p) => at += p + 1,
            None => return false,
        }
    }
    true
}

/// ASCII `needle` (already folded unless `case_sensitive`) occurs in `hay`:
/// memchr on its first byte (both cases), then the rest compared in place.
#[inline]
fn contains(hay: &[u8], needle: &[u8], case_sensitive: bool) -> bool {
    let Some(&first) = needle.first() else {
        return true;
    };
    if case_sensitive {
        return memchr::memmem::find(hay, needle).is_some();
    }
    let n = needle.len();
    if n > hay.len() {
        return false;
    }
    let tail = &needle[1..];
    let limit = hay.len() - n + 1;
    let check = |p: usize| hay[p + 1..p + n].eq_ignore_ascii_case(tail);
    if first.is_ascii_lowercase() {
        memchr::memchr2_iter(first, first.to_ascii_uppercase(), &hay[..limit]).any(check)
    } else {
        memchr::memchr_iter(first, &hay[..limit]).any(check)
    }
}

/// `^x`, `x$`, `^x$` on ASCII: the line without its outer white space starts
/// with, ends with, or is `needle` (same trimming as the scorer).
#[inline]
fn anchored(kind: Kind, hay: &[u8], needle: &[u8], case_sensitive: bool) -> bool {
    let start = hay.iter().position(|c| !c.space()).unwrap_or(hay.len());
    let stop = hay
        .iter()
        .rposition(|c| !c.space())
        .map_or(start, |i| i + 1);
    let span = &hay[start..stop.max(start)];
    let n = needle.len();
    let eq = |s: &[u8]| {
        if case_sensitive {
            s == needle
        } else {
            s.eq_ignore_ascii_case(needle)
        }
    };
    match kind {
        Kind::Prefix => span.len() >= n && eq(&span[..n]),
        Kind::Suffix => span.len() >= n && eq(&span[span.len() - n..]),
        _ => eq(span),
    }
}

/// Whether `f` holds for some copy of `needle` with 1..=`k` of its units left
/// out (k <= 2): the candidates a term with typos may still match as.
fn with_typos<T: Copy>(needle: &[T], k: u8, mut f: impl FnMut(&[T]) -> bool) -> bool {
    let m = needle.len();
    let mut v: Vec<T> = Vec::with_capacity(m);
    for a in 0..m {
        v.clear();
        v.extend(
            needle
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != a)
                .map(|(_, &c)| c),
        );
        if f(&v) {
            return true;
        }
    }
    if k >= 2 {
        for a in 0..m {
            for b in a + 1..m {
                v.clear();
                v.extend(
                    needle
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != a && i != b)
                        .map(|(_, &c)| c),
                );
                if f(&v) {
                    return true;
                }
            }
        }
    }
    false
}

/// The best score of `needle` in `hay` with 1..=`k` of its units left out, the
/// penalty included (k <= 2; fewer typos are tried first and win ties).
fn fuzzy_typos<T: Ch>(
    chars: &[T],
    hay: &[T],
    needle: &[T],
    k: u8,
    path: bool,
    window: usize,
    mut pos: Option<&mut Vec<usize>>,
) -> Option<(i32, u32, u32)> {
    let m = needle.len();
    let mut best: Option<(i32, u32, u32, Vec<usize>)> = None;
    let mut v: Vec<T> = Vec::with_capacity(m);
    let try_variant = |v: &[T], typos: i32, best: &mut Option<(i32, u32, u32, Vec<usize>)>| {
        let mut p = Vec::new();
        let want = pos.is_some();
        if let Some((sc, b, e)) = fuzzy(
            chars,
            hay,
            v,
            path,
            window,
            if want { Some(&mut p) } else { None },
        ) {
            let sc = sc + typos * TYPO_PENALTY;
            if best.as_ref().is_none_or(|x| sc > x.0) {
                *best = Some((sc, b, e, p));
            }
        }
    };
    for a in 0..m {
        v.clear();
        v.extend(
            needle
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != a)
                .map(|(_, &c)| c),
        );
        try_variant(&v, 1, &mut best);
    }
    if k >= 2 && best.is_none() {
        for a in 0..m {
            for b in a + 1..m {
                v.clear();
                v.extend(
                    needle
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != a && i != b)
                        .map(|(_, &c)| c),
                );
                try_variant(&v, 2, &mut best);
            }
        }
    }
    let (sc, b, e, p) = best?;
    if let Some(out) = pos.as_deref_mut() {
        out.extend(p);
    }
    Some((sc, b, e))
}

/// Whether UTF-8 `b` has a letter that normalisation maps to ASCII (all of
/// them are in Latin-1 Supplement / Latin Extended-A: lead bytes C3, C4, C5).
#[inline]
fn has_latin_accents(b: &[u8]) -> bool {
    memchr::memchr3(0xC3, 0xC4, 0xC5, b).is_some()
}

/// The UTF-8 counterpart: every character of the term, in order, in `hay`.
/// The search goes by the LAST byte of a character: in Cyrillic (and most
/// two-byte scripts) every letter starts with the same byte or two, so the
/// first byte says nothing, while the last one tells the letters apart.
#[inline]
fn subsequence_utf8(needle: &[Utf8Char], hay: &[u8]) -> bool {
    let mut at = 0;
    for ch in needle {
        let la = *ch.a.last().unwrap();
        let lb = ch.b.as_ref().map(|b| *b.last().unwrap());
        let mut from = at;
        loop {
            let rest = &hay[from.min(hay.len())..];
            let found = match lb {
                Some(lb) if lb != la => memchr::memchr2(la, lb, rest),
                _ => memchr::memchr(la, rest),
            };
            let Some(p) = found else { return false };
            let end = from + p + 1; // just after the last byte
            let fits = |s: &[u8]| end >= at + s.len() && &hay[end - s.len()..end] == s;
            if fits(&ch.a) || ch.b.as_deref().is_some_and(fits) {
                at = end;
                break;
            }
            from = end;
        }
    }
    true
}

/// A parsed query: an AND of OR-groups of terms.
#[derive(Clone, Debug)]
pub struct Pattern {
    groups: Vec<Vec<Term>>,
    normalize: bool,
    path_scheme: bool,
    /// MatchConfig::exact_window, usize::MAX for 0 (always exact)
    window: usize,
}

/// Match outcome for one item.
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub score: i32,
    pub begin: u32,
    pub end: u32,
}

impl Pattern {
    pub fn parse(query: &str, cfg: &MatchConfig) -> Pattern {
        // as fzf: a query that has accented letters itself is matched as typed
        let local;
        let cfg = if cfg.normalize && query.chars().any(|c| normalize_char(c) != c) {
            local = MatchConfig {
                normalize: false,
                ..cfg.clone()
            };
            &local
        } else {
            cfg
        };
        let mut groups: Vec<Vec<Term>> = Vec::new();
        if !cfg.extended {
            let q = query;
            if !q.is_empty() {
                let kind = if cfg.exact { Kind::Exact } else { Kind::Fuzzy };
                groups.push(vec![make_term(q, kind, false, cfg)]);
            }
        } else {
            let mut join_next = false;
            for tok in split_terms(query) {
                if tok == "|" {
                    join_next = !groups.is_empty();
                    continue;
                }
                let Some(term) = parse_term(&tok, cfg) else {
                    continue;
                };
                if join_next {
                    if let Some(last) = groups.last_mut() {
                        last.push(term);
                    }
                    join_next = false;
                } else {
                    groups.push(vec![term]);
                }
            }
        }
        Pattern {
            groups,
            normalize: cfg.normalize,
            path_scheme: cfg.scheme == Scheme::Path,
            window: if cfg.exact_window == 0 {
                usize::MAX
            } else {
                cfg.exact_window
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The characters an ASCII candidate must have (as a presence signature,
    /// see [`crate::store::signature`]) and how many of those character
    /// classes may be missing (the terms' typos). Only lone, non-negated terms
    /// count: an OR group or a negation requires nothing for sure.
    pub fn signature(&self) -> (u32, u32) {
        let mut sig = 0u32;
        let mut allowed = 0u32;
        for group in &self.groups {
            if let [t] = group.as_slice()
                && !t.inverse
                && let Some(a) = &t.ascii
            {
                sig |= crate::store::signature(a);
                allowed += t.typos as u32;
            }
        }
        (sig, allowed)
    }

    /// A cheap necessary condition on ASCII text: every AND-group with no
    /// negated term needs one term whose characters occur in `text` in order.
    /// False means `text` cannot match; true decides nothing.
    /// The query as one plain fuzzy word, when it is exactly that (no
    /// operators, no OR, no typos, ASCII, at most 16 characters): the case the
    /// SIMD scorer handles. Returns (the word, folded unless case-sensitive;
    /// case-sensitive; the path scheme).
    /// The query as plain fuzzy words (AND) plus negated words, when it is
    /// exactly that: every word at most 16 characters of the Basic
    /// Multilingual Plane (no NUL), no typos; a negation must be a plain ASCII
    /// word (`may_match` decides it on ASCII text). The score is then the sum
    /// of the words' scores. Returns ([(word, case-sensitive)], path scheme,
    /// has a negation).
    pub fn plain_words(&self) -> Option<(Vec<(&[char], bool)>, bool, bool)> {
        let mut words = Vec::new();
        let mut neg = false;
        for group in &self.groups {
            let [t] = group.as_slice() else { return None };
            if t.inverse {
                if t.kind != Kind::Exact || t.ascii.is_none() {
                    return None;
                }
                neg = true;
                continue;
            }
            let w = &t.text[..];
            if t.kind != Kind::Fuzzy
                || t.typos != 0
                || w.is_empty()
                || w.len() > 16
                || w.iter().any(|&c| c == '\0' || c as u32 >= 0xFFFF)
            {
                return None;
            }
            words.push((w, t.case_sensitive));
        }
        (!words.is_empty()).then_some((words, self.path_scheme, neg))
    }

    /// The pattern as one literal term (`^x`, `x$`, `'x`) without negations
    /// or alternatives: (kind, the text as the pattern holds it — folded
    /// unless case-sensitive —, case-sensitive). None for anything else.
    pub fn single_literal(&self) -> Option<(Literal, &[char], bool)> {
        let [group] = self.groups.as_slice() else {
            return None;
        };
        let [t] = group.as_slice() else { return None };
        if t.inverse || t.text.is_empty() {
            return None;
        }
        let kind = match t.kind {
            Kind::Prefix => Literal::Prefix,
            Kind::Suffix => Literal::Suffix,
            Kind::Exact => Literal::Exact,
            _ => return None,
        };
        Some((kind, &t.text, t.case_sensitive))
    }

    /// Whether the path scheme's bonuses apply.
    pub fn path_scheme(&self) -> bool {
        self.path_scheme
    }

    /// Whether Latin letters with diacritics are matched by their base letter.
    pub fn normalizes(&self) -> bool {
        self.normalize
    }

    /// The cheap necessary checks (subsequence, literal terms, negations):
    /// false means `text` cannot match.
    pub fn may_match(&self, text: &str) -> bool {
        self.may_match_with(text.as_bytes(), text.is_ascii())
    }

    /// `may_match` for a text known to be ASCII.
    pub fn may_match_ascii(&self, b: &[u8]) -> bool {
        self.may_match_with(b, true)
    }

    fn may_match_with(&self, b: &[u8], ascii: bool) -> bool {
        'groups: for group in &self.groups {
            if group.iter().any(|t| t.inverse) {
                // a lone `!word` is decided exactly: the word must not occur
                if let [t] = group.as_slice()
                    && ascii
                    && t.kind == Kind::Exact
                    && let Some(a) = &t.ascii
                    && contains(b, a, t.case_sensitive)
                {
                    return false;
                }
                continue;
            }
            for t in group {
                let possible = if t.typos > 0 {
                    // a term that may miss characters: some variant must fit
                    !ascii
                        || t.ascii.as_ref().is_some_and(|a| {
                            subsequence(a, b, t.case_sensitive)
                                || with_typos(a, t.typos, |v| subsequence(v, b, t.case_sensitive))
                        })
                } else if ascii {
                    t.ascii.as_ref().is_some_and(|a| match t.kind {
                        // literal terms are decided here, on the raw bytes: no
                        // folding of the whole line, no scoring of a miss
                        Kind::Fuzzy => subsequence(a, b, t.case_sensitive),
                        Kind::Exact | Kind::ExactBoundary => contains(b, a, t.case_sensitive),
                        Kind::Prefix | Kind::Suffix | Kind::Equal => {
                            anchored(t.kind, b, a, t.case_sensitive)
                        }
                    })
                } else if let Some(u) = &t.utf8 {
                    subsequence_utf8(u, b)
                } else if let Some(a) = &t.ascii {
                    // an ASCII term in non-ASCII text: normalisation can only
                    // help through Latin letters with diacritics (lead bytes
                    // C3..C5); without any, the plain check decides
                    subsequence(a, b, t.case_sensitive) || (self.normalize && has_latin_accents(b))
                } else {
                    true
                };
                if possible {
                    continue 'groups;
                }
            }
            return false;
        }
        true
    }

    /// Score `text`, or None when it does not match.
    /// `score` for an ASCII text whose upper-case presence is already known
    /// (the store keeps both flags): no scan for them, and no folded copy when
    /// the text has no upper-case letter (it is its own folded form).
    pub fn score_ascii(&self, text: &str, has_upper: bool) -> Option<Score> {
        if !self.may_match_ascii(text.as_bytes()) {
            return None;
        }
        let b = text.as_bytes();
        if !has_upper {
            return self.score_units(b, b, None, |t| t.ascii.as_deref());
        }
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            s.bytes.clear();
            s.bytes.extend(b.iter().map(u8::to_ascii_lowercase));
            self.score_units(b, &s.bytes, None, |t| t.ascii.as_deref())
        })
    }

    pub fn score(&self, text: &str) -> Option<Score> {
        if !self.may_match(text) {
            return None;
        }
        if text.is_ascii() {
            // ASCII: scored on the bytes, without decoding (same scores)
            let b = text.as_bytes();
            return SCRATCH.with(|s| {
                let s = &mut *s.borrow_mut();
                s.bytes.clear();
                s.bytes.extend(b.iter().map(u8::to_ascii_lowercase));
                self.score_units(b, &s.bytes, None, |t| t.ascii.as_deref())
            });
        }
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            fill_chars(text, self.normalize, &mut s.chars, &mut s.folded);
            self.score_chars(&s.chars, &s.folded, None)
        })
    }

    /// Score `text` and collect the matched char indices.
    pub fn positions(&self, text: &str) -> Option<(Score, Vec<usize>)> {
        if !self.may_match(text) {
            return None;
        }
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            fill_chars(text, self.normalize, &mut s.chars, &mut s.folded);
            let mut pos = Vec::new();
            let sc = self.score_chars(&s.chars, &s.folded, Some(&mut pos))?;
            pos.sort_unstable();
            pos.dedup();
            Some((sc, pos))
        })
    }

    /// Char positions to highlight in a matching `text`: the scored alignment
    /// plus EVERY other occurrence of each (non-negated) term, so a query that
    /// appears in both the path and the text of a line is marked in both.
    pub fn highlights(&self, text: &str) -> Option<Vec<usize>> {
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            fill_chars(text, self.normalize, &mut s.chars, &mut s.folded);
            let mut pos = Vec::new();
            self.score_chars(&s.chars, &s.folded, Some(&mut pos))?;
            for term in self.groups.iter().flatten() {
                if term.inverse {
                    continue;
                }
                let hay = if term.case_sensitive {
                    &s.chars
                } else {
                    &s.folded
                };
                let n = term.text.len();
                if n == 0 || n > hay.len() {
                    continue;
                }
                match term.kind {
                    // anchored terms match once: the alignment already has them
                    Kind::Prefix | Kind::Suffix | Kind::Equal => {}
                    Kind::Fuzzy | Kind::Exact | Kind::ExactBoundary => {
                        let boundary = term.kind == Kind::ExactBoundary;
                        for p in 0..=hay.len() - n {
                            if hay[p..p + n] != term.text[..] {
                                continue;
                            }
                            if boundary {
                                let left = p == 0 || !s.chars[p - 1].is_alphanumeric();
                                let right =
                                    p + n == s.chars.len() || !s.chars[p + n].is_alphanumeric();
                                if !(left && right) {
                                    continue;
                                }
                            }
                            pos.extend(p..p + n);
                        }
                    }
                }
            }
            pos.sort_unstable();
            pos.dedup();
            Some(pos)
        })
    }

    fn score_chars(
        &self,
        chars: &[char],
        folded: &[char],
        pos: Option<&mut Vec<usize>>,
    ) -> Option<Score> {
        self.score_units(chars, folded, pos, |t| Some(&t.text[..]))
    }

    /// The scorer over chars or bytes; `needle` gives a term's text in the same
    /// unit (None: it cannot occur in this text).
    fn score_units<'t, T: Ch + 't>(
        &'t self,
        chars: &[T],
        folded: &[T],
        mut pos: Option<&mut Vec<usize>>,
        needle: impl Fn(&'t Term) -> Option<&'t [T]>,
    ) -> Option<Score> {
        let mut total = 0i32;
        let mut begin = u32::MAX;
        let mut end = 0u32;
        for group in &self.groups {
            let mut best: Option<(i32, u32, u32, Vec<usize>)> = None;
            let mut matched = false;
            for term in group {
                let hay = if term.case_sensitive { chars } else { folded };
                let text = needle(term);
                if term.inverse {
                    if text.is_none_or(|t| {
                        find_term(
                            term.kind,
                            0,
                            t,
                            chars,
                            hay,
                            self.path_scheme,
                            self.window,
                            None,
                        )
                        .is_none()
                    }) {
                        matched = true;
                        if best.is_none() {
                            best = Some((0, u32::MAX, 0, Vec::new()));
                        }
                    }
                    continue;
                }
                let Some(text) = text else { continue };
                let mut p = Vec::new();
                let want_pos = pos.is_some();
                if let Some((sc, b, e)) = find_term(
                    term.kind,
                    term.typos,
                    text,
                    chars,
                    hay,
                    self.path_scheme,
                    self.window,
                    if want_pos { Some(&mut p) } else { None },
                ) {
                    matched = true;
                    if best.as_ref().is_none_or(|x| sc > x.0) {
                        best = Some((sc, b, e, p));
                    }
                }
            }
            if !matched {
                return None;
            }
            if let Some((sc, b, e, p)) = best {
                total = total.saturating_add(sc);
                if b != u32::MAX {
                    begin = begin.min(b);
                    end = end.max(e);
                }
                if let Some(out) = pos.as_deref_mut() {
                    out.extend(p);
                }
            }
        }
        if begin == u32::MAX {
            begin = 0;
        }
        Some(Score {
            score: total,
            begin,
            end,
        })
    }
}

struct Scratch {
    chars: Vec<char>,
    folded: Vec<char>,
    /// The folded bytes of an ASCII text.
    bytes: Vec<u8>,
}

thread_local! {
    /// The two DP rows of the score-only path, reused between items.
    static DP_ROWS: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch { chars: Vec::new(), folded: Vec::new(), bytes: Vec::new() }) };
}

fn fill_chars(text: &str, normalize: bool, chars: &mut Vec<char>, folded: &mut Vec<char>) {
    chars.clear();
    folded.clear();
    for c in text.chars() {
        let c = if normalize { normalize_char(c) } else { c };
        chars.push(c);
        folded.push(fold(c));
    }
}

/// The bonus of a match at byte `j` of ASCII `hay` (the matcher's rule).
pub fn byte_bonus(hay: &[u8], j: usize, path: bool) -> i32 {
    bonus_at(hay, j, path)
}

/// The feasible window of ASCII `needle` (folded unless `case_sensitive`) in
/// `hay`: the first byte of the earliest in-order occurrence and the last
/// byte of the latest one. None when `needle` is not a subsequence.
pub fn window_ascii(hay: &[u8], needle: &[u8], case_sensitive: bool) -> Option<(usize, usize)> {
    if hay.len() <= MASK_MAX && needle.len() <= 16 {
        return window_masks(hay, needle, case_sensitive);
    }
    let find = |rest: &[u8], c: u8| {
        if case_sensitive || !c.is_ascii_lowercase() {
            memchr::memchr(c, rest)
        } else {
            memchr::memchr2(c, c.to_ascii_uppercase(), rest)
        }
    };
    let rfind = |rest: &[u8], c: u8| {
        if case_sensitive || !c.is_ascii_lowercase() {
            memchr::memrchr(c, rest)
        } else {
            memchr::memrchr2(c, c.to_ascii_uppercase(), rest)
        }
    };
    let mut at = 0;
    let mut first = 0;
    for (k, &c) in needle.iter().enumerate() {
        let p = at + find(&hay[at..], c)?;
        if k == 0 {
            first = p;
        }
        at = p + 1;
    }
    let mut end = hay.len();
    let mut last = 0;
    for (k, &c) in needle.iter().rev().enumerate() {
        let p = rfind(&hay[..end], c)?;
        if k == 0 {
            last = p;
        }
        end = p;
    }
    Some((first, last))
}

/// The longest line `window_ascii` handles with masks.
pub const MASK_MAX: usize = 256;

/// `window_ascii` for a line of at most [`MASK_MAX`] bytes: one pass builds,
/// for every needle character, a bit mask of where it occurs (AVX2 compares
/// when available; both cases at once when folding), and the earliest /
/// latest in-order occurrence come from bit scans on the masks — no repeated
/// searches over the line.
fn window_masks(hay: &[u8], needle: &[u8], case_sensitive: bool) -> Option<(usize, usize)> {
    const W: usize = MASK_MAX / 64;
    let n = hay.len();
    let words = n.div_ceil(64);
    let mut tail = [0u8; 32];
    tail[..n % 32].copy_from_slice(&hay[n - n % 32..]);
    let mask_of = |c: u8, out: &mut [u64; W]| {
        let other = if case_sensitive || !c.is_ascii_lowercase() {
            c
        } else {
            c.to_ascii_uppercase()
        };
        #[cfg(target_arch = "x86_64")]
        if crate::simd::available() {
            // SAFETY: AVX2 checked; the masks are read straight from the line
            // (the last partial 32 bytes through a small buffer).
            unsafe { crate::simd::eq2_masks(hay, &tail, c, other, &mut out[..words]) };
            return;
        }
        for k in 0..words {
            let end = (k * 64 + 64).min(n);
            out[k] = hay[k * 64..end]
                .iter()
                .enumerate()
                .fold(0u64, |m, (j, &b)| {
                    m | (((b == c || b == other) as u64) << j)
                });
        }
    };
    // the first set bit at or after `from`
    let next = |m: &[u64; W], from: usize| -> Option<usize> {
        let mut k = from / 64;
        if k >= words {
            return None;
        }
        let mut x = m[k] & (u64::MAX << (from % 64));
        loop {
            if x != 0 {
                return Some(k * 64 + x.trailing_zeros() as usize);
            }
            k += 1;
            if k >= words {
                return None;
            }
            x = m[k];
        }
    };
    // the last set bit before `to`
    let prev = |m: &[u64; W], to: usize| -> Option<usize> {
        if to == 0 {
            return None;
        }
        let last = to - 1;
        let mut k = last / 64;
        let sh = 63 - last % 64;
        let mut x = (m[k] << sh) >> sh;
        loop {
            if x != 0 {
                return Some(k * 64 + 63 - x.leading_zeros() as usize);
            }
            if k == 0 {
                return None;
            }
            k -= 1;
            x = m[k];
        }
    };
    let mut masks = [[0u64; W]; 16];
    for (k, &c) in needle.iter().enumerate() {
        mask_of(c, &mut masks[k]);
    }
    let m = needle.len();
    let mut at = 0;
    let mut first = 0;
    for (k, mask) in masks.iter().enumerate().take(m) {
        let p = next(mask, at)?;
        if k == 0 {
            first = p;
        }
        at = p + 1;
    }
    let mut end = n;
    let mut last = 0;
    for k in (0..m).rev() {
        let p = prev(&masks[k], end)?;
        if k == m - 1 {
            last = p;
        }
        end = p;
    }
    Some((first, last))
}

/// The rebased recursion (docs/scoring.md) over one window, the scalar
/// reference of the SIMD kernel: `fold[k]` / `bonus[k]` are the folded byte
/// and the match bonus of window column k, `needle` the folded needle.
/// Returns the best value of the last row, NEG-ish when there is no
/// alignment; the score is that minus (line length - 1).
pub fn rebased_window(fold: &[i16], bonus: &[i16], needle: &[i16]) -> i32 {
    const NEGV: i32 = i16::MIN as i32 / 2;
    let w = fold.len();
    let m = needle.len();
    let mut d = vec![NEGV; w];
    let mut mm = vec![NEGV; w];
    let mut best = NEGV;
    for (i, &want) in needle.iter().enumerate() {
        let mut nd = vec![NEGV; w];
        let mut nm = vec![NEGV; w];
        for j in 0..w {
            let b = bonus[j] as i32;
            let dv = if fold[j] != want {
                NEGV
            } else if i == 0 {
                b
            } else {
                let (f, c) = if j == 0 {
                    (NEGV, NEGV)
                } else {
                    (mm[j - 1], d[j - 1])
                };
                (f + b + 1).max(c + b + 17)
            };
            nd[j] = dv.max(NEGV);
            if i == m - 1 {
                best = best.max(dv);
            } else {
                let prev = if j == 0 { NEGV } else { nm[j - 1] - 1 };
                nm[j] = dv.max(prev).max(NEGV);
            }
        }
        d = nd;
        mm = nm;
    }
    best
}

/// A character folded to lower case (as the matcher compares them).
pub fn fold_char(c: char) -> char {
    fold(c)
}

/// A non-ASCII character the UTF-8 group kernel handles by its bytes:
/// basic Cyrillic (А-я, Ё, ё: two bytes, lead D0 / D1, a case), or a
/// character without case that is no space, is not normalized and not folded
/// (arrows, box drawing, dashes, CJK: never a boundary, letter or digit for
/// the bonus rules, and never equal to a needle character of the kernel).
pub fn simple_utf8_char(c: char) -> bool {
    matches!(c, 'А'..='я' | 'Ё' | 'ё')
        || (!c.is_ascii()
            && !c.is_lowercase()
            && !c.is_uppercase()
            && !c.is_whitespace()
            && normalize_char(c) == c
            && fold(c) == c)
}

/// The bonus for a match on `cur` right after `prev` (the matcher's rule; a
/// match on the first character gets BONUS_FIRST instead).
pub fn char_bonus(prev: char, cur: char, path: bool) -> i32 {
    bonus_at(&[prev, cur], 1, path)
}

#[inline]
fn fold(c: char) -> char {
    match c {
        _ if c.is_ascii() => c.to_ascii_lowercase(),
        // Cyrillic, without the general Unicode tables
        'А'..='Я' => char::from_u32(c as u32 + 0x20).unwrap_or(c),
        'Ѐ'..='Џ' => char::from_u32(c as u32 + 0x50).unwrap_or(c),
        'а'..='џ' => c,
        _ => c.to_lowercase().next().unwrap_or(c),
    }
}

/// Split a query on unescaped spaces; `\ ` is a literal space.
fn split_terms(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = query.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' && it.peek() == Some(&' ') {
            cur.push(' ');
            it.next();
        } else if c == ' ' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_term(tok: &str, cfg: &MatchConfig) -> Option<Term> {
    let mut t = tok;
    let mut inverse = false;
    let mut kind = if cfg.exact { Kind::Exact } else { Kind::Fuzzy };
    if t.len() > 1 && t.starts_with('!') {
        inverse = true;
        kind = Kind::Exact;
        t = &t[1..];
    }
    if t != "$" && t.len() > 1 && t.ends_with('$') {
        kind = Kind::Suffix;
        t = &t[..t.len() - 1];
    }
    if t.starts_with('\'') {
        if t.len() > 2 && t.ends_with('\'') {
            kind = Kind::ExactBoundary;
            t = &t[1..t.len() - 1];
        } else if cfg.exact && !inverse {
            kind = Kind::Fuzzy;
            t = &t[1..];
        } else {
            kind = Kind::Exact;
            t = &t[1..];
        }
    } else if t.starts_with('^') {
        kind = if kind == Kind::Suffix {
            Kind::Equal
        } else {
            Kind::Prefix
        };
        t = &t[1..];
    }
    if t.is_empty() {
        return None;
    }
    Some(make_term(t, kind, inverse, cfg))
}

fn make_term(t: &str, kind: Kind, inverse: bool, cfg: &MatchConfig) -> Term {
    let case_sensitive = match cfg.case {
        CaseMode::Respect => true,
        CaseMode::Ignore => false,
        CaseMode::Smart => t.chars().any(|c| c.is_uppercase()),
    };
    let text = t
        .chars()
        .map(|c| if cfg.normalize { normalize_char(c) } else { c })
        .map(|c| if case_sensitive { c } else { fold(c) })
        .collect::<Vec<char>>();
    let ascii = text
        .iter()
        .all(char::is_ascii)
        .then(|| text.iter().map(|&c| c as u8).collect());
    let utf8 = utf8_chars(&text, case_sensitive, cfg.normalize);
    let typos = if kind != Kind::Fuzzy || inverse {
        0
    } else if cfg.typos == TYPOS_AUTO {
        match text.len() {
            0..=4 => 0,
            5..=8 => 1,
            _ => 2,
        }
    } else {
        // never so many that a term could match with nothing
        cfg.typos.min(text.len().saturating_sub(1) as u8)
    };
    Term {
        kind,
        inverse,
        text,
        case_sensitive,
        ascii,
        utf8,
        typos,
    }
}

/// The term for the UTF-8 pre-check, or None when no check is safe: with
/// normalisation an ASCII letter also matches its accented forms (é for e),
/// and a case pair must have the same UTF-8 length.
fn utf8_chars(text: &[char], case_sensitive: bool, normalize: bool) -> Option<Box<[Utf8Char]>> {
    let enc = |c: char| -> Box<[u8]> {
        let mut buf = [0u8; 4];
        c.encode_utf8(&mut buf).as_bytes().into()
    };
    let mut out = Vec::with_capacity(text.len());
    for &c in text {
        if normalize && c.is_ascii_alphabetic() {
            return None;
        }
        let other = if case_sensitive {
            None
        } else {
            let mut up = c.to_uppercase();
            match (up.next(), up.next()) {
                (Some(u), None) if u != c => {
                    if u.len_utf8() != c.len_utf8() {
                        return None;
                    }
                    Some(enc(u))
                }
                (Some(_), None) => None,
                _ => return None,
            }
        };
        out.push(Utf8Char {
            a: enc(c),
            b: other,
        });
    }
    Some(out.into())
}

/// A unit of text the scorer runs over: a `char` (any text) or a byte (ASCII
/// text, scored without decoding it). For ASCII both give the same answers, so
/// the two paths give the same scores.
trait Ch: Copy + PartialEq + 'static {
    /// A number for set membership (the byte, or the code point).
    fn key(self) -> u32;
    /// The first `c` in `hay`.
    fn find(hay: &[Self], c: Self) -> Option<usize> {
        hay.iter().position(|&x| x == c)
    }
    /// The last `c` in `hay`.
    fn rfind(hay: &[Self], c: Self) -> Option<usize> {
        hay.iter().rposition(|&x| x == c)
    }
    fn boundary(self) -> bool;
    fn slash(self) -> bool;
    fn lower(self) -> bool;
    fn upper(self) -> bool;
    fn digit(self) -> bool;
    fn space(self) -> bool;
    fn alnum(self) -> bool;
}

impl Ch for char {
    #[inline]
    fn key(self) -> u32 {
        self as u32
    }
    #[inline]
    fn boundary(self) -> bool {
        matches!(
            self,
            '/' | '_' | '-' | '.' | ' ' | ':' | ',' | ';' | '|' | '\t'
        ) || self.is_whitespace()
    }
    #[inline]
    fn slash(self) -> bool {
        self == '/'
    }
    #[inline]
    fn lower(self) -> bool {
        self.is_lowercase()
    }
    #[inline]
    fn upper(self) -> bool {
        self.is_uppercase()
    }
    #[inline]
    fn digit(self) -> bool {
        self.is_ascii_digit()
    }
    #[inline]
    fn space(self) -> bool {
        self.is_whitespace()
    }
    #[inline]
    fn alnum(self) -> bool {
        self.is_alphanumeric()
    }
}

impl Ch for u8 {
    #[inline]
    fn key(self) -> u32 {
        self as u32
    }
    #[inline]
    fn find(hay: &[u8], c: u8) -> Option<usize> {
        memchr::memchr(c, hay)
    }
    #[inline]
    fn rfind(hay: &[u8], c: u8) -> Option<usize> {
        memchr::memrchr(c, hay)
    }
    #[inline]
    fn boundary(self) -> bool {
        matches!(
            self,
            b'/' | b'_' | b'-' | b'.' | b' ' | b':' | b',' | b';' | b'|' | b'\t'
        ) || self.is_ascii_whitespace()
            || self == 0x0b
    }
    #[inline]
    fn slash(self) -> bool {
        self == b'/'
    }
    #[inline]
    fn lower(self) -> bool {
        self.is_ascii_lowercase()
    }
    #[inline]
    fn upper(self) -> bool {
        self.is_ascii_uppercase()
    }
    #[inline]
    fn digit(self) -> bool {
        self.is_ascii_digit()
    }
    #[inline]
    fn space(self) -> bool {
        self.is_ascii_whitespace() || self == 0x0b
    }
    #[inline]
    fn alnum(self) -> bool {
        self.is_ascii_alphanumeric()
    }
}

#[inline]
fn bonus_at<T: Ch>(chars: &[T], j: usize, path: bool) -> i32 {
    if j == 0 {
        return BONUS_FIRST;
    }
    let prev = chars[j - 1];
    let cur = chars[j];
    if prev.boundary() {
        if path && prev.slash() {
            BONUS_BOUNDARY + 2
        } else {
            BONUS_BOUNDARY
        }
    } else if prev.lower() && cur.upper() {
        BONUS_CAMEL
    } else if !prev.digit() && cur.digit() {
        BONUS_CAMEL / 2
    } else {
        0
    }
}

/// Find a term in `hay` (case-folded or not) and score it against `chars`
/// (for bonuses). Returns (score, begin, end) where end is exclusive.
fn find_term<T: Ch>(
    kind: Kind,
    typos: u8,
    text: &[T],
    chars: &[T],
    hay: &[T],
    path: bool,
    window: usize,
    pos: Option<&mut Vec<usize>>,
) -> Option<(i32, u32, u32)> {
    let n = text.len();
    match kind {
        Kind::Fuzzy if typos > 0 => {
            // exact first; only a miss tries the variants with typos
            let mut pos = pos;
            match fuzzy(chars, hay, text, path, window, pos.as_deref_mut()) {
                Some(r) => Some(r),
                None => fuzzy_typos(chars, hay, text, typos.min(2), path, window, pos),
            }
        }
        Kind::Fuzzy => fuzzy(chars, hay, text, path, window, pos),
        Kind::Exact | Kind::ExactBoundary => {
            let boundary = kind == Kind::ExactBoundary;
            if n > hay.len() {
                return None;
            }
            let mut best: Option<(i32, usize)> = None;
            let first = text[0];
            for p in 0..=hay.len() - n {
                // the first unit before the whole slice: most positions stop here
                if hay[p] != first || hay[p..p + n] != text[..] {
                    continue;
                }
                if boundary {
                    let left_ok = p == 0 || !chars[p - 1].alnum();
                    let right_ok = p + n == chars.len() || !chars[p + n].alnum();
                    if !(left_ok && right_ok) {
                        continue;
                    }
                }
                let sc = segment_score(chars, p, n, path);
                if best.is_none_or(|b| sc > b.0) {
                    best = Some((sc, p));
                }
            }
            let (sc, p) = best?;
            if let Some(out) = pos {
                out.extend(p..p + n);
            }
            Some((sc, p as u32, (p + n) as u32))
        }
        Kind::Prefix | Kind::Suffix | Kind::Equal => {
            let start = hay.iter().position(|c| !c.space()).unwrap_or(hay.len());
            let stop = hay
                .iter()
                .rposition(|c| !c.space())
                .map_or(start, |i| i + 1);
            let span = &hay[start..stop.max(start)];
            let p = match kind {
                Kind::Prefix if span.starts_with(text) => start,
                Kind::Suffix if span.ends_with(text) => stop - n,
                Kind::Equal if span == text => start,
                _ => return None,
            };
            if let Some(out) = pos {
                out.extend(p..p + n);
            }
            Some((segment_score(chars, p, n, path), p as u32, (p + n) as u32))
        }
    }
}

fn segment_score<T: Ch>(chars: &[T], p: usize, n: usize, path: bool) -> i32 {
    let mut sc = p as i32 * GAP_LEADING;
    for k in 0..n {
        sc += bonus_at(chars, p + k, path);
        if k > 0 {
            sc += BONUS_CONSEC;
        }
    }
    sc + (chars.len() - p - n) as i32 * GAP_TRAILING
}

/// Affine-gap fuzzy alignment. With `pos`, keeps the full DP and backtracks.
/// The leftmost in-order occurrence of `needle` from `first`, tightened from
/// its end backwards (each character as late as possible before the next):
/// a compact alignment in linear time.
fn greedy_positions<T: Ch>(hay: &[T], needle: &[T], first: usize) -> Option<Vec<usize>> {
    let m = needle.len();
    let mut at = first;
    let mut end = 0;
    for &c in needle {
        let p = at + T::find(&hay[at..], c)?;
        at = p + 1;
        end = at;
    }
    let mut tight = vec![0usize; m];
    for (k, &c) in needle.iter().enumerate().rev() {
        let p = T::rfind(&hay[..end], c)?;
        tight[k] = p;
        end = p;
    }
    Some(tight)
}

/// The score of an alignment at `positions` in a line of `n` characters —
/// exactly what the matrix gives that alignment: bonuses, consecutive
/// bonus, leading / inner / trailing gaps.
fn positions_score<T: Ch>(chars: &[T], n: usize, positions: &[usize], path: bool) -> i32 {
    let mut sc = positions[0] as i32 * GAP_LEADING;
    for (k, &p) in positions.iter().enumerate() {
        sc += bonus_at(chars, p, path);
        if k > 0 {
            let gap = p - positions[k - 1] - 1;
            if gap == 0 {
                sc += BONUS_CONSEC;
            } else {
                sc += gap as i32 * GAP_INNER;
            }
        }
    }
    sc + (n - 1 - positions[positions.len() - 1]) as i32 * GAP_TRAILING
}

fn fuzzy<T: Ch>(
    chars: &[T],
    hay: &[T],
    needle: &[T],
    path: bool,
    window: usize,
    pos: Option<&mut Vec<usize>>,
) -> Option<(i32, u32, u32)> {
    let m = needle.len();
    let n = hay.len();
    if m == 0 || m > n {
        return None;
    }
    // Prefilter: subsequence check, also gives the earliest and latest
    // feasible window for the DP (jumps from character to character; memchr /
    // memrchr on bytes).
    let mut at = 0;
    let mut first = usize::MAX;
    for (ni, &c) in needle.iter().enumerate() {
        let p = at + T::find(&hay[at..], c)?;
        if ni == 0 {
            first = p;
        }
        at = p + 1;
    }
    let mut end = n;
    let mut last = 0;
    for (k, &c) in needle.iter().rev().enumerate() {
        let p = T::rfind(&hay[..end], c)?;
        if k == 0 {
            last = p;
        }
        end = p;
    }
    let lastrow = m - 1;
    if m > 1 && last - first + 1 > window {
        let tight = greedy_positions(hay, needle, first)?;
        let sc = positions_score(chars, n, &tight, path);
        if let Some(out) = pos {
            out.extend(&tight);
        }
        return Some((sc, tight[0] as u32, (tight[m - 1] + 1) as u32));
    }
    if pos.is_none() {
        if m == 1 {
            // one character: the best single position, its leading and
            // trailing gaps; no matrix
            let mut best = NEG;
            for j in first..=last {
                if hay[j] == needle[0] {
                    let sc = j as i32 * GAP_LEADING
                        + bonus_at(chars, j, path)
                        + (n - 1 - j) as i32 * GAP_TRAILING;
                    best = best.max(sc);
                }
            }
            return Some((best, 0, (last + 1) as u32));
        }
        // Only the window [first, last] can hold the alignment: before it the
        // first needle character never occurs, after it the last one does not,
        // so the last row only collects trailing gaps there.
        // The two rows live in one reused buffer (no allocation per item), and
        // are updated in place from the last needle position down, so each
        // cell still reads the previous column's values of the row above.
        let sc = DP_ROWS.with(|r| {
            let r = &mut *r.borrow_mut();
            r.clear();
            r.resize(2 * m, NEG);
            let (d, mrow) = r.split_at_mut(m);
            // Columns whose character is not in the needle only add each row's
            // gap: they are counted and the gaps applied in one step at the next
            // column that matters (no bonus, no per-cell work for them).
            // the needle's characters as a bit set (one test per column instead
            // of a search through the needle); code points past 255 fall back
            let mut set = [0u64; 4];
            let mut wide = false;
            for &c in needle {
                let k = c.key();
                if k < 256 {
                    set[(k >> 6) as usize] |= 1 << (k & 63);
                } else {
                    wide = true;
                }
            }
            let in_needle = |c: T| {
                let k = c.key();
                if k < 256 {
                    set[(k >> 6) as usize] >> (k & 63) & 1 == 1
                } else {
                    wide && needle.contains(&c)
                }
            };
            let mut skipped = 0i32;
            for j in first..=last {
                let c = hay[j];
                if !in_needle(c) {
                    skipped += 1;
                    continue;
                }
                if skipped > 0 {
                    for i in 0..m {
                        let gap = if i == lastrow {
                            GAP_TRAILING
                        } else {
                            GAP_INNER
                        };
                        if mrow[i] > NEG / 2 {
                            mrow[i] += skipped * gap;
                        }
                        d[i] = NEG;
                    }
                    skipped = 0;
                }
                let b = bonus_at(chars, j, path);
                for i in (0..m).rev() {
                    let gap = if i == lastrow {
                        GAP_TRAILING
                    } else {
                        GAP_INNER
                    };
                    let dval = if c == needle[i] {
                        let (fresh, consec) = if i == 0 {
                            (j as i32 * GAP_LEADING, NEG)
                        } else {
                            (mrow[i - 1], d[i - 1])
                        };
                        (fresh + b).max(consec + b + BONUS_CONSEC)
                    } else {
                        NEG
                    };
                    mrow[i] = dval.max(mrow[i] + gap);
                    d[i] = dval;
                }
            }
            let mut sc = mrow[lastrow];
            if sc > NEG / 2 {
                sc += skipped * GAP_TRAILING;
            }
            sc
        });
        if sc <= NEG / 2 {
            return None;
        }
        let sc = sc + (n - 1 - last) as i32 * GAP_TRAILING;
        return Some((sc, 0, (last + 1) as u32));
    }
    // A huge line (minified code, a log line) would need a huge matrix just to
    // place the highlights: then the score still comes exact from the
    // matrix-free pass, and only the positions from a linear greedy pass (the
    // leftmost occurrence, then tightened from its end backwards).
    if m * n > MAX_CELLS {
        let (sc, _, _) = fuzzy(chars, hay, needle, path, window, None)?;
        let out = pos.unwrap();
        let mut at = first;
        let mut fwd = Vec::with_capacity(m);
        for &c in needle {
            let p = at + T::find(&hay[at..], c)?;
            fwd.push(p);
            at = p + 1;
        }
        // tighten: from the last match backwards, each needle char as late as possible
        let mut end = fwd[m - 1] + 1;
        let mut tight = vec![0usize; m];
        for (k, &c) in needle.iter().enumerate().rev() {
            let p = T::rfind(&hay[..end], c)?;
            tight[k] = p;
            end = p;
        }
        out.extend(&tight);
        return Some((sc, tight[0] as u32, (tight[m - 1] + 1) as u32));
    }
    // Full matrices for backtracking.
    let mut d = vec![NEG; m * n];
    let mut mm = vec![NEG; m * n];
    for j in 0..n {
        let b = bonus_at(chars, j, path);
        let c = hay[j];
        for i in 0..m {
            let dval = if c == needle[i] {
                let fresh = if i == 0 {
                    j as i32 * GAP_LEADING
                } else if j > 0 {
                    mm[(i - 1) * n + j - 1]
                } else {
                    NEG
                };
                let consec = if i > 0 && j > 0 {
                    d[(i - 1) * n + j - 1]
                } else {
                    NEG
                };
                (fresh + b).max(consec + b + BONUS_CONSEC)
            } else {
                NEG
            };
            let gap = if i == lastrow {
                GAP_TRAILING
            } else {
                GAP_INNER
            };
            let prev = if j > 0 { mm[i * n + j - 1] } else { NEG };
            d[i * n + j] = dval;
            mm[i * n + j] = dval.max(prev + gap);
        }
    }
    let sc = mm[lastrow * n + n - 1];
    if sc <= NEG / 2 {
        return None;
    }
    let out = pos.unwrap();
    let start_len = out.len();
    let mut i = lastrow as isize;
    let mut j = n as isize - 1;
    let mut must_match = false;
    while i >= 0 && j >= 0 {
        let (iu, ju) = (i as usize, j as usize);
        let dv = d[iu * n + ju];
        let take = if must_match {
            true
        } else {
            dv > NEG / 2 && dv == mm[iu * n + ju]
        };
        if take {
            out.push(ju);
            let b = bonus_at(chars, ju, path);
            must_match = iu > 0 && ju > 0 && dv == d[(iu - 1) * n + ju - 1] + b + BONUS_CONSEC;
            i -= 1;
            j -= 1;
        } else {
            j -= 1;
        }
    }
    let got = &out[start_len..];
    let begin = got.iter().copied().min().unwrap_or(0) as u32;
    let end = got.iter().copied().max().map_or(0, |e| e + 1) as u32;
    Some((sc, begin, end))
}

/// Map common Latin letters with diacritics to their base letter (fzf's
/// default normalisation; `--literal` turns it off).
pub fn normalize_char(c: char) -> char {
    // everything mapped below is in U+00C0..=U+017E
    if c.is_ascii() || c > 'ž' {
        return c;
    }
    match c {
        'À'..='Å' | 'Ā' | 'Ă' | 'Ą' => 'A',
        'à'..='å' | 'ā' | 'ă' | 'ą' => 'a',
        'Ç' | 'Ć' | 'Ĉ' | 'Ċ' | 'Č' => 'C',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'Ď' | 'Đ' => 'D',
        'ď' | 'đ' => 'd',
        'È'..='Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => 'E',
        'è'..='ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'Ĝ' | 'Ğ' | 'Ġ' | 'Ģ' => 'G',
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
        'Ĥ' | 'Ħ' => 'H',
        'ĥ' | 'ħ' => 'h',
        'Ì'..='Ï' | 'Ĩ' | 'Ī' | 'Ĭ' | 'Į' | 'İ' => 'I',
        'ì'..='ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
        'Ĵ' => 'J',
        'ĵ' => 'j',
        'Ķ' => 'K',
        'ķ' => 'k',
        'Ĺ' | 'Ļ' | 'Ľ' | 'Ŀ' | 'Ł' => 'L',
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
        'Ñ' | 'Ń' | 'Ņ' | 'Ň' => 'N',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'Ò'..='Ö' | 'Ø' | 'Ō' | 'Ŏ' | 'Ő' => 'O',
        'ò'..='ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'Ŕ' | 'Ŗ' | 'Ř' => 'R',
        'ŕ' | 'ŗ' | 'ř' => 'r',
        'Ś' | 'Ŝ' | 'Ş' | 'Š' => 'S',
        'ś' | 'ŝ' | 'ş' | 'š' => 's',
        'Ţ' | 'Ť' | 'Ŧ' => 'T',
        'ţ' | 'ť' | 'ŧ' => 't',
        'Ù'..='Ü' | 'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => 'U',
        'ù'..='ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'Ŵ' => 'W',
        'ŵ' => 'w',
        'Ý' | 'Ŷ' | 'Ÿ' => 'Y',
        'ý' | 'ÿ' | 'ŷ' => 'y',
        'Ź' | 'Ż' | 'Ž' => 'Z',
        'ź' | 'ż' | 'ž' => 'z',
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MatchConfig {
        MatchConfig {
            case: CaseMode::Smart,
            extended: true,
            exact: false,
            normalize: true,
            scheme: Scheme::Default,
            typos: 0,
            exact_window: EXACT_WINDOW,
        }
    }

    fn m(q: &str, t: &str) -> bool {
        Pattern::parse(q, &cfg()).score(t).is_some()
    }

    #[test]
    fn fuzzy_and_case() {
        assert!(m("abc", "a_b_c"));
        assert!(m("abc", "ABC"));
        assert!(!m("Abc", "abc"));
        assert!(!m("abd", "abc"));
    }

    #[test]
    fn extended_syntax() {
        assert!(m("'bc", "abcd"));
        assert!(!m("'bd", "abcd"));
        assert!(m("^ab", "  abcd"));
        assert!(!m("^bc", "abcd"));
        assert!(m("cd$", "abcd"));
        assert!(m("^abcd$", "abcd"));
        assert!(m("!xyz", "abcd"));
        assert!(!m("!bc", "abcd"));
        assert!(m("xyz | bc", "abcd"));
        assert!(!m("ab xyz", "abcd"));
        assert!(m("'foo'", "a foo b"));
        assert!(!m("'foo'", "afoob"));
        assert!(m("a\\ b", "a b"));
    }

    #[test]
    fn normalisation() {
        assert!(m("cafe", "café"));
        let lit = MatchConfig {
            normalize: false,
            ..cfg()
        };
        assert!(Pattern::parse("cafe", &lit).score("café").is_none());
    }

    #[test]
    fn positions_match_score() {
        let p = Pattern::parse("fb", &cfg());
        let (s1, pos) = p.positions("foo/bar").unwrap();
        let s2 = p.score("foo/bar").unwrap();
        assert_eq!(s1.score, s2.score);
        assert_eq!(pos, vec![0, 4]);
    }

    #[test]
    fn highlights_every_occurrence() {
        let p = Pattern::parse("health", &cfg());
        let t = "lvim-ts/health.lua:21: h.health(x)";
        let hl = p.highlights(t).unwrap();
        // both "health" in the path (8..14) and in the text (25..31)
        assert!((8..14).all(|i| hl.contains(&i)));
        assert!((25..31).all(|i| hl.contains(&i)));
        // a fuzzy-only match still gets its alignment
        assert_eq!(
            Pattern::parse("fb", &cfg()).highlights("foo/bar").unwrap(),
            vec![0, 4]
        );
        assert!(p.highlights("nothing").is_none());
    }

    /// The full scorer over chars, bypassing every fast path.
    fn slow(p: &Pattern, t: &str) -> Option<i32> {
        SCRATCH.with(|s| {
            let s = &mut *s.borrow_mut();
            fill_chars(t, p.normalize, &mut s.chars, &mut s.folded);
            p.score_chars(&s.chars, &s.folded, None).map(|x| x.score)
        })
    }

    #[test]
    fn fast_paths_score_like_the_full_scorer() {
        // a small LCG: the same pseudo-random texts and queries every run
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        let alpha = b"abcdefgABCDEF/_-. :0123xyzXYZ";
        let words = [
            "a", "ab", "Ab", "abc", "f/b", "x1", "'ab", "^a", "c$", "a !b", "AB", "a b", "zz", "1",
            "'ab'",
        ];
        let mut checked = 0;
        for path in [false, true] {
            let c = MatchConfig {
                scheme: if path { Scheme::Path } else { Scheme::Default },
                ..cfg()
            };
            for q in words {
                let p = Pattern::parse(q, &c);
                for _ in 0..400 {
                    let len = rnd(40);
                    let t: String = (0..len).map(|_| alpha[rnd(alpha.len())] as char).collect();
                    let fast = p.score(&t).map(|x| x.score);
                    assert_eq!(fast, slow(&p, &t), "query {q:?} on {t:?}");
                    // the full matrix (whole row, no window, no shortcuts) as the reference
                    assert_eq!(
                        fast,
                        p.positions(&t).map(|x| x.0.score),
                        "full DP: query {q:?} on {t:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 2 * words.len() * 400);
        // non-ASCII text (the char path), against the full matrix
        let alpha: Vec<char> = "абвгдежАБВГ/_- .éèaA1".chars().collect();
        for q in [
            "а", "аб", "Аб", "вг", "а !б", "'аб", "^а", "г$", "e", "ae", "zz",
        ] {
            let p = Pattern::parse(q, &cfg());
            for _ in 0..400 {
                let len = rnd(30);
                let t: String = (0..len).map(|_| alpha[rnd(alpha.len())]).collect();
                let fast = p.score(&t).map(|x| x.score);
                assert_eq!(
                    fast,
                    p.positions(&t).map(|x| x.0.score),
                    "query {q:?} on {t:?}"
                );
            }
        }
    }

    #[test]
    fn prefilters_agree_with_matching() {
        // the pre-checks may only reject what cannot match
        let texts = [
            "Плъгин за търсене",
            "ПЛЪГИН",
            "плъгин/конфигурация.lua",
            "café crème",
            "src/main.rs",
            "Ünïcödé ✓ mix",
            "",
        ];
        let queries = [
            "плъгин",
            "ПЛЪГ",
            "пгн",
            "тър !кон",
            "cafe",
            "café",
            "src",
            "SRC",
            "ün",
            "✓",
            "zz",
            "ия",
        ];
        for q in queries {
            let p = Pattern::parse(q, &cfg());
            for t in texts {
                let full = SCRATCH.with(|s| {
                    let s = &mut *s.borrow_mut();
                    fill_chars(t, p.normalize, &mut s.chars, &mut s.folded);
                    p.score_chars(&s.chars, &s.folded, None)
                });
                assert_eq!(p.score(t).is_some(), full.is_some(), "query {q:?} on {t:?}");
            }
        }
        assert!(subsequence_utf8(
            &utf8_chars(&['п', 'г'], false, true).unwrap(),
            "ПЛЪГИН".as_bytes()
        ));
        assert!(!subsequence_utf8(
            &utf8_chars(&['г', 'п'], false, true).unwrap(),
            "плъгин".as_bytes()
        ));
    }

    #[test]
    fn huge_lines_keep_exact_scores() {
        // a 1 MB "minified" line: positions must not build a 1 MB × m matrix
        let mut t = "x".repeat(500_000);
        t.push_str("lvim/fuzzy");
        t.push_str(&"y".repeat(500_000));
        let p = Pattern::parse("lvfz", &cfg());
        let (sc, pos) = p.positions(&t).unwrap();
        assert_eq!(Some(sc.score), p.score(&t).map(|x| x.score));
        assert_eq!(pos.len(), 4);
        assert!(pos.windows(2).all(|w| w[0] < w[1]));
        let b: Vec<char> = t.chars().collect();
        let got: String = pos.iter().map(|&i| b[i]).collect();
        assert_eq!(got, "lvfz");
    }

    #[test]
    fn typos() {
        let with = |n: u8| MatchConfig { typos: n, ..cfg() };
        // "lvmi" is not in "lvim" in order; with one typo it is ("lvm" or "lvi")
        assert!(Pattern::parse("lvmi", &cfg()).score("lvim").is_none());
        let p = Pattern::parse("lvmi", &with(1));
        assert!(p.score("lvim").is_some());
        // an exact match always ranks before a match with a typo
        let exact = p.score("lvmi.rs").unwrap().score;
        let typo = p.score("lvim.rs").unwrap().score;
        assert!(exact > typo, "{exact} vs {typo}");
        // positions agree with the score
        assert_eq!(p.positions("lvim.rs").map(|x| x.0.score), Some(typo));
        // non-ASCII text, two typos
        let q = Pattern::parse("плъгинн", &with(2));
        assert!(q.score("плъгин").is_some());
        // auto: short terms get none
        let a = MatchConfig {
            typos: TYPOS_AUTO,
            ..cfg()
        };
        assert!(Pattern::parse("lvmi", &a).score("lvim").is_none());
        assert!(
            Pattern::parse("lvimfuzzzy", &a)
                .score("lvimfuzzy")
                .is_some()
        );
        // literal and negated terms are never fuzzy about it
        assert!(Pattern::parse("'lvmi", &with(1)).score("lvim").is_none());
        assert!(Pattern::parse("!lvmi", &with(1)).score("lvim").is_some());
    }

    #[test]
    fn rebased_window_scores_like_fuzzy() {
        // every string up to length 7 over a small alphabet, needles up to 4
        let alpha = b"aAb/_1 ";
        let needles: &[&[u8]] = &[
            b"a", b"ab", b"aa", b"ba", b"a1", b"ab_", b"aab", b"abab", b"b/a",
        ];
        let mut count = 0;
        for len in 1..=7usize {
            let total = alpha.len().pow(len as u32);
            for code in 0..total {
                let mut x = code;
                let hay: Vec<u8> = (0..len)
                    .map(|_| {
                        let c = alpha[x % alpha.len()];
                        x /= alpha.len();
                        c
                    })
                    .collect();
                let folded: Vec<u8> = hay.iter().map(u8::to_ascii_lowercase).collect();
                for &nd in needles {
                    for path in [false, true] {
                        let want =
                            fuzzy::<u8>(&hay, &folded, nd, path, usize::MAX, None).map(|r| r.0);
                        let got = window_ascii(&hay, nd, false).map(|(f, l)| {
                            let fold: Vec<i16> = folded[f..=l].iter().map(|&c| c as i16).collect();
                            let bonus: Vec<i16> =
                                (f..=l).map(|j| byte_bonus(&hay, j, path) as i16).collect();
                            let need: Vec<i16> = nd.iter().map(|&c| c as i16).collect();
                            rebased_window(&fold, &bonus, &need) - (hay.len() as i32 - 1)
                        });
                        assert_eq!(
                            got,
                            want,
                            "needle {:?} in {:?}",
                            std::str::from_utf8(nd),
                            std::str::from_utf8(&hay)
                        );
                        count += 1;
                    }
                }
            }
        }
        assert!(count > 1_000_000, "{count}");
    }

    #[test]
    fn mask_window_equals_the_search_window() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha = b"abcABC/ x";
        for _ in 0..200_000 {
            let n = 1 + rnd(MASK_MAX);
            let hay: Vec<u8> = (0..n).map(|_| alpha[rnd(alpha.len())]).collect();
            let m = 1 + rnd(5);
            let cs = rnd(4) == 0;
            let needle: Vec<u8> = (0..m)
                .map(|_| {
                    let c = alpha[rnd(alpha.len())];
                    if cs { c } else { c.to_ascii_lowercase() }
                })
                .collect();
            // the memchr search, the reference
            let reference = {
                let f = |c: u8, h: &[u8]| {
                    if cs || !c.is_ascii_lowercase() {
                        memchr::memchr(c, h)
                    } else {
                        memchr::memchr2(c, c.to_ascii_uppercase(), h)
                    }
                };
                let r = |c: u8, h: &[u8]| {
                    if cs || !c.is_ascii_lowercase() {
                        memchr::memrchr(c, h)
                    } else {
                        memchr::memrchr2(c, c.to_ascii_uppercase(), h)
                    }
                };
                (|| {
                    let mut at = 0;
                    let mut first = 0;
                    for (k, &c) in needle.iter().enumerate() {
                        let p = at + f(c, &hay[at..])?;
                        if k == 0 {
                            first = p;
                        }
                        at = p + 1;
                    }
                    let mut end = hay.len();
                    let mut last = 0;
                    for (k, &c) in needle.iter().rev().enumerate() {
                        let p = r(c, &hay[..end])?;
                        if k == 0 {
                            last = p;
                        }
                        end = p;
                    }
                    Some((first, last))
                })()
            };
            assert_eq!(
                window_masks(&hay, &needle, cs),
                reference,
                "{:?} in {:?}",
                needle,
                hay
            );
        }
    }

    #[test]
    fn positions_score_is_the_matrix_score_of_its_alignment() {
        // the matrix's best alignment, scored by the formula, gives the matrix's score
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let alpha = b"abAB/_ 1";
        for _ in 0..50_000 {
            let n = 1 + rnd(40);
            let hay: Vec<u8> = (0..n).map(|_| alpha[rnd(alpha.len())]).collect();
            let folded: Vec<u8> = hay.iter().map(u8::to_ascii_lowercase).collect();
            let needle: Vec<u8> = (0..1 + rnd(4)).map(|_| b"ab/_1"[rnd(5)]).collect();
            let path = rnd(2) == 0;
            let mut pos = Vec::new();
            if let Some((sc, _, _)) =
                fuzzy::<u8>(&hay, &folded, &needle, path, usize::MAX, Some(&mut pos))
            {
                pos.sort_unstable(); // the backtracking gives them from the end
                assert_eq!(
                    positions_score::<u8>(&hay, n, &pos, path),
                    sc,
                    "{:?} in {:?}",
                    needle,
                    hay
                );
            }
        }
    }

    #[test]
    fn wide_windows_are_matched_greedily() {
        let mut hay = String::from("x/foo");
        hay.push_str(&"z".repeat(EXACT_WINDOW + 10));
        hay.push_str("bar");
        let p = Pattern::parse("fobar", &cfg());
        let sc = p.score(&hay).unwrap();
        let (_, pos) = p.positions(&hay).unwrap();
        // leftmost "fo" + "bar", tightened from the end: the later "o"
        assert_eq!(pos, vec![2, 4, hay.len() - 3, hay.len() - 2, hay.len() - 1]);
        let chars: Vec<char> = hay.chars().collect();
        assert_eq!(
            sc.score,
            positions_score::<char>(&chars, chars.len(), &pos, false)
        );
        // exact_window 0: always the matrix — the best alignment ("fo"
        // together), so at least the greedy score; here 16 more (consecutive)
        let exact = Pattern::parse(
            "fobar",
            &MatchConfig {
                exact_window: 0,
                ..cfg()
            },
        );
        assert_eq!(exact.score(&hay).map(|s| s.score), Some(sc.score + 16));
    }

    #[test]
    fn boundary_beats_middle() {
        let p = Pattern::parse("b", &cfg());
        assert!(p.score("a/b").unwrap().score > p.score("abc").unwrap().score);
    }
}
