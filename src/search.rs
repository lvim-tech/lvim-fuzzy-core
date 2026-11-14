//! Searching a [`Store`]: every candidate on every core, the ordered result,
//! cancellation, narrowing from the previous result while typing, grep scopes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use crate::matcher::{MatchConfig, Pattern, Tiebreak};
use crate::store::Store;

/// One match in the result: the candidate's index and its score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub idx: u32,
    pub score: i32,
}

/// A match while the search runs: with the tiebreak keys the ordering needs
/// (some come from the match itself: begin, end), dropped from the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Scored {
    idx: u32,
    score: i32,
    keys: [u32; 3],
}

impl Scored {
    #[inline]
    fn hit(&self) -> Hit {
        Hit {
            idx: self.idx,
            score: self.score,
        }
    }
}

/// The ordered result.
#[derive(Clone, Debug)]
pub enum Hits {
    /// Every candidate, in input order (or reversed with `tac`): an empty query.
    All {
        count: usize,
        tac: bool,
    },
    List(Arc<Vec<Hit>>),
}

impl Hits {
    pub fn len(&self) -> usize {
        match self {
            Hits::All { count, .. } => *count,
            Hits::List(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The candidate at rank `i`.
    pub fn get(&self, i: usize) -> Option<u32> {
        match self {
            Hits::All { count, tac } => (i < *count).then(|| {
                if *tac {
                    (*count - 1 - i) as u32
                } else {
                    i as u32
                }
            }),
            Hits::List(v) => v.get(i).map(|h| h.idx),
        }
    }

    /// The rank of candidate `idx`.
    pub fn position(&self, idx: u32) -> Option<usize> {
        match self {
            Hits::All { count, tac } => {
                let i = idx as usize;
                (i < *count).then(|| if *tac { *count - 1 - i } else { i })
            }
            Hits::List(v) => v.iter().position(|h| h.idx == idx),
        }
    }

    /// The matching candidates in index order (for narrowing the next search).
    pub fn indices(&self) -> Option<Vec<u32>> {
        match self {
            Hits::All { .. } => None,
            Hits::List(v) => {
                let mut i: Vec<u32> = v.iter().map(|h| h.idx).collect();
                i.par_sort_unstable();
                Some(i)
            }
        }
    }
}

/// What the filter reads in a grep result (`path:line:text`); other
/// candidates are always read whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The whole candidate.
    All,
    /// The line's text, not the path.
    Line,
    /// The file path only.
    Path,
    /// The whole content of the file (every term exact).
    File,
}

impl Scope {
    pub fn parse(s: &str) -> Option<Scope> {
        Some(match s {
            "all" => Scope::All,
            "line" => Scope::Line,
            "path" => Scope::Path,
            "file" => Scope::File,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Scope::All => "all",
            Scope::Line => "line",
            Scope::Path => "path",
            Scope::File => "file",
        }
    }

    /// line → path → file → line
    pub fn next(self) -> Scope {
        match self {
            Scope::Line => Scope::Path,
            Scope::Path => Scope::File,
            Scope::File | Scope::All => Scope::Line,
        }
    }
}

/// How to search.
#[derive(Clone, Debug)]
pub struct Options {
    pub cfg: MatchConfig,
    /// Order by score (else keep the input order).
    pub sort: bool,
    /// Reverse the input order (also the last tiebreak).
    pub tac: bool,
    pub tiebreak: Vec<Tiebreak>,
    pub scope: Scope,
}

/// Whether every match of `new` is also a match of `old`, so `new` only
/// needs to look at what `old` found: `new` extends `old` and neither has a
/// negation or an OR (which let in what `old` left out) nor an escape.
pub fn narrows(old: &str, new: &str) -> bool {
    !old.is_empty()
        && new.len() > old.len()
        && new.starts_with(old)
        && ![old, new].iter().any(|q| q.contains(['!', '|', '\\']))
}

/// The text the filter reads for candidate `i` under `scope`.
pub enum Target<'a> {
    Borrowed(&'a str),
    File(Arc<str>),
}

impl std::ops::Deref for Target<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            Target::Borrowed(s) => s,
            Target::File(s) => s,
        }
    }
}

pub fn scoped(store: &Store, i: usize, scope: Scope) -> Target<'_> {
    let text = store.text(i);
    let Some((path_end, text_start)) = store.grep(i) else {
        return Target::Borrowed(text);
    };
    match scope {
        Scope::All => Target::Borrowed(text),
        Scope::Line => Target::Borrowed(&text[text_start as usize..]),
        Scope::Path => Target::Borrowed(&text[..path_end as usize]),
        Scope::File => match file_text(&text[..path_end as usize]) {
            Some(t) => Target::File(t),
            None => Target::Borrowed(""),
        },
    }
}

/// Files read for the `file` scope, kept for the session. Large (> 2 MiB)
/// and binary files count as empty.
fn file_text(path: &str) -> Option<Arc<str>> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static CACHE: OnceLock<RwLock<HashMap<String, Option<Arc<str>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(v) = cache.read().unwrap().get(path) {
        return v.clone();
    }
    let v = std::fs::metadata(path)
        .ok()
        .filter(|m| m.is_file() && m.len() <= 2 << 20)
        .and_then(|_| std::fs::read(path).ok())
        .filter(|b| !b[..b.len().min(8192)].contains(&0))
        .map(|b| Arc::<str>::from(String::from_utf8_lossy(&b).as_ref()));
    cache.write().unwrap().insert(path.to_string(), v.clone());
    v
}

fn tiebreak_key(t: Tiebreak, text: &str, begin: u32, end: u32, index: u32) -> u32 {
    match t {
        Tiebreak::Length => text.trim().chars().count() as u32,
        Tiebreak::Begin => begin,
        Tiebreak::End => (text.chars().count() as u32).saturating_sub(end),
        Tiebreak::Index => index,
        Tiebreak::Chunk => {
            let chars: Vec<char> = text.chars().collect();
            let (b, e) = (begin as usize, (end as usize).min(chars.len()));
            let mut s = b;
            while s > 0 && !chars[s - 1].is_whitespace() {
                s -= 1;
            }
            let mut x = e;
            while x < chars.len() && !chars[x].is_whitespace() {
                x += 1;
            }
            (x - s) as u32
        }
        Tiebreak::Pathname => {
            let chars: Vec<char> = text.chars().collect();
            let base = chars
                .iter()
                .rposition(|&c| c == '/' || c == '\\')
                .map_or(0, |p| p + 1) as u32;
            if begin >= base {
                begin - base
            } else {
                (1 << 30) + (base - begin)
            }
        }
    }
}

/// The ordered hits of `query` in `store`, or None when cancelled (the
/// generation in `cancel` moved on). `subset` = (the previous result's
/// matches in index order, the first candidate added since): search only
/// those — valid when [`narrows`] says so.
pub fn search(
    store: &Store,
    query: &str,
    o: &Options,
    cancel: Option<(&AtomicU64, u64)>,
    subset: Option<(&[u32], usize)>,
) -> Option<Hits> {
    // Over a whole file a fuzzy subsequence matches almost anything, so the
    // `file` scope reads every term as an exact occurrence.
    let pat = if o.scope == Scope::File {
        Pattern::parse(
            query,
            &MatchConfig {
                exact: true,
                ..o.cfg.clone()
            },
        )
    } else {
        Pattern::parse(query, &o.cfg)
    };
    if pat.is_empty() {
        return Some(Hits::All {
            count: store.len(),
            tac: o.tac,
        });
    }
    let (need, allowed) = pat.signature();
    // the signature is of the whole candidate: a scoped grep result reads a part
    let use_sig = need != 0 && o.scope == Scope::All;
    let needs_pos = o.tiebreak.iter().any(|t| {
        matches!(
            t,
            Tiebreak::Begin | Tiebreak::End | Tiebreak::Chunk | Tiebreak::Pathname
        )
    });
    let cancelled = || cancel.is_some_and(|(a, g)| a.load(Ordering::Relaxed) != g);
    let score_full = |i: usize| -> Option<Scored> {
        if use_sig && (need & !store.sig(i)).count_ones() > allowed {
            return None;
        }
        let target = scoped(store, i, o.scope);
        let text: &str = &target;
        let sc = if needs_pos {
            pat.positions(text).map(|(s, _)| s)
        } else if o.scope == Scope::All && store.is_ascii(i) {
            pat.score_ascii(text, store.has_upper(i))
        } else {
            pat.score(text)
        }?;
        let mut keys = [0u32; 3];
        for (k, t) in o.tiebreak.iter().take(3).enumerate() {
            keys[k] = if *t == Tiebreak::Length && o.scope == Scope::All {
                store.trimmed_len(i)
            } else {
                tiebreak_key(*t, text, sc.begin, sc.end, i as u32)
            };
        }
        Some(Scored {
            idx: i as u32,
            score: sc.score,
            keys,
        })
    };
    let t0 = std::time::Instant::now();
    // long candidates are scored afterwards, each as its own task
    let split_long = !store.long().is_empty();
    let score_one = |i: usize| -> Option<Scored> {
        if split_long && store.text(i).len() > crate::store::LONG {
            return None;
        }
        score_full(i)
    };
    let n = store.len();
    // plain words: ASCII candidates along the line (score_window), others
    // 16 at a time in tiles (one character: the scalar closed form)
    let simd = (o.scope == Scope::All && !needs_pos && crate::simd::available())
        .then(|| pat.plain_words())
        .flatten();
    let one_char = simd
        .as_ref()
        .is_some_and(|(w, _, _)| w.len() == 1 && w[0].0.len() == 1);
    // long lines stay on their own tasks: in a block they hold up one thread,
    // and the scalar path's work follows the needle, not the window
    let long_ascii_inline = false;
    // the words as ASCII, once (None: a word that cannot occur in ASCII text)
    let ascii_words: Option<Vec<AsciiWord>> = simd.as_ref().and_then(|(words, _, _)| {
        words
            .iter()
            .map(|&(w, cs)| {
                w.iter().all(char::is_ascii).then(|| AsciiWord {
                    bytes: w.iter().map(|&c| c as u8).collect(),
                    needle: w.iter().map(|&c| c as i16).collect(),
                    cs,
                })
            })
            .collect()
    });
    // the words for the UTF-8 group kernel (None: a character it does not take)
    let utf8_words: Option<Vec<Utf8Word>> = simd.as_ref().and_then(|(words, _, _)| {
        words
            .iter()
            .map(|&(w, cs)| {
                let chars: Option<Vec<_>> =
                    w.iter().map(|&c| crate::simd::utf8_char(c, cs)).collect();
                chars.map(|chars| Utf8Word { chars })
            })
            .collect()
    });
    // the kernel stops where the matcher would turn greedy (the user's
    // exact_window, 0 = never) or at its own limit
    let window_max = match o.cfg.exact_window {
        0 => KERNEL_MAX,
        w => w.min(KERNEL_MAX),
    };
    let score_block = |ids: &mut dyn Iterator<Item = usize>| -> Vec<Scored> {
        let Some((words, path, has_neg)) = &simd else {
            return ids.filter_map(score_one).collect();
        };
        // every result goes to its candidate's position in the block, so the
        // block comes out in index order without sorting; the buffers are kept
        // per thread
        let mut bufs = BLOCK_BUFS.with(|b| std::mem::take(&mut *b.borrow_mut()));
        let mut winbufs = WIN_BUFS.with(|b| std::mem::take(&mut *b.borrow_mut()));
        let win = &mut winbufs;
        let BlockBufs {
            slots,
            cands,
            raw,
            fold,
            bonus,
        } = &mut bufs;
        slots.clear();
        cands.clear();
        let need_raw = words.iter().any(|w| w.1);
        let need_fold = words.iter().any(|w| !w.1);
        for i in ids {
            let p = slots.len() as u32;
            slots.push(None);
            if use_sig && (need & !store.sig(i)).count_ones() > allowed {
                continue;
            }
            let text = store.text(i);
            let ascii = store.is_ascii(i);
            if ascii && !(split_long && text.len() > crate::store::LONG) {
                // ASCII: scored right here, along the line (score_window), in
                // index order — no tile, no sorting
                // the window search decides a plain word; a negation needs may_match
                if *has_neg && !pat.may_match_ascii(text.as_bytes()) {
                    continue;
                }
                slots[p as usize] = match &ascii_words {
                    Some(aw) => {
                        score_ascii_words(store, i, aw, *path, window_max, &o.tiebreak, win)
                            .unwrap_or_else(|()| score_full(i))
                    }
                    None => None, // a word with non-ASCII characters: no match in ASCII text
                };
                continue;
            }
            // long lines: the scalar path, whose work follows the
            // needle's characters, not the length (a tile of 1-2 long lines wastes it)
            if split_long && text.len() > crate::store::LONG {
                continue; // a task of its own, below
            }
            if *has_neg || one_char || text.len() > crate::simd::TILE_MAX {
                slots[p as usize] = score_one(i);
                continue;
            }
            if !(if ascii {
                pat.may_match_ascii(text.as_bytes())
            } else {
                pat.may_match(text)
            }) {
                continue;
            }
            let len = if ascii {
                text.len()
            } else {
                text.chars().count()
            };
            if len == 0 {
                continue;
            }
            // tiles by length rounded up to 8: full registers even when the
            // lengths vary (the padding is added back below)
            cands.push((len.div_ceil(8) as u32 * 8, len as u32, p, i as u32));
        }
        cands.sort_unstable();
        for run in cands.chunk_by(|a, b| a.0 == b.0) {
            let width = run[0].0 as usize;
            let cells = width * crate::simd::LANES;
            for group in run.chunks(crate::simd::LANES) {
                for v in [&mut *raw, &mut *fold] {
                    v.clear();
                    v.resize(cells, 0);
                }
                bonus.clear();
                bonus.resize(cells, 0);
                for lane in 0..crate::simd::LANES {
                    // a short last group repeats its last candidate (lanes ignored)
                    let (_, _, _, idx) = group[lane.min(group.len() - 1)];
                    crate::simd::fill_lane(
                        store.text(idx as usize),
                        lane,
                        pat.normalizes(),
                        *path,
                        need_raw.then_some(&mut raw[..]),
                        need_fold.then_some(&mut fold[..]),
                        bonus,
                    );
                }
                // every word scored over the same tile; the score is their sum,
                // and each word paid GAP_TRAILING (-1) per padded character
                let mut sums = [0i32; crate::simd::LANES];
                // a lane matches only if every word does (the prefilter is a
                // necessary condition, not a sufficient one)
                let mut all = [true; crate::simd::LANES];
                for &(word, cs) in words {
                    let needle: Vec<u16> = word.iter().map(|&c| c as u16).collect();
                    let keys = if cs { &raw[..] } else { &fold[..] };
                    let mut scores = [0i16; crate::simd::LANES];
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 checked by simd::available(); keys and bonus
                    // hold width * 16 cells; the word is 1..=16 characters,
                    // none 0 (plain_words); width <= MAX_LEN + 7.
                    unsafe {
                        crate::simd::score_tile(keys, bonus, width, &needle, &mut scores)
                    };
                    for ((s, ok), &x) in sums.iter_mut().zip(all.iter_mut()).zip(&scores) {
                        *s += x as i32;
                        *ok &= (x as i32) > -16000;
                    }
                }
                for (lane, &(_, len, p, idx)) in group.iter().enumerate() {
                    if !all[lane] {
                        continue;
                    }
                    let i = idx as usize;
                    let pad = (width as u32 - len) as i32 * words.len() as i32;
                    let mut keys = [0u32; 3];
                    for (k, t) in o.tiebreak.iter().take(3).enumerate() {
                        keys[k] = if *t == Tiebreak::Length {
                            store.trimmed_len(i)
                        } else {
                            tiebreak_key(*t, store.text(i), 0, 0, idx)
                        };
                    }
                    slots[p as usize] = Some(Scored {
                        idx,
                        score: sums[lane] + pad,
                        keys,
                    });
                }
            }
        }
        let out: Vec<Scored> = slots.iter().flatten().copied().collect();
        BLOCK_BUFS.with(|b| *b.borrow_mut() = bufs);
        WIN_BUFS.with(|b| *b.borrow_mut() = winbufs);
        out
    };
    // plain ASCII words: the group kernel (32 lines at a time) for the short
    // ASCII lines, the paths above for the rest
    let fast = (allowed == 0 || !use_sig)
        .then_some(())
        .and(simd.as_ref())
        .zip(utf8_words.as_ref());
    // one literal term (^x, x$, 'x) of ASCII / basic Cyrillic characters
    let literal = (o.scope == Scope::All && !needs_pos)
        .then(|| pat.single_literal())
        .flatten()
        .and_then(|(kind, text, cs)| {
            LitQuery::new(
                kind,
                text,
                cs,
                pat.path_scheme(),
                if use_sig { need } else { 0 },
                &o.tiebreak,
            )
        });
    let block = |ids: Ids| -> Option<Vec<Scored>> {
        if let Some(lq) = &literal {
            return Some(literal_block(store, ids, lq, split_long, &|i| score_one(i)));
        }
        match fast {
            #[cfg(target_arch = "x86_64")]
            Some(((_, path, has_neg), uw)) => {
                let f = FastQuery {
                    need: if use_sig { need } else { 0 },
                    ascii: ascii_words.as_deref(),
                    utf8: uw,
                    path: *path,
                    neg: has_neg.then_some(&pat),
                    tiebreak: &o.tiebreak,
                };
                // SAFETY: AVX2 checked (simd is Some only then).
                unsafe { block_fast(store, ids, &f, &cancelled, &mut |rest| score_block(rest)) }
            }
            _ => Some(match ids {
                Ids::Range(a, b) => score_block(&mut (a..b)),
                Ids::List(l, n) => {
                    score_block(&mut l.iter().map(|&i| i as usize).filter(|&i| i < n))
                }
            }),
        }
    };
    let parts: Vec<Option<Vec<Scored>>> = match subset {
        Some((prev, from)) => {
            let fresh: Vec<u32> = (from as u32..n as u32).collect();
            prev.par_chunks(8192)
                .chain(fresh.par_chunks(8192))
                .map(|chunk| {
                    if cancelled() {
                        return None;
                    }
                    block(Ids::List(chunk, n))
                })
                .collect()
        }
        None => (0..n.div_ceil(8192))
            .into_par_iter()
            .map(|c| {
                if cancelled() {
                    return None;
                }
                block(Ids::Range(c * 8192, ((c + 1) * 8192).min(n)))
            })
            .collect(),
    };
    let t1 = t0.elapsed();
    // the parts copied into one list in parallel, each to its own place
    let mut parts: Vec<Vec<Scored>> = parts.into_iter().collect::<Option<_>>()?;
    if split_long {
        // the long candidates (of those searched), one task each, merged back
        // in index order so that ties keep their input order
        let wanted: Vec<u32> = match subset {
            Some((prev, from)) => store
                .long()
                .iter()
                .copied()
                .filter(|&i| i as usize >= from || prev.binary_search(&i).is_ok())
                .collect(),
            None => store.long().to_vec(),
        };
        // (long ASCII lines were scored in their blocks)
        let wanted: Vec<u32> = if long_ascii_inline {
            wanted
                .into_iter()
                .filter(|&i| !store.is_ascii(i as usize))
                .collect()
        } else {
            wanted
        };
        // plain words without negations on an ASCII line: along the line
        // (a window past the kernel's limit goes to the matcher)
        let long_ascii = simd
            .as_ref()
            .filter(|(_, _, has_neg)| !has_neg)
            .zip(ascii_words.as_ref());
        let long_hits: Vec<Scored> = wanted
            .par_iter()
            .with_max_len(1)
            .filter_map(|&i| {
                let i = i as usize;
                match long_ascii {
                    Some(((_, path, _), aw)) if store.is_ascii(i) => WIN_BUFS.with(|b| {
                        let b = &mut *b.borrow_mut();
                        score_ascii_words(store, i, aw, *path, window_max, &o.tiebreak, b)
                            .unwrap_or_else(|()| score_full(i))
                    }),
                    _ => score_full(i),
                }
            })
            .collect();
        // each into the part that covers its index (the parts are in index
        // order), at its place: a few hundred inserts, not a sort of all hits
        let firsts: Vec<(usize, u32)> = parts
            .iter()
            .enumerate()
            .filter_map(|(k, p)| p.first().map(|h| (k, h.idx)))
            .collect();
        for h in long_hits {
            let k = match firsts.partition_point(|&(_, f)| f <= h.idx) {
                0 => firsts.first().map_or(0, |f| f.0),
                j => firsts[j - 1].0,
            };
            if parts.is_empty() {
                parts.push(Vec::new());
            }
            let at = parts[k].partition_point(|x| x.idx < h.idx);
            parts[k].insert(at, h);
        }
    }
    if cancelled() {
        return None;
    }
    let total: usize = parts.iter().map(Vec::len).sum();
    // sorted by counting straight from the parts into the final list (no
    // separate gathering, no copy), when the (score, key) range is small
    if o.sort && !o.tac && total >= 1 << 14 {
        let t2 = t0.elapsed();
        if let Some(hits) = counting_sort_parts(&parts, total, o, store) {
            if std::env::var_os("CORE_TIMES").is_some() {
                eprintln!(
                    "{query:?}: score {:.2} collect+sort {:.2} ms",
                    t1.as_secs_f64() * 1e3,
                    (t0.elapsed() - t2).as_secs_f64() * 1e3
                );
            }
            return Some(Hits::List(Arc::new(hits)));
        }
    }
    let mut hits: Vec<Scored> = Vec::with_capacity(total);
    {
        let spare = &mut hits.spare_capacity_mut()[..total];
        let mut slots = Vec::with_capacity(parts.len());
        let mut rest = spare;
        for p in &parts {
            let (head, tail) = rest.split_at_mut(p.len());
            slots.push(head);
            rest = tail;
        }
        slots
            .into_par_iter()
            .zip(parts.par_iter())
            .for_each(|(dst, src)| {
                for (d, s) in dst.iter_mut().zip(src) {
                    d.write(*s);
                }
            });
    }
    // SAFETY: every one of the `total` slots was written just above.
    unsafe { hits.set_len(total) };
    let t2 = t0.elapsed();
    sort(&mut hits, o);
    let hits: Vec<Hit> = hits.par_iter().map(Scored::hit).collect();
    if std::env::var_os("CORE_TIMES").is_some() {
        eprintln!(
            "{query:?}: score {:.2} collect {:.2} sort {:.2} ms",
            t1.as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            (t0.elapsed() - t2).as_secs_f64() * 1e3
        );
    }
    Some(Hits::List(Arc::new(hits)))
}

/// Sort by counting, when the (score, tiebreak key) pairs fall in a small
/// range (the usual case: scores and lengths are small integers). The parts
/// are in index order; each counts its pairs, the counts give every pair's
/// place, and each part writes its hits straight there — O(n), on every core,
/// one write per hit. Parts keep their order within a pair, so ties stay in
/// index order. None when the range is too wide (then a comparison sort).
fn counting_sort_parts(
    parts: &[Vec<Scored>],
    total: usize,
    o: &Options,
    store: &Store,
) -> Option<Vec<Hit>> {
    let with_key = match o.tiebreak.as_slice() {
        [] | [Tiebreak::Index] => false,
        [Tiebreak::Length] => true,
        _ => return None,
    };
    // the bucket key: for the same score, the length plus the score orders
    // like the length, and its range is narrow (the score mostly falls with
    // the length: it is about the bonuses minus the length), where the
    // length's own range is as wide as the lines differ
    let key = |h: &Scored| {
        if with_key {
            h.keys[0] as i64 + h.score as i64
        } else {
            0
        }
    };
    let (smin, smax, kmin, kmax) = parts
        .par_iter()
        .flat_map_iter(|p| p.iter())
        .fold(
            || (i32::MAX, i32::MIN, i64::MAX, i64::MIN),
            |(a, b, c, d), h| (a.min(h.score), b.max(h.score), c.min(key(h)), d.max(key(h))),
        )
        .reduce(
            || (i32::MAX, i32::MIN, i64::MAX, i64::MIN),
            |x, y| (x.0.min(y.0), x.1.max(y.1), x.2.min(y.2), x.3.max(y.3)),
        );
    // counting pays per bucket (per group): the buckets must be few next to
    // the results. A few outliers (a huge line: a very low score, a great
    // length) must not widen the range for all: past the caps they share
    // overflow buckets that are sorted by comparison afterwards — keys past
    // the cap in the last key bucket of their score, scores past the cap in
    // one bucket at the very end (all lower than the rest).
    // a group per 16 384 results, at most one per core; each group counts
    // into its own histogram, so the buckets times the groups must stay
    // within the results
    let groups = rayon::current_num_threads()
        .clamp(1, 64)
        .min(total / 16384)
        .max(1);
    let budget = (total / groups).clamp(4096, 1 << 16);
    // the key gets at most budget / 256 buckets (at least 64), so the scores
    // keep a window of 256: its overflow falls into many small buckets (one
    // per score, sorted in parallel), the score's into one at the end
    let krange = ((kmax - kmin + 1).min(u32::MAX as i64) as usize)
        .min((budget / 256).max(64))
        .min(budget / 2)
        .max(1);
    let srange_full = (smax as i64 - smin as i64 + 1) as usize;
    let srange = srange_full.min(budget / krange - 1).max(1);
    let score_tail = srange < srange_full;
    let buckets = srange * krange + score_tail as usize;
    let key_tail = krange < (kmax - kmin + 1).min(u32::MAX as i64) as usize;
    // higher score first, then the smaller key
    let bucket = |h: &Scored| {
        let s = (smax - h.score) as usize;
        if s >= srange {
            return srange * krange; // the low-score tail
        }
        s * krange + ((key(h) - kmin) as usize).min(krange - 1)
    };
    let per = parts.len().div_ceil(groups).max(1);
    let grouped: Vec<&[Vec<Scored>]> = parts.chunks(per).collect();
    let counts: Vec<Vec<u32>> = grouped
        .par_iter()
        .map(|g| {
            let mut h = vec![0u32; buckets];
            for x in g.iter().flatten() {
                h[bucket(x)] += 1;
            }
            h
        })
        .collect();
    // where each group starts writing each bucket: the bucket totals first,
    // their running sum, then each group's offset inside its bucket
    let mut base = vec![0u32; buckets];
    for c in &counts {
        for (t, &x) in base.iter_mut().zip(c) {
            *t += x;
        }
    }
    let mut at = 0u32;
    for t in base.iter_mut() {
        let n = *t;
        *t = at;
        at += n;
    }
    let base_start = base.clone();
    let mut starts = Vec::with_capacity(counts.len());
    for c in &counts {
        starts.push(base.clone());
        for (t, &x) in base.iter_mut().zip(c) {
            *t += x;
        }
    }
    let mut out: Vec<Hit> = Vec::with_capacity(total);
    let dst = SendPtr(out.as_mut_ptr());
    grouped
        .par_iter()
        .zip(starts.into_par_iter())
        .for_each(|(g, mut pos)| {
            let dst = &dst;
            for x in g.iter().flatten() {
                let b = bucket(x);
                // SAFETY: the counts give every hit its own slot in 0..total; each
                // group writes only the slots counted for it, so no writes overlap.
                unsafe { dst.0.add(pos[b] as usize).write(x.hit()) };
                pos[b] += 1;
            }
        });
    // SAFETY: all `total` slots were written above (the counts add up to total).
    unsafe { out.set_len(total) };
    // the overflow buckets, sorted by comparison (each keeps index order among
    // ties, the comparator ends with the index)
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    if key_tail {
        for s in 0..srange {
            let b = s * krange + krange - 1;
            let n = bucket_len(&base_start, b, total);
            if n > 1 {
                ranges.push((base_start[b] as usize, n));
            }
        }
    }
    if score_tail {
        let b = srange * krange;
        let n = bucket_len(&base_start, b, total);
        if n > 1 {
            ranges.push((base_start[b] as usize, n));
        }
    }
    if !ranges.is_empty() {
        let mut slices = Vec::with_capacity(ranges.len());
        let mut rest: &mut [Hit] = &mut out;
        let mut at = 0;
        for (start, n) in ranges {
            let (_, tail) = rest.split_at_mut(start - at);
            let (seg, tail) = tail.split_at_mut(n);
            slices.push(seg);
            rest = tail;
            at = start + n;
        }
        // the result has no keys: the length comes from the store again (the
        // only key the counting sort takes), once per hit, packed with the
        // score into one u64 (higher score first, then the shorter)
        let trimmed = store.trimmed_table();
        slices.into_par_iter().for_each(|seg| {
            let mut keyed: Vec<(u64, u32)> = seg
                .iter()
                .map(|h| {
                    let key = if with_key { trimmed[h.idx as usize] } else { 0 };
                    (
                        ((i32::MAX - h.score) as u32 as u64) << 32 | key as u64,
                        h.idx,
                    )
                })
                .collect();
            if keyed.len() > 1 << 14 {
                keyed.par_sort_unstable();
            } else {
                keyed.sort_unstable();
            }
            for (h, &(k, idx)) in seg.iter_mut().zip(&keyed) {
                *h = Hit {
                    idx,
                    score: i32::MAX - (k >> 32) as i32,
                };
            }
        });
    }
    Some(out)
}

/// The size of bucket `b` from the bucket starts.
fn bucket_len(starts: &[u32], b: usize, total: usize) -> usize {
    let end = starts.get(b + 1).map_or(total, |&e| e as usize);
    end - starts[b] as usize
}

/// The scratch of the along-the-line kernel, reused by each thread.
#[derive(Default)]
struct WinBufs {
    fold: Vec<i16>,
    bonus: Vec<i16>,
    rows: Vec<i16>,
}

thread_local! {
    static WIN_BUFS: std::cell::RefCell<WinBufs> = std::cell::RefCell::new(WinBufs::default());
}

/// One plain word, prepared once per search for the ASCII path.
struct AsciiWord {
    bytes: Vec<u8>,
    needle: Vec<i16>,
    cs: bool,
}

/// The byte classes the bonus rules look at.
const C_BOUND: u8 = 1;
const C_SLASH: u8 = 2;
const C_LOWER: u8 = 4;
const C_UPPER: u8 = 8;
const C_DIGIT: u8 = 16;

/// Per byte: its class bits, and its lower-case form.
fn byte_tables() -> &'static ([u8; 256], [u8; 256]) {
    static T: std::sync::OnceLock<([u8; 256], [u8; 256])> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut class = [0u8; 256];
        let mut lower = [0u8; 256];
        for b in 0..256usize {
            let c = b as u8;
            let bound = crate::simd::is_boundary(c);
            class[b] = (bound as u8 * C_BOUND)
                | ((c == b'/') as u8 * C_SLASH)
                | (c.is_ascii_lowercase() as u8 * C_LOWER)
                | (c.is_ascii_uppercase() as u8 * C_UPPER)
                | (c.is_ascii_digit() as u8 * C_DIGIT);
            lower[b] = c.to_ascii_lowercase();
        }
        (class, lower)
    })
}

/// The match bonus by (class of the previous byte, class of this byte), for
/// each scheme: the matcher's rule as a table (32 x 32 per scheme).
fn bonus_table(path: bool) -> &'static [i16; 1024] {
    static T: std::sync::OnceLock<[[i16; 1024]; 2]> = std::sync::OnceLock::new();
    &T.get_or_init(|| {
        let rule = |path: bool, p: u8, c: u8| -> i16 {
            use crate::matcher::{BONUS_BOUNDARY, BONUS_CAMEL};
            (if p & C_BOUND != 0 {
                if path && p & C_SLASH != 0 {
                    BONUS_BOUNDARY + 2
                } else {
                    BONUS_BOUNDARY
                }
            } else if p & C_LOWER != 0 && c & C_UPPER != 0 {
                BONUS_CAMEL
            } else if p & C_DIGIT == 0 && c & C_DIGIT != 0 {
                BONUS_CAMEL / 2
            } else {
                0
            }) as i16
        };
        let mut t = [[0i16; 1024]; 2];
        for (k, tab) in t.iter_mut().enumerate() {
            for p in 0..32u8 {
                for c in 0..32u8 {
                    tab[(p as usize) << 5 | c as usize] = rule(k == 1, p, c);
                }
            }
        }
        t
    })[path as usize]
}

/// The candidates of one block: a range of indices, or a list (a narrowed
/// search); the loops over them are monomorphic (no call per candidate).
#[derive(Clone, Copy)]
enum Ids<'a> {
    Range(usize, usize),
    /// indices below the count only (an older list may name more)
    List(&'a [u32], usize),
}

impl Ids<'_> {
    #[inline(always)]
    fn each(self, mut f: impl FnMut(usize)) {
        match self {
            Ids::Range(a, b) => (a..b).for_each(f),
            Ids::List(l, n) => {
                for &i in l {
                    if (i as usize) < n {
                        f(i as usize);
                    }
                }
            }
        }
    }
}

/// A literal query (`^x`, `x$`, `'x`), prepared once.
struct LitQuery<'a> {
    kind: crate::matcher::Literal,
    /// the needle as UTF-8 bytes, folded unless case-sensitive
    bytes: Vec<u8>,
    /// each character with its other case (simd::utf8_char)
    chars: Vec<crate::simd::Utf8Char>,
    /// its length in characters
    n: usize,
    cs: bool,
    ascii: bool,
    path: bool,
    need: u32,
    /// the needle starts or ends with white space (then a match alone does
    /// not prove that the span past the trimmed edges holds it)
    edge_space: bool,
    finder: memchr::memmem::Finder<'a>,
    tiebreak: &'a [Tiebreak],
    /// the byte classes and the bonus table (looked up once)
    class: &'static [u8; 256],
    bonus: &'static [i16; 1024],
}

impl<'a> LitQuery<'a> {
    fn new(
        kind: crate::matcher::Literal,
        text: &[char],
        cs: bool,
        path: bool,
        need: u32,
        tiebreak: &'a [Tiebreak],
    ) -> Option<LitQuery<'a>> {
        let chars: Option<Vec<_>> = text
            .iter()
            .map(|&c| crate::simd::utf8_char(c, cs))
            .collect();
        let bytes: Vec<u8> = text.iter().collect::<String>().into_bytes();
        let finder = memchr::memmem::Finder::new(&bytes).into_owned();
        // ^x and x$: the bytes at the edge decide as fast as the signature
        // would, and almost always the same way (a branch the processor
        // predicts); the signature's pass or fail is a coin toss per line
        let need = if kind == crate::matcher::Literal::Exact {
            need
        } else {
            0
        };
        Some(LitQuery {
            kind,
            chars: chars?,
            n: text.len(),
            cs,
            ascii: text.iter().all(char::is_ascii),
            path,
            need,
            edge_space: text.first().is_some_and(|c| c.is_whitespace())
                || text.last().is_some_and(|c| c.is_whitespace()),
            finder,
            bytes,
            tiebreak,
            class: &byte_tables().0,
            bonus: bonus_table(path),
        })
    }

    /// Whether the needle is at byte `p` of `t` (either case unless
    /// case-sensitive).
    #[inline]
    fn at(&self, t: &[u8], p: usize) -> bool {
        if self.ascii {
            // ASCII: the bytes (the needle is folded unless case-sensitive)
            let Some(seg) = t.get(p..p + self.bytes.len()) else {
                return false;
            };
            return if self.cs {
                seg == self.bytes.as_slice()
            } else {
                seg.iter()
                    .zip(&self.bytes)
                    .all(|(&a, &b)| a.to_ascii_lowercase() == b)
            };
        }
        let mut j = p;
        for ch in &self.chars {
            if ch[1] == 0 {
                match t.get(j) {
                    Some(&b) if b == ch[0] || b == ch[2] => j += 1,
                    _ => return false,
                }
            } else {
                match t.get(j..j + 2) {
                    Some(&[a, b]) if (a == ch[0] && b == ch[1]) || (a == ch[2] && b == ch[3]) => {
                        j += 2
                    }
                    _ => return false,
                }
            }
        }
        true
    }
}

/// The score of the literal at byte `p` of line `t` (ASCII or simple UTF-8):
/// the leading and trailing gaps together are -(N - n) whatever the place,
/// so only the bonuses of the n characters vary.
#[inline]
fn literal_score(t: &[u8], ascii: bool, p: usize, q: &LitQuery) -> i32 {
    let (class, bonus_of) = (q.class, q.bonus);
    let mut sum = 0;
    if ascii {
        for j in p..p + q.n {
            sum += if j == 0 {
                crate::matcher::BONUS_FIRST
            } else {
                bonus_of[(class[t[j - 1] as usize] as usize) << 5 | class[t[j] as usize] as usize]
                    as i32
            };
        }
        return -((t.len() - q.n) as i32) + crate::matcher::BONUS_CONSEC * (q.n as i32 - 1) + sum;
    }
    // SAFETY: a stored line is valid UTF-8, and p is a character boundary
    // (where the needle starts).
    let s = unsafe { std::str::from_utf8_unchecked(t) };
    let total = s.chars().count();
    let mut prev = s[..p].chars().next_back();
    for c in s[p..].chars().take(q.n) {
        sum += match prev {
            None => crate::matcher::BONUS_FIRST,
            Some(pc) => crate::matcher::char_bonus(pc, c, q.path),
        };
        prev = Some(c);
    }
    -((total - q.n) as i32) + crate::matcher::BONUS_CONSEC * (q.n as i32 - 1) + sum
}

/// Fold a simple line (ASCII letters, basic Cyrillic) to lower case; the
/// byte length does not change.
fn fold_simple(t: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(t);
    let mut j = 0;
    while j < out.len() {
        let b = out[j];
        if b.is_ascii_uppercase() {
            out[j] = b + 32;
        } else if b == 0xD0 && j + 1 < out.len() {
            let c = out[j + 1];
            match c {
                0x90..=0x9F => out[j + 1] = c + 0x20,
                0xA0..=0xAF => {
                    out[j] = 0xD1;
                    out[j + 1] = c - 0x20;
                }
                0x81 => {
                    out[j] = 0xD1;
                    out[j + 1] = 0x91;
                }
                _ => {}
            }
            j += 1;
        }
        j += 1;
    }
}

thread_local! {
    static FOLD_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One block of candidates for a literal query: ASCII and simple UTF-8 lines
/// by their bytes (the score in closed form), the rest by `full`.
fn literal_block(
    store: &Store,
    ids: Ids,
    q: &LitQuery,
    skip_long: bool,
    full: &dyn Fn(usize) -> Option<Scored>,
) -> Vec<Scored> {
    match ids {
        Ids::Range(a, b) => literal_run(store, a..b, q, skip_long, full),
        Ids::List(l, n) => literal_run(
            store,
            l.iter().map(|&i| i as usize).filter(|&i| i < n),
            q,
            skip_long,
            full,
        ),
    }
}

/// The loop of `literal_block`, monomorphic in the candidates.
#[inline]
fn literal_run(
    store: &Store,
    ids: impl Iterator<Item = usize>,
    q: &LitQuery,
    skip_long: bool,
    full: &dyn Fn(usize) -> Option<Scored>,
) -> Vec<Scored> {
    use crate::matcher::Literal;
    let mut out = Vec::new();
    let space = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let nb = q.bytes.len();
    let (data, ends, sigs, flags) = store.tables();
    for i in ids {
        if q.need & !sigs[i] != 0 {
            continue;
        }
        let start = if i == 0 { 0 } else { ends[i - 1] as usize };
        let t = &data[start..ends[i] as usize];
        if skip_long && t.len() > crate::store::LONG {
            continue; // a task of its own, after the blocks
        }
        let f = flags[i];
        let ascii = f & 1 != 0;
        if !ascii && f & 4 == 0 {
            if let Some(h) = full(i) {
                out.push(h);
            }
            continue;
        }
        if ascii && !q.ascii {
            continue; // a non-ASCII needle character never occurs here
        }
        let p = match q.kind {
            Literal::Prefix => {
                // the first non-space byte; the span after it holds the needle
                // when the needle ends with a non-space byte that matched
                let start = if t.first().is_some_and(|&b| !space(b)) {
                    0
                } else {
                    match t.iter().position(|&b| !space(b)) {
                        Some(s) => s,
                        None => continue,
                    }
                };
                if !q.at(t, start) {
                    continue;
                }
                if q.edge_space {
                    let stop = t.iter().rposition(|&b| !space(b)).map_or(start, |e| e + 1);
                    if stop - start < nb {
                        continue;
                    }
                }
                start
            }
            Literal::Suffix => {
                let stop = if t.last().is_some_and(|&b| !space(b)) {
                    t.len()
                } else {
                    match t.iter().rposition(|&b| !space(b)) {
                        Some(e) => e + 1,
                        None => continue,
                    }
                };
                if stop < nb || !q.at(t, stop - nb) {
                    continue;
                }
                if q.edge_space {
                    let start = t.iter().position(|&b| !space(b)).unwrap_or(t.len());
                    if stop - start < nb {
                        continue;
                    }
                }
                stop - nb
            }
            Literal::Exact => {
                // the best of every occurrence (in the folded line unless
                // case-sensitive: folding keeps the byte offsets)
                let best = FOLD_BUF.with(|buf| {
                    let buf = &mut *buf.borrow_mut();
                    let hay: &[u8] = if q.cs || !store.has_upper(i) {
                        t
                    } else {
                        fold_simple(t, buf);
                        buf
                    };
                    q.finder
                        .find_iter(hay)
                        .map(|p| literal_score(t, ascii, p, q))
                        .max()
                });
                let Some(score) = best else { continue };
                out.push(Scored {
                    idx: i as u32,
                    score,
                    keys: literal_keys(store, i, q.tiebreak),
                });
                continue;
            }
        };
        out.push(Scored {
            idx: i as u32,
            score: literal_score(t, ascii, p, q),
            keys: literal_keys(store, i, q.tiebreak),
        });
    }
    out
}

/// The tiebreak keys of a hit on the literal path (no key needs the match's
/// place: those searches keep the positions and do not come here).
#[inline]
fn literal_keys(store: &Store, i: usize, tiebreak: &[Tiebreak]) -> [u32; 3] {
    let mut keys = [0u32; 3];
    for (k, t) in tiebreak.iter().take(3).enumerate() {
        keys[k] = if *t == Tiebreak::Length {
            store.trimmed_len(i)
        } else {
            tiebreak_key(*t, store.text(i), 0, 0, i as u32)
        };
    }
    keys
}

/// A block of candidates (at most this many).
const BLOCK: usize = 8192;
/// Below this many survivors of the in-group filter, a group's lines wait
/// for a full group of their own.
const GROUP_DENSE: usize = 12;
/// No score (in the per-candidate scores of a block).
const NONE: i32 = i32::MIN;

/// What the group path needs of the query.
struct FastQuery<'a> {
    /// the signature bits every candidate must have (0: none)
    need: u32,
    /// the words for ASCII lines (None: a word has a non-ASCII character,
    /// so no ASCII line matches)
    ascii: Option<&'a [AsciiWord]>,
    /// the words for the simple UTF-8 lines (each character ASCII or basic
    /// Cyrillic)
    utf8: &'a [Utf8Word],
    path: bool,
    /// the pattern, when it has negations (checked on the survivors)
    neg: Option<&'a Pattern>,
    tiebreak: &'a [Tiebreak],
}

/// One plain word for the UTF-8 group kernel.
struct Utf8Word {
    chars: Vec<crate::simd::Utf8Char>,
}

/// The width classes of a block: four of 64 bytes for ASCII lines, four for
/// simple UTF-8 lines.
const CLASSES: usize = 8;

/// The scratch of the group path, reused by each thread.
struct FastBufs {
    cand: Vec<u32>,
    score: Vec<i32>,
    lists: [Vec<u16>; CLASSES],
    surv: Vec<u16>,
    others: Vec<u32>,
    #[cfg(target_arch = "x86_64")]
    cols: Box<[crate::simd::V; crate::simd::GROUP_COLS]>,
    #[cfg(target_arch = "x86_64")]
    fold: Box<[crate::simd::V; crate::simd::GROUP_COLS]>,
    #[cfg(target_arch = "x86_64")]
    bonus: Box<[crate::simd::V; crate::simd::GROUP_COLS]>,
    #[cfg(target_arch = "x86_64")]
    cont: Box<[crate::simd::V; crate::simd::GROUP_COLS]>,
}

#[cfg(target_arch = "x86_64")]
impl Default for FastBufs {
    fn default() -> FastBufs {
        // SAFETY: an all-zero __m256i is a valid value.
        let zero =
            || Box::new(unsafe { std::mem::zeroed::<[crate::simd::V; crate::simd::GROUP_COLS]>() });
        FastBufs {
            cand: vec![0; BLOCK],
            score: vec![NONE; BLOCK],
            lists: std::array::from_fn(|_| vec![0; BLOCK]),
            surv: vec![0; BLOCK],
            others: Vec::new(),
            cols: zero(),
            fold: zero(),
            bonus: zero(),
            cont: zero(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    static FAST_BUFS: std::cell::RefCell<Option<FastBufs>> = const { std::cell::RefCell::new(None) };
}

/// One block of candidates (at most BLOCK, in index order) for plain words,
/// in passes with simple loops: the signature (survivors compacted without a
/// branch); the split into the lines the group kernels take — ASCII up to
/// 256 bytes, simple UTF-8 up to 255 — by width class (64 bytes each), and
/// the others; the kernels 32 lines at a time, each result written at its
/// candidate's position (so the order stays the index order without
/// sorting); the others through `rest` (the per-line paths); the two merged
/// by index. None when cancelled.
///
/// # Safety
/// AVX2 ([`crate::simd::available`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn block_fast(
    store: &Store,
    ids: Ids,
    q: &FastQuery,
    cancelled: &dyn Fn() -> bool,
    rest: &mut dyn FnMut(&mut dyn Iterator<Item = usize>) -> Vec<Scored>,
) -> Option<Vec<Scored>> {
    use crate::simd::{GROUP, GROUP_COLS};
    let mut bufs = FAST_BUFS
        .with(|b| b.borrow_mut().take())
        .unwrap_or_default();
    let out = block_fast_in(store, ids, q, cancelled, rest, &mut bufs);
    FAST_BUFS.with(|b| *b.borrow_mut() = Some(bufs));
    return out;

    #[target_feature(enable = "avx2")]
    unsafe fn block_fast_in(
        store: &Store,
        ids: Ids,
        q: &FastQuery,
        cancelled: &dyn Fn() -> bool,
        rest: &mut dyn FnMut(&mut dyn Iterator<Item = usize>) -> Vec<Scored>,
        b: &mut FastBufs,
    ) -> Option<Vec<Scored>> {
        let (data, ends, sigs, flags) = store.tables();
        // 1. the signature: every id written, the count moves only for a survivor
        let mut k = 0;
        let cand_buf = &mut b.cand;
        ids.each(|i| {
            cand_buf[k] = i as u32;
            k += (q.need & !sigs[i] == 0) as usize;
        });
        let cand = &b.cand[..k];
        // 2. the width classes (positions in cand) and the others
        let mut fill = [0usize; CLASSES];
        b.others.clear();
        let safe_end = data.len().saturating_sub(GROUP_COLS);
        // UTF-8 lines with a negation: the per-line paths (decided by
        // may_match there)
        let take_utf8 = q.neg.is_none();
        for (p, &i) in cand.iter().enumerate() {
            let i = i as usize;
            let start = if i == 0 { 0 } else { ends[i - 1] as usize };
            let len = ends[i] as usize - start;
            let f = flags[i];
            let near_end = start > safe_end;
            if f & 1 != 0 && len != 0 && len <= GROUP_COLS && !near_end {
                if q.ascii.is_none() {
                    b.score[p] = NONE; // a non-ASCII needle character: no match
                    continue;
                }
                let class = (len - 1) / 64;
                b.lists[class][fill[class]] = p as u16;
                fill[class] += 1;
            } else if f & 4 != 0 && take_utf8 && len < GROUP_COLS && !near_end {
                let class = 4 + (len - 1) / 64;
                b.lists[class][fill[class]] = p as u16;
                fill[class] += 1;
            } else {
                b.score[p] = NONE; // scored on the per-line paths, merged below
                b.others.push(i as u32);
            }
        }
        // 3. the groups
        let need_fold = q.ascii.is_some_and(|w| w.iter().any(|w| !w.cs));
        for class in 0..CLASSES {
            if fill[class] == 0 {
                continue;
            }
            if cancelled() {
                return None;
            }
            let utf8 = class >= 4;
            // the in-group subsequence filter pays only for words of two or
            // more characters on ASCII (one character: the signature is
            // exact there); on UTF-8 lines the signature does not filter
            let filter = if utf8 {
                true
            } else {
                q.ascii.is_some_and(|w| w.iter().any(|w| w.bytes.len() > 1))
            };
            for pass in 0..2 {
                if pass == 1 && !filter {
                    break;
                }
                let mut ns = 0;
                let list = std::mem::take(&mut b.lists[class]);
                for group in list[..fill[class]].chunks(GROUP) {
                    let mut ptrs = [data.as_ptr(); GROUP];
                    let mut lens = [0u16; GROUP];
                    let mut width = 0;
                    for (l, &p) in group.iter().enumerate() {
                        let i = cand[p as usize] as usize;
                        let start = if i == 0 { 0 } else { ends[i - 1] as usize };
                        let len = ends[i] as usize - start;
                        ptrs[l] = data.as_ptr().add(start);
                        lens[l] = len as u16;
                        width = width.max(len);
                    }
                    crate::simd::group_transpose(&ptrs, &lens, width, &mut b.cols);
                    if utf8 {
                        // the kernels read the byte after the last column
                        b.cols[width] = std::mem::zeroed();
                    }
                    if filter && pass == 0 {
                        let mut mask = u32::MAX;
                        if utf8 {
                            for w in q.utf8 {
                                mask &= crate::simd::group_subseq_utf8(&b.cols, width, &w.chars);
                            }
                        } else {
                            for w in q.ascii.unwrap_or_default() {
                                mask &= crate::simd::group_subseq(&b.cols, width, &w.bytes, !w.cs);
                            }
                        }
                        if group.len() < GROUP {
                            mask &= (1u32 << group.len()) - 1;
                        }
                        // no survivor: every score stays NONE
                        if mask == 0 {
                            for &p in group {
                                b.score[p as usize] = NONE;
                            }
                            continue;
                        }
                        // few survivors: they wait for a full group of their own
                        if (mask.count_ones() as usize) < GROUP_DENSE {
                            for (l, &p) in group.iter().enumerate() {
                                b.score[p as usize] = NONE;
                                b.surv[ns] = p;
                                ns += (mask >> l & 1) as usize;
                            }
                            continue;
                        }
                    }
                    let mut sums = [0i32; GROUP];
                    let mut ok = [true; GROUP];
                    if utf8 {
                        let mut chars = [0u8; GROUP];
                        crate::simd::group_classify_utf8(
                            &b.cols,
                            width,
                            q.path,
                            &mut b.bonus,
                            &mut b.cont,
                            &mut chars,
                        );
                        for w in q.utf8 {
                            let mut best = [0i32; GROUP];
                            crate::simd::group_best_utf8(
                                &b.cols, &b.bonus, &b.cont, width, &w.chars, &mut best, &mut ok,
                            );
                            for l in 0..GROUP {
                                sums[l] += best[l] - (chars[l] as i32 - 1);
                            }
                        }
                    } else {
                        crate::simd::group_classify(
                            &b.cols,
                            width,
                            q.path,
                            need_fold.then_some(&mut *b.fold),
                            &mut b.bonus,
                        );
                        for w in q.ascii.unwrap_or_default() {
                            let mut best = [0i32; GROUP];
                            let keys = if w.cs { &*b.cols } else { &*b.fold };
                            crate::simd::group_best(
                                keys, &b.bonus, width, &w.bytes, &mut best, &mut ok,
                            );
                            for l in 0..GROUP {
                                sums[l] += best[l] - (lens[l] as i32 - 1);
                            }
                        }
                    }
                    for (l, &p) in group.iter().enumerate() {
                        b.score[p as usize] = if ok[l] { sums[l] } else { NONE };
                    }
                }
                b.lists[class] = list;
                if filter && pass == 0 {
                    // the survivors replace the class's list (still in index order)
                    let (lists, surv) = (&mut b.lists, &b.surv);
                    lists[class][..ns].copy_from_slice(&surv[..ns]);
                    fill[class] = ns;
                }
            }
        }
        // 4. the others, on the per-line paths (in index order)
        let others = if b.others.is_empty() {
            Vec::new()
        } else {
            rest(&mut b.others.iter().map(|&i| i as usize))
        };
        // 5. the hits in index order, the others merged in
        let mut hits: Vec<Scored> = Vec::with_capacity(k + others.len());
        let by_length = match q.tiebreak {
            [] => Some(false),
            [Tiebreak::Length] => Some(true),
            _ => None,
        };
        if let (Some(with_len), None, true) = (by_length, q.neg, others.is_empty()) {
            // the common case: every hit written, the count moves only for a
            // score (no branch per candidate)
            let trimmed = store.trimmed_table();
            let dst = hits.as_mut_ptr();
            let mut h = 0;
            for (p, &i) in cand.iter().enumerate() {
                let s = b.score[p];
                let key = if with_len { trimmed[i as usize] } else { 0 };
                // SAFETY: h < k <= capacity (at most one hit per candidate).
                dst.add(h).write(Scored {
                    idx: i,
                    score: s,
                    keys: [key, 0, 0],
                });
                h += (s != NONE) as usize;
            }
            // SAFETY: the first h slots were written above.
            hits.set_len(h);
            return Some(hits);
        }
        let mut others = others.into_iter().peekable();
        for (p, &i) in cand.iter().enumerate() {
            // every position was written: a score, or NONE (no match, or a
            // line of the others)
            let s = b.score[p];
            if s == NONE {
                continue;
            }
            if let Some(pat) = q.neg {
                let start = if i == 0 {
                    0
                } else {
                    ends[i as usize - 1] as usize
                };
                let len = ends[i as usize] as usize - start;
                if !pat.may_match_ascii(&data[start..start + len]) {
                    continue;
                }
            }
            while others.peek().is_some_and(|h| h.idx < i) {
                hits.push(others.next().unwrap());
            }
            let mut keys = [0u32; 3];
            for (kk, t) in q.tiebreak.iter().take(3).enumerate() {
                keys[kk] = if *t == Tiebreak::Length {
                    store.trimmed_len(i as usize)
                } else {
                    tiebreak_key(*t, store.text(i as usize), 0, 0, i)
                };
            }
            hits.push(Scored {
                idx: i,
                score: s,
                keys,
            });
        }
        hits.extend(others);
        Some(hits)
    }
}

/// The widest window the i16 kernel takes (the values drift by one per
/// column over a gap and must stay far from the i16 limits); a wider one, or
/// one past the configured exact window, goes to the matcher.
const KERNEL_MAX: usize = 4096;

/// Score ASCII candidate `i` for plain words: each word over its own window
/// with the along-the-line kernel, the sum. Err when a window is too wide for
/// the kernel (the caller scores the line on the scalar path).
fn score_ascii_words(
    store: &Store,
    i: usize,
    words: &[AsciiWord],
    path: bool,
    window_max: usize,
    tiebreak: &[Tiebreak],
    b: &mut WinBufs,
) -> Result<Option<Scored>, ()> {
    let hay = store.text(i).as_bytes();
    let n = hay.len() as i32;
    let (class, lower) = byte_tables();
    let bonus_of = bonus_table(path);
    let mut total = 0i32;
    for word in words {
        if word.bytes.len() == 1 {
            // one character: the largest bonus among its places (row 0 of the
            // recursion is the whole answer), one pass over the line
            let want = word.bytes[0];
            let other = if word.cs || !want.is_ascii_lowercase() {
                want
            } else {
                want.to_ascii_uppercase()
            };
            let mut best = i32::MIN;
            let top = if path {
                crate::matcher::BONUS_BOUNDARY + 2
            } else {
                crate::matcher::BONUS_BOUNDARY
            };
            for j in memchr::memchr2_iter(want, other, hay) {
                let b = if j == 0 {
                    crate::matcher::BONUS_FIRST
                } else {
                    bonus_of[(class[hay[j - 1] as usize] as usize) << 5
                        | class[hay[j] as usize] as usize] as i32
                };
                best = best.max(b);
                // nothing scores higher: the first column, or a boundary
                // (after a slash in the path scheme)
                if best >= top {
                    break;
                }
            }
            if best == i32::MIN {
                return Ok(None);
            }
            total += best - (n - 1);
            continue;
        }
        let Some((first, last)) = crate::matcher::window_ascii(hay, &word.bytes, word.cs) else {
            return Ok(None);
        };
        let w = last - first + 1;
        if w > window_max {
            return Err(());
        }
        let padded = w.div_ceil(16) * 16;
        if b.fold.len() < padded {
            b.fold.resize(padded, 0);
            b.bonus.resize(padded, 0);
        }
        let (fold, bonus) = (&mut b.fold[..padded], &mut b.bonus[..padded]);
        let mut prev = if first == 0 {
            0
        } else {
            class[hay[first - 1] as usize]
        };
        for (k, &c) in hay[first..=last].iter().enumerate() {
            let cl = class[c as usize];
            fold[k] = if word.cs { c } else { lower[c as usize] } as i16;
            bonus[k] = bonus_of[(prev as usize) << 5 | cl as usize];
            prev = cl;
        }
        if first == 0 {
            bonus[0] = crate::matcher::BONUS_FIRST as i16;
        }
        fold[w..].fill(0);
        bonus[w..].fill(0);
        let need = 4 * (padded + 16);
        if b.rows.len() < need {
            b.rows.resize(need, 0);
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2 checked by simd::available() before this path is taken;
        // fold / bonus are padded to 16, rows hold 4 * (padded + 16); the word
        // is 1..=16 ASCII characters, none 0 (plain_words).
        let best = unsafe {
            crate::simd::score_window(
                &b.fold[..padded],
                &b.bonus[..padded],
                &word.needle,
                &mut b.rows,
            )
        };
        #[cfg(not(target_arch = "x86_64"))]
        let best = crate::matcher::rebased_window(&b.fold[..w], &b.bonus[..w], &word.needle);
        if best < -10000 {
            return Ok(None);
        }
        total += best - (n - 1);
    }
    let mut keys = [0u32; 3];
    for (k, t) in tiebreak.iter().take(3).enumerate() {
        keys[k] = if *t == Tiebreak::Length {
            store.trimmed_len(i)
        } else {
            tiebreak_key(*t, store.text(i), 0, 0, i as u32)
        };
    }
    Ok(Some(Scored {
        idx: i as u32,
        score: total,
        keys,
    }))
}

/// The scratch of one block of the SIMD path, reused by each thread.
#[derive(Default)]
struct BlockBufs {
    slots: Vec<Option<Scored>>,
    /// (rounded length, length, slot, index) of the candidates for the tiles
    cands: Vec<(u32, u32, u32, u32)>,
    raw: Vec<u16>,
    fold: Vec<u16>,
    bonus: Vec<i16>,
}

thread_local! {
    static BLOCK_BUFS: std::cell::RefCell<BlockBufs> = std::cell::RefCell::new(BlockBufs::default());
}

struct SendPtr(*mut Hit);
// SAFETY: used only for the disjoint writes in counting_sort.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Score order: score (higher first), the tiebreak keys, then the index
/// (lower first, or higher with `tac`). Without sorting: input order.
fn sort(hits: &mut [Scored], o: &Options) {
    let tac = o.tac;
    if !o.sort {
        // the parts arrive in index order already, but a narrowed search may not
        hits.par_sort_unstable_by_key(|h| if tac { u32::MAX - h.idx } else { h.idx });
    } else if o.tiebreak.len() <= 1 {
        // the usual case: the whole order in one u64 (score over the key), then the index
        hits.par_sort_unstable_by_key(|h| {
            let score = u32::MAX - (h.score as i64 - i32::MIN as i64) as u32;
            (
                ((score as u64) << 32) | h.keys[0] as u64,
                if tac { u32::MAX - h.idx } else { h.idx },
            )
        });
    } else {
        hits.par_sort_unstable_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then(a.keys.cmp(&b.keys))
                .then(if tac {
                    b.idx.cmp(&a.idx)
                } else {
                    a.idx.cmp(&b.idx)
                })
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matcher::{CaseMode, Scheme};

    /// Every line scored one by one, the keys from the match, sorted by the
    /// comparator: what search must return.
    fn reference(s: &Store, q: &str, o: &Options) -> Vec<Hit> {
        let pat = Pattern::parse(q, &o.cfg);
        let mut all: Vec<Scored> = (0..s.len())
            .filter_map(|i| {
                let sc = pat.score(s.text(i))?;
                let mut keys = [0u32; 3];
                for (k, t) in o.tiebreak.iter().take(3).enumerate() {
                    keys[k] = tiebreak_key(*t, s.text(i), sc.begin, sc.end, i as u32);
                }
                Some(Scored {
                    idx: i as u32,
                    score: sc.score,
                    keys,
                })
            })
            .collect();
        sort(&mut all, o);
        all.iter().map(Scored::hit).collect()
    }

    fn opts(scope: Scope) -> Options {
        Options {
            cfg: MatchConfig {
                case: CaseMode::Smart,
                extended: true,
                exact: false,
                normalize: true,
                scheme: Scheme::Default,
                typos: 0,
                exact_window: crate::matcher::EXACT_WINDOW,
            },
            sort: true,
            tac: false,
            tiebreak: vec![Tiebreak::Length],
            scope,
        }
    }

    fn store(lines: &[String]) -> Store {
        let mut s = Store::new();
        for l in lines {
            s.push(l.as_bytes());
        }
        s
    }

    fn order(h: &Hits) -> Vec<u32> {
        (0..h.len()).map(|i| h.get(i).unwrap()).collect()
    }

    #[test]
    fn grep_scopes() {
        let dir = std::env::temp_dir().join(format!("lvf-core-scope-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("health.lua");
        let b = dir.join("other.lua");
        std::fs::write(&a, "local h = vim.health\nh.start('x')\n").unwrap();
        std::fs::write(&b, "-- mentions health here\nreturn 1\n").unwrap();
        let s = store(&[
            format!("{}:2:h.start('x')", a.display()),
            format!("{}:2:return 1", b.display()),
            format!("{}:1:-- mentions health here", b.display()),
        ]);
        let ids = |scope| {
            let mut v = order(&search(&s, "health", &opts(scope), None, None).unwrap());
            v.sort();
            v
        };
        assert_eq!(ids(Scope::Line), vec![2]);
        assert_eq!(ids(Scope::Path), vec![0]);
        assert_eq!(ids(Scope::File), vec![0, 1, 2]);
        assert_eq!(
            search(&s, "hrt", &opts(Scope::File), None, None)
                .unwrap()
                .len(),
            0
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn corpus() -> Store {
        let words = [
            "lvim",
            "fuzzy",
            "picker",
            "src",
            "main",
            "Config",
            "théme",
            "плъгин",
            "test",
            "a,b",
            "x|y",
        ];
        store(
            &(0..4000u32)
                .map(|i| {
                    let w = |k: u32| words[((i * 7 + k * 13) % words.len() as u32) as usize];
                    format!("{}/{}_{}.{}", w(1), w(2), i, w(3))
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn signature_never_drops_a_match() {
        let s = corpus();
        for q in [
            "lvim", "fz", "cfg", "théme", "theme", "плъг", "src ^lv", "a,b", "'x|y", "zzz", "lvmi",
        ] {
            for typos in [0u8, 1, 2] {
                let mut o = opts(Scope::All);
                o.cfg.typos = typos;
                let got = order(&search(&s, q, &o, None, None).unwrap());
                // the same without the signature: every candidate scored
                let pat = Pattern::parse(q, &o.cfg);
                let want = (0..s.len())
                    .filter(|&i| pat.score(s.text(i)).is_some())
                    .count();
                assert_eq!(got.len(), want, "query {q:?}, typos {typos}");
            }
        }
    }

    #[test]
    fn narrowing_gives_the_full_result() {
        let s = corpus();
        let o = opts(Scope::All);
        let chains: [&[&str]; 5] = [
            &["l", "lv", "lvi", "lvim", "lvim f", "lvim fu"],
            &["s", "sr", "src", "src ^"],
            &["th", "thé", "théme"],
            &["п", "пл", "плъг"],
            &["'te", "'tes", "'test"],
        ];
        for chain in chains {
            let mut prev: Option<(String, Vec<u32>)> = None;
            for q in chain {
                let full = search(&s, q, &o, None, None).unwrap();
                let subset = prev
                    .as_ref()
                    .filter(|(p, _)| narrows(p, q))
                    .map(|(_, idx)| (&idx[..], s.len()));
                let inc = search(&s, q, &o, None, subset).unwrap();
                assert_eq!(order(&full), order(&inc), "query {q:?}");
                prev = full.indices().map(|i| (q.to_string(), i));
            }
        }
        assert!(!narrows("!te", "!tes"));
        assert!(!narrows("a", "a | b"));
        assert!(!narrows("", "a"));
        assert!(narrows("lv", "lvim"));
    }

    #[test]
    fn packed_sort_orders_like_the_comparator() {
        let s = store(
            &(0..3000u32)
                .map(|i| format!("{}a{}b/{}", "x".repeat((i % 7) as usize), i % 13, i % 5))
                .collect::<Vec<_>>(),
        );
        for tac in [false, true] {
            for tb in [vec![Tiebreak::Length], vec![Tiebreak::Index], vec![]] {
                let mut o = opts(Scope::All);
                o.tac = tac;
                o.tiebreak = tb;
                let Hits::List(packed) = search(&s, "ab", &o, None, None).unwrap() else {
                    panic!()
                };
                let reference = reference(&s, "ab", &o);
                if let Some(k) = (0..packed.len()).find(|&k| packed[k] != reference[k]) {
                    panic!(
                        "tac {tac} tb {:?}: first difference at {k}: {:?} vs {:?}",
                        o.tiebreak,
                        &packed[k.saturating_sub(1)..k + 2],
                        &reference[k.saturating_sub(1)..k + 2]
                    );
                }
                assert!(!packed.is_empty());
            }
        }
    }

    #[test]
    fn simd_search_equals_scoring_one_by_one() {
        // ASCII and non-ASCII, many lengths: the plain-word path (SIMD where
        // available) must give exactly the hits of scoring every candidate
        let s = store(
            &(0..20_000u32)
                .map(|i| match i % 5 {
                    0 => format!("src/lvim_fuzzy/{i}.rs"),
                    1 => format!("Lua/LvimPicker{i}"),
                    2 => format!("plain text {i} lvim"),
                    3 if i % 2 == 0 => format!("плъгин lvim {i} théme"),
                    3 => format!("Плъгин_ТЕМА/{i}"),
                    _ => format!(
                        "{}l{}v{}",
                        "x".repeat((i % 40) as usize),
                        i % 7,
                        "y".repeat((i % 3) as usize)
                    ),
                })
                .collect::<Vec<_>>(),
        );
        for q in [
            "lv",
            "lvim",
            "LP",
            "src",
            "l7v",
            "fz",
            "a",
            "zzz",
            "src lv",
            "lv !rs",
            "lua !test lv",
            "lv rs !x",
            "пл",
            "плъг",
            "Плъ",
            "тема",
            "theme",
            "théme",
            "lv пл",
        ] {
            for tb in [vec![Tiebreak::Length], vec![]] {
                let mut o = opts(Scope::All);
                o.tiebreak = tb;
                let Hits::List(got) = search(&s, q, &o, None, None).unwrap() else {
                    panic!()
                };
                let pat = Pattern::parse(q, &o.cfg);
                let mut want: Vec<Scored> = (0..s.len())
                    .filter_map(|i| {
                        let sc = pat.score(s.text(i))?;
                        let mut keys = [0u32; 3];
                        for (k, t) in o.tiebreak.iter().take(3).enumerate() {
                            keys[k] = tiebreak_key(*t, s.text(i), sc.begin, sc.end, i as u32);
                        }
                        Some(Scored {
                            idx: i as u32,
                            score: sc.score,
                            keys,
                        })
                    })
                    .collect();
                sort(&mut want, &o);
                let want: Vec<Hit> = want.iter().map(Scored::hit).collect();
                assert_eq!(got.as_ref(), &want, "query {q:?}");
            }
        }
    }

    #[test]
    fn counting_sort_orders_like_the_comparator() {
        let s = store(
            &(0..60_000u32)
                .map(|i| {
                    // every 997th line an outlier: long (a low score, a great
                    // length), past the counting sort's caps
                    let pad = if i % 997 == 0 {
                        200 + (i as usize % 2800)
                    } else {
                        (i % 9) as usize
                    };
                    format!(
                        "{}l{}v{}/{}",
                        "x".repeat(pad),
                        i % 11,
                        "y".repeat((i % 4) as usize),
                        i
                    )
                })
                .collect::<Vec<_>>(),
        );
        for q in ["l", "lv", "v"] {
            for tb in [vec![Tiebreak::Length], vec![Tiebreak::Index], vec![]] {
                let mut o = opts(Scope::All);
                o.tiebreak = tb.clone();
                let Hits::List(got) = search(&s, q, &o, None, None).unwrap() else {
                    panic!()
                };
                assert!(
                    got.len() >= 1 << 14,
                    "the test must reach the counting sort"
                );
                let reference = reference(&s, q, &o);
                assert!(got.as_ref() == &reference, "query {q:?} tiebreak {tb:?}");
            }
        }
    }

    /// The common guard of every fast path: random candidates and queries
    /// over a mixed alphabet (ASCII, Cyrillic in both cases, Latin with
    /// diacritics, 3-byte characters, emoji, NBSP, tabs; lengths 0..6000) —
    /// `search` must return exactly what scoring every candidate one by one
    /// with the scalar scorer gives, in the same order.
    #[test]
    fn fast_paths_equal_the_scalar_scorer() {
        let mut seed = 0x51_7cc1_b727_220a_95u64;
        let mut rnd = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n.max(1) as u64) as usize
        };
        let alpha: Vec<char> = "abcdeflsrvLSRVA/_-.:,;| \t0129аблгтАБЛГТéèÉÈçñ中€\u{a0}😀"
            .chars()
            .collect();
        // half the lines ASCII only (the group kernel takes them up to 256
        // bytes), some empty, the rest from the whole alphabet
        let ascii: Vec<char> = alpha.iter().copied().filter(char::is_ascii).collect();
        let mut lines = Vec::new();
        for i in 0..6000 {
            let len = match i % 50 {
                0 => 1000 + rnd(5000),
                1..=4 => 200 + rnd(300),
                5 => 0,
                _ => rnd(80),
            };
            let from = if i % 2 == 0 { &ascii } else { &alpha };
            lines.push((0..len).map(|_| from[rnd(from.len())]).collect::<String>());
        }
        let s = store(&lines);
        let words: Vec<String> = (0..60)
            .map(|_| {
                (0..1 + rnd(6))
                    .map(|_| alpha[rnd(alpha.len())])
                    .filter(|c| !matches!(c, ' ' | '\t' | '|' | '\\'))
                    .collect()
            })
            .filter(|w: &String| !w.is_empty())
            .collect();
        let mut queries: Vec<String> = words.clone();
        for k in 0..words.len() {
            let a = &words[k];
            let b = &words[(k + 7) % words.len()];
            queries.push(format!("{a} {b}"));
            queries.push(format!("{a} !{b}"));
            queries.push(format!("'{a}"));
            queries.push(format!("^{a}"));
            queries.push(format!("{a}$"));
        }
        queries.extend(
            [
                "lv",
                "LV",
                "Лг",
                "лг",
                "e",
                "é",
                "abcdeflsrvabcdef",
                "s/r",
                "1 2",
                "^Ls",
                "^ls",
                "^лг",
                "^ЛГ",
                "'Лг",
                "'лг",
                "V$",
                "v$",
                "'s/r",
                "^abcdeflsrvabcdeflsrvabcdeflsrvabcdeflsrvabcdeflsrvabcdeflsrvabcdeflsrvabcdeflsrv",
            ]
            .map(String::from),
        );
        for path in [false, true] {
            let mut o = opts(Scope::All);
            o.cfg.scheme = if path { Scheme::Path } else { Scheme::Default };
            for q in &queries {
                let Hits::List(got) = search(&s, q, &o, None, None).unwrap() else {
                    continue;
                };
                let pat = Pattern::parse(q, &o.cfg);
                let mut want: Vec<Scored> = (0..s.len())
                    .filter_map(|i| {
                        let sc = pat.score(s.text(i))?;
                        Some(Scored {
                            idx: i as u32,
                            score: sc.score,
                            keys: [
                                tiebreak_key(
                                    Tiebreak::Length,
                                    s.text(i),
                                    sc.begin,
                                    sc.end,
                                    i as u32,
                                ),
                                0,
                                0,
                            ],
                        })
                    })
                    .collect();
                sort(&mut want, &o);
                let want: Vec<Hit> = want.iter().map(Scored::hit).collect();
                assert!(
                    got.as_ref() == &want,
                    "query {q:?} (path {path}): {} hits vs {}",
                    got.len(),
                    want.len()
                );
                // narrowing from the query without its last character
                let shorter: String = {
                    let mut c: Vec<char> = q.chars().collect();
                    c.pop();
                    c.into_iter().collect()
                };
                if !shorter.trim().is_empty() && narrows(&shorter, q) {
                    let mut ou = o.clone();
                    ou.sort = false;
                    // Hits::All (every line) leaves nothing to narrow
                    let Some(prev) = search(&s, &shorter, &ou, None, None).unwrap().indices()
                    else {
                        continue;
                    };
                    let Hits::List(inc) = search(&s, q, &o, None, Some((&prev, s.len()))).unwrap()
                    else {
                        panic!("narrowed {q:?}");
                    };
                    assert!(inc.as_ref() == &want, "narrowed {q:?} (path {path})");
                }
            }
        }
    }

    #[test]
    fn byte_tables_follow_the_bonus_rule() {
        let (class, _) = byte_tables();
        for path in [false, true] {
            let t = bonus_table(path);
            for p in 0..=255u8 {
                for c in 0..=255u8 {
                    let want = crate::matcher::byte_bonus(&[p, c], 1, path) as i16;
                    let got = t[(class[p as usize] as usize) << 5 | class[c as usize] as usize];
                    assert_eq!(got, want, "bytes {p} {c}, path {path}");
                }
            }
        }
    }

    #[test]
    fn unsorted_keeps_input_order() {
        let s = corpus();
        let mut o = opts(Scope::All);
        o.sort = false;
        let v = order(&search(&s, "lv", &o, None, None).unwrap());
        assert!(v.windows(2).all(|w| w[0] < w[1]));
        o.tac = true;
        let v = order(&search(&s, "lv", &o, None, None).unwrap());
        assert!(v.windows(2).all(|w| w[0] > w[1]));
    }
}
