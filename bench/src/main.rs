//! lvim-fuzzy-core against other fuzzy matchers, in memory: every line
//! scored, the matches collected and ordered by score; the fastest of N runs.
//!
//!   core-bench CORPUS QUERY...
//!   core-bench gen paths|cyrillic LINES FILE   (the generated corpora)
//!   THREADS=1    one thread (default: every core)
//!   CORE_ONLY=1  only lvim-fuzzy-core (for working on it: no waiting on the others)
//!   ENGINES=core,frizbee  which engines to run (core, frizbee, nucleo, skim,
//!                sublime; default all); core always runs
//!   RUNS=N       runs per measurement (default 15)
//!
//! bench/run.sh builds it, makes the corpora and runs the whole set.
//!
//! frizbee and nucleo understand fzf's syntax ('exact ^prefix suffix$ !not);
//! skim's fuzzy-matcher and sublime_fuzzy take one plain word, so a query
//! with spaces or operators is not run on them ("—").

use std::time::{Duration, Instant};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use lvim_fuzzy_core::{CaseMode, MatchConfig, Options, Scheme, Scope, Store, Tiebreak, search};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};
use rayon::prelude::*;

fn fastest(runs: usize, mut f: impl FnMut() -> usize) -> (f64, usize) {
    let mut best = Duration::MAX;
    let mut n = 0;
    for _ in 0..runs {
        let t = Instant::now();
        n = f();
        best = best.min(t.elapsed());
    }
    (best.as_secs_f64() * 1000.0, n)
}

/// Score in parallel (a matcher per thread), keep the matches, sort them.
fn ranked<S: Send, F>(lines: &[&str], init: impl Fn() -> S + Sync + Send, score: F) -> usize
where
    F: Fn(&mut S, &str) -> Option<i64> + Sync + Send,
{
    let mut hits: Vec<(i64, u32)> = lines
        .par_iter()
        .enumerate()
        .map_init(&init, |st, (i, l)| score(st, l).map(|s| (s, i as u32)))
        .flatten()
        .collect();
    hits.par_sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    hits.len()
}

/// A small xorshift generator: the same corpus on every machine.
struct Rng(u64);
impl Rng {
    fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// `gen paths|cyrillic LINES FILE`
fn generate(kind: &str, lines: usize, file: &str) {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(file).unwrap());
    let mut r = Rng(0x2545F4914F6CDD1D);
    match kind {
        "paths" => {
            let w = [
                "src", "lua", "lvim", "tui", "view", "core", "utils", "config", "test", "docs",
                "plugin", "init", "main", "model", "render", "cache", "theme", "picker", "fuzzy",
                "engine",
            ];
            let e = [".rs", ".lua", ".md", ".toml", ".txt", ".json"];
            for i in 0..lines {
                let depth = 2 + r.next(5);
                let dirs: Vec<&str> = (0..depth).map(|_| w[r.next(w.len())]).collect();
                writeln!(
                    out,
                    "{}/{}_{i}{}",
                    dirs.join("/"),
                    w[r.next(w.len())],
                    e[r.next(e.len())]
                )
                .unwrap();
            }
        }
        "cyrillic" => {
            let w = "търсене плъгин конфигурация тема прозорец редактор файл папка заявка резултат клавиш режим лента бутон грешка сесия"
                .split(' ')
                .collect::<Vec<_>>();
            for i in 0..lines {
                let n = 4 + r.next(9);
                let words: Vec<&str> = (0..n).map(|_| w[r.next(w.len())]).collect();
                writeln!(out, "{} #{i}", words.join(" ")).unwrap();
            }
        }
        _ => panic!("gen paths|cyrillic LINES FILE"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("gen") {
        return generate(&args[2], args[3].parse().unwrap(), &args[4]);
    }
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(15);
    let single = std::env::var("THREADS").is_ok_and(|t| t == "1");
    let core_only = std::env::var("CORE_ONLY").is_ok_and(|v| v == "1");
    let engines: Vec<String> = std::env::var("ENGINES")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|_| {
            ["frizbee", "nucleo", "skim", "sublime"]
                .map(String::from)
                .to_vec()
        });
    let on = |e: &str| !core_only && engines.iter().any(|x| x == e);
    let names = [
        ("frizbee", "frizbee"),
        ("nucleo", "nucleo"),
        ("skim", "skim (fuzzy-matcher)"),
        ("sublime", "sublime_fuzzy"),
    ];
    let threads = if single {
        1
    } else {
        rayon::current_num_threads()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let corpus = std::fs::read(&args[1]).unwrap();
    let mut store = Store::new();
    store.append(&corpus, b'\n');
    store.flush(b'\n');
    let text = String::from_utf8_lossy(&corpus);
    let lines: Vec<&str> = text.lines().collect();
    let o = Options {
        cfg: MatchConfig {
            case: CaseMode::Smart,
            extended: true,
            exact: false,
            normalize: true,
            scheme: Scheme::Default,
            typos: 0,
            exact_window: lvim_fuzzy_core::EXACT_WINDOW,
        },
        sort: true,
        tac: false,
        tiebreak: vec![Tiebreak::Length],
        scope: Scope::All,
    };
    let shown: Vec<&str> = names
        .iter()
        .filter(|(k, _)| on(k))
        .map(|(_, n)| *n)
        .collect();
    println!(
        "| query | lvim-fuzzy-core |{}",
        shown.iter().map(|n| format!(" {n} |")).collect::<String>()
    );
    println!("|---|---:|{}", "---:|".repeat(shown.len()));
    for q in &args[2..] {
        let plain = !q.contains([' ', '\'', '^', '$', '!', '|']);
        // (time, hits) per shown engine; None = not run for this query
        let mut cells: Vec<Option<(f64, usize)>> = Vec::new();
        pool.install(|| {
            cells.push(Some(fastest(runs, || {
                search(&store, q, &o, None, None).unwrap().len()
            })));
            if on("frizbee") {
                cells.push(Some(fastest(runs, || {
                    let mut m = frizbee::Matcher::from_query(q, &frizbee::Config::default());
                    if single {
                        m.match_list(&lines).len()
                    } else {
                        m.match_list_parallel(&lines, threads).len()
                    }
                })));
            }
            if on("nucleo") {
                cells.push(Some(fastest(runs, || {
                    let pat = Pattern::parse(q, CaseMatching::Smart, Normalization::Smart);
                    ranked(
                        &lines,
                        || (Matcher::new(Config::DEFAULT), Vec::new()),
                        |(m, buf), l| pat.score(Utf32Str::new(l, buf), m).map(i64::from),
                    )
                })));
            }
            if on("skim") {
                cells.push(plain.then(|| {
                    fastest(runs, || {
                        ranked(&lines, SkimMatcherV2::default, |m, l| m.fuzzy_match(l, q))
                    })
                }));
            }
            if on("sublime") {
                cells.push(plain.then(|| {
                    fastest(runs, || {
                        ranked(
                            &lines,
                            || (),
                            |_, l| sublime_fuzzy::best_match(q, l).map(|m| m.score() as i64),
                        )
                    })
                }));
            }
        });
        let best = cells.iter().flatten().map(|c| c.0).fold(f64::MAX, f64::min);
        let mut row = format!("| `{q}` |");
        for c in &cells {
            match c {
                Some((t, n)) => {
                    let cell = format!("{t:.2} ms ({n})");
                    row.push_str(&if *t == best {
                        format!(" **{cell}** |")
                    } else {
                        format!(" {cell} |")
                    });
                }
                None => row.push_str(" — |"),
            }
        }
        println!("{row}");
    }
}
