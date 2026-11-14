# lvim-fuzzy-core

The lvim fuzzy search engine, in Rust: one implementation of the scoring
model, the query syntax and the search, shared by

- [lvim-fuzzy](https://github.com/lvim-tech/lvim-fuzzy) — Neovim (lvim-picker,
  lvim-cmp), through a thin C ABI wrapper;
- [lvim-fuzzy-cli](https://github.com/lvim-tech/lvim-fuzzy-cli) — the terminal
  finder, as a Rust dependency.

So a query ranks the same everywhere, and every speed-up reaches all of them.

## What is in it

- `Store` — the candidates in one buffer with an offset table (no string per
  candidate); a presence signature, an ASCII flag and the grep shape per
  candidate; streamed input in chunks of any size.
- `Pattern` — fzf's extended search syntax (`'exact`, `^prefix`, `suffix$`,
  `!not`, `a | b`), smart case, Latin normalisation, optional typos.
- `search` — every core, cancellation, narrowing while typing, tiebreaks,
  `tac`, grep scopes (`line`, `path`, `file`).
- The scoring model: [docs/scoring.md](docs/scoring.md).

## Benchmark: lvim-fuzzy-core vs frizbee

The ordered result of a query (every line scored, the matches sorted), in
milliseconds, the fastest of 15 runs; the faster side in bold. AMD Ryzen 9
3900XT (24 threads, AVX2), on every core and on one thread. The hits column
is core's count; frizbee's is the same everywhere except code `zzzz` (2220).

Out of 39 queries core is faster in 27 on every core and in
26 on one thread.

**paths** — 1 000 000 generated file paths

| query | hits | core, every core | frizbee, every core | core, one thread | frizbee, one thread |
|---|---:|---:|---:|---:|---:|
| `a` | 620164 | **3.50** | 5.41 | **19.41** | 30.20 |
| `lv` | 289599 | **2.72** | 4.05 | **13.41** | 24.43 |
| `lvim` | 254548 | **3.22** | 3.90 | **13.17** | 32.98 |
| `config` | 224784 | **3.12** | 3.66 | **13.92** | 39.41 |
| `srcmainrs` | 6080 | **1.21** | 2.15 | **6.45** | 19.92 |
| `pickerfuzzyengine` | 1576 | **0.96** | 2.17 | **9.96** | 15.69 |
| `zzzz` | 24505 | **1.44** | 2.24 | **7.34** | 10.40 |
| `'theme` | 224593 | **2.50** | 3.12 | 20.79 | **20.22** |
| `^src` | 50137 | **1.79** | 2.21 | 6.19 | **3.49** |
| `.rs$` | 166667 | **2.09** | 2.82 | 12.66 | **7.72** |
| `src lua` | 98723 | **2.14** | 3.50 | **10.57** | 39.94 |
| `lua !test` | 315037 | 5.35 | **4.86** | **44.26** | 46.66 |

**real** — 125 800 real file paths (a source tree)

| query | hits | core, every core | frizbee, every core | core, one thread | frizbee, one thread |
|---|---:|---:|---:|---:|---:|
| `a` | 113849 | **0.81** | 1.25 | 4.79 | **4.63** |
| `lv` | 27366 | **0.59** | 0.80 | **1.11** | 2.60 |
| `main` | 56186 | **1.01** | 1.17 | **4.39** | 7.59 |
| `readme` | 5531 | **0.60** | 0.70 | **1.70** | 3.03 |
| `srcmainrs` | 1941 | **0.45** | 0.68 | **1.74** | 2.74 |
| `zzzz` | 7 | **0.07** | 0.55 | **0.19** | 0.88 |
| `'test` | 2176 | 0.81 | **0.59** | 3.88 | **1.97** |
| `.rs$` | 930 | **0.20** | 0.52 | 0.77 | **0.37** |
| `src lua` | 16677 | **0.82** | 1.04 | **2.55** | 7.73 |

**code** — 400 000 lines of code (Lua, Rust, …)

| query | hits | core, every core | frizbee, every core | core, one thread | frizbee, one thread |
|---|---:|---:|---:|---:|---:|
| `a` | 388969 | 4.80 | **3.59** | 23.49 | **20.24** |
| `fn` | 164354 | 3.08 | **2.38** | 16.08 | **14.57** |
| `local` | 130316 | 3.68 | **2.92** | **24.15** | 31.27 |
| `function` | 27206 | 2.85 | **1.74** | **13.50** | 13.68 |
| `require` | 6976 | **1.36** | 1.61 | **5.32** | 8.78 |
| `vim.api` | 69172 | 3.52 | **2.48** | **19.61** | 29.30 |
| `zzzz` | 2224 | **1.10** | 1.47 | **4.59** | 5.04 |
| `'return nil` | 13859 | 4.35 | **1.82** | 38.94 | **12.20** |
| `fn !test` | 150029 | 5.42 | **2.73** | 40.28 | **21.12** |

**cyrillic** — 300 000 generated Cyrillic lines

| query | hits | core, every core | frizbee, every core | core, one thread | frizbee, one thread |
|---|---:|---:|---:|---:|---:|
| `а` | 298920 | **2.86** | 4.83 | **24.87** | 43.90 |
| `тр` | 263202 | **3.98** | 5.30 | **27.32** | 51.90 |
| `търс` | 122405 | 4.08 | **3.71** | **32.49** | 39.57 |
| `плъгин` | 119115 | 5.03 | **4.29** | **38.63** | 50.87 |
| `конфигурация` | 118128 | **7.32** | 9.20 | **63.68** | 121.87 |
| `тема` | 153487 | **4.59** | 4.94 | **36.28** | 56.19 |
| `zzzz` | 0 | **1.06** | 1.32 | 11.41 | **3.27** |
| `'сесия` | 118225 | 2.62 | **2.32** | 24.91 | **22.30** |
| `^търс` | 18570 | **1.06** | 1.45 | 4.47 | **4.19** |

Where frizbee is still ahead: literals on one thread (`^src`, `.rs$`, `'test`:
the cost per line), queries that mix a literal with a fuzzy word or a negation
(`'return nil`, `fn !test`), and the dense one-letter query on code.

The two do not do quite the same work:

- core scores exactly up to a window of 4096 characters (`exact_window`, 0 =
  always exactly); frizbee matches greedily past 1024 bytes.
- core breaks ties by length (fzf's default tiebreak), normalises Latin
  letters (`cafe` finds `café`) and takes fzf's whole query syntax.
- Not in the table: narrowing while typing (the next letter searches only the
  previous matches), which a library like frizbee leaves to its caller.

Run it on your machine (only core and frizbee; without `--frizbee` also
nucleo, skim's fuzzy-matcher and sublime_fuzzy):

    bench/run.sh --frizbee [SOURCE_TREE]

The generated corpora are the same everywhere; the real paths and the code
lines come from `SOURCE_TREE` (default: the current directory).

## Tests

    cargo test

## License

BSD-3-Clause
