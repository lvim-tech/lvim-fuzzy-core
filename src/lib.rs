//! lvim-fuzzy-core: the lvim fuzzy search engine.
//!
//! One implementation of the lvim scoring model ([docs/scoring.md]), its query
//! syntax and its search, shared by lvim-fuzzy (Neovim, through a thin C ABI
//! wrapper) and lvim-fuzzy-cli (as a Rust dependency).
//!
//! - [`Store`]: the candidates in one buffer (no string per candidate);
//! - [`Pattern`]: a parsed query (fzf's extended syntax);
//! - [`search`]: the ordered result over a store, on every core.
//!
//! [docs/scoring.md]: https://github.com/lvim-tech/lvim-fuzzy-core/blob/main/docs/scoring.md

pub mod matcher;
pub mod search;
pub mod simd;
pub mod store;

pub use matcher::{
    CaseMode, EXACT_WINDOW, Literal, MatchConfig, Pattern, Scheme, Score, TYPO_PENALTY, TYPOS_AUTO,
    Tiebreak, normalize_char,
};
pub use search::{Hit, Hits, Options, Scope, narrows, search};
pub use store::Store;
