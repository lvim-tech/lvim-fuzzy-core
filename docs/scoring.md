# The scoring model

One affine-gap local alignment of the query (the needle) over a candidate
(the haystack), the same in Neovim (lvim-fuzzy) and the terminal
(lvim-fuzzy-cli).

## Constants

| | |
|---|---|
| match on the first character | +18 |
| match after a word boundary | +16 (+18 after `/` with the `path` scheme) |
| lower → Upper (camelCase) | +14 |
| a digit after a non-digit | +7 |
| a match right after the previous match | +16 |
| each character skipped before the first match | −1 |
| each character skipped between matches | −2 |
| each character after the last match | −1 |
| each character a term may miss (typos) | −40 |

Word boundaries: `/ _ - . : , ; |` and any white space.

## Case

`Smart` (default: a query with an upper-case letter is case-sensitive),
`Ignore`, `Respect`. Completion (lvim-cmp) uses `Ignore`.

## Ties

Equal scores are ordered by the tiebreak (default: the shorter candidate),
then by input order.

## How the scores are computed fast

The rules above define the score; three exact ways compute it:

- **The matrix** (`matcher::fuzzy`): the affine-gap alignment over the
  feasible window, two rows reused. The reference for everything else.
- **The rebased recursion** (`matcher::rebased_window`, `simd::score_window`):
  with `X' = X + j` the per-column gap disappears from the inner rows:

  ```
  row 0:     D'[j] = match ? b[j] : -inf
  row i > 0: D'[j] = match ? max(M'[i-1][j-1] + b[j] + 1, D'[i-1][j-1] + b[j] + 17) : -inf
  inner:     M'[i][j] = max(D'[i][j], M'[i][j-1] - 1)
  last row:  best = max over j of D'[m-1][j]
  score = best - (n - 1)
  ```

- **The group kernel** (`simd::group_*`): the same recursion for 32 lines at
  once, one `u8` lane per line, column by column, after transposing the lines.
  The values are biased so that 0 means unreachable: row `i` carries
  `B_i = 202 - 35 i`, and each step down a row subtracts 35 with saturation,
  which changes the bias and kills unreachable sources (they reach at most
  `18 + 17 = 35`). Exact while every reachable value of an inner row stays
  `>= 35` and no sum saturates before the subtraction: a line of width `W`
  for a needle of `m` characters needs `W <= 202 - 35 (m - 1)` (m = 1: 202,
  2: 167, 3: 132, 4: 97, 5: 62, 6: 27). Wider groups and needles up to 16 use
  the same kernel with `u16` lanes (bias 1024). Lines up to 256 bytes, ASCII.

Windows wider than `exact_window` (4096 by default, 0 = never) are matched
greedily: the leftmost occurrence, tightened from its end backwards, scored by
the same rules.

## Where the two old implementations differed

The Neovim matcher (lvim-fuzzy native, before this crate) had a smaller set of
word boundaries (`/ _ - . : ` and space), no digit bonus, no `path` scheme, and
always ignored case; ties went by byte length. This crate uses the rules
above for both, so rankings in Neovim can move where a candidate has `,` `;`
`|` or a tab before a match, or a digit run.
