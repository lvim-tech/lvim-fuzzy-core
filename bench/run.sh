#!/bin/sh
# lvim-fuzzy-core against frizbee, nucleo, skim's fuzzy-matcher and
# sublime_fuzzy on four corpora, on every core and on one thread; prints
# Markdown tables (the fastest in bold, the hit count in brackets).
#
#   bench/run.sh [--frizbee] [SRC]
#
# --frizbee: only lvim-fuzzy-core and frizbee (the same as ENGINES=frizbee;
# ENGINES takes any list of frizbee, nucleo, skim, sublime).
#
# SRC: a source tree for the "real paths" and "code lines" corpora (default:
# the current directory). The two generated corpora are the same on every
# machine. Corpora and the build go to ${TMPDIR:-/tmp}/lvim-fuzzy-core-bench.
# A quiet machine gives steadier numbers (the search uses every core).
set -e
if [ "$1" = "--frizbee" ]; then
    export ENGINES=frizbee
    shift
fi
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "${1:-.}" && pwd)
tmp=${TMPDIR:-/tmp}/lvim-fuzzy-core-bench
mkdir -p "$tmp"
(cd "$here" && CARGO_TARGET_DIR="$tmp/target" cargo build --release -q)
bin=$tmp/target/release/core-bench

[ -s "$tmp/paths.txt" ] || "$bin" gen paths 1000000 "$tmp/paths.txt"
[ -s "$tmp/cyrillic.txt" ] || "$bin" gen cyrillic 300000 "$tmp/cyrillic.txt"
files() { if command -v rg >/dev/null; then rg --files --hidden -g '!.git' .; else find . -type f -not -path '*/.git/*'; fi; }
lines() { if command -v rg >/dev/null; then rg -n --no-heading -g '*.rs' -g '*.lua' -g '*.c' -g '*.go' -g '*.py' -g '*.js' -g '*.ts' .; else grep -rn --include='*.rs' --include='*.lua' --include='*.c' --include='*.py' . ; fi; }
[ -s "$tmp/real.txt" ] || (cd "$src" && files | head -1000000 > "$tmp/real.txt")
[ -s "$tmp/code.txt" ] || (cd "$src" && lines | head -400000 > "$tmp/code.txt")

cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')
echo "CPU: ${cpu:-unknown}, $(nproc 2>/dev/null || echo ?) threads; load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
for mode in all single; do
    if [ "$mode" = single ]; then export THREADS=1; title="one thread"; else unset THREADS; title="every core"; fi
    for corpus in paths real code cyrillic; do
        f=$tmp/$corpus.txt
        echo
        echo "### $corpus ($(wc -l < "$f") lines), $title"
        echo
        case $corpus in
            paths) "$bin" "$f" a lv lvim config srcmainrs pickerfuzzyengine zzzz "'theme" "^src" '.rs$' "src lua" "lua !test" ;;
            real) "$bin" "$f" a lv main readme srcmainrs zzzz "'test" '.rs$' "src lua" ;;
            code) "$bin" "$f" a fn local function require vim.api zzzz "'return nil" "fn !test" ;;
            cyrillic) "$bin" "$f" а тр търс плъгин конфигурация тема zzzz "'сесия" "^търс" ;;
        esac
    done
done
