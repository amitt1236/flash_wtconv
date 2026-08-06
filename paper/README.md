# Wavelet Convolutions Are Memory-Bound — TMLR submission

Source for the paper. Builds `main.pdf` with TeX Live 2026.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.bbl` is committed so the source can be uploaded to OpenReview/arXiv
without a bibtex run. Carry `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` with the
`.tex` — the bundled `fancyhdr.sty` shadows the system one and the build breaks
without it.

`math_commands.tex` is present but deliberately **not** loaded: it redefines
`\eqref` to print "equation N" instead of "(N)" and claims a very large macro
namespace. The preamble loads `amsmath` instead.

## Build modes

The preamble has all three TMLR modes; exactly one is active.

| Mode | Line | Effect |
|---|---|---|
| Submission (active) | `\usepackage{tmlr}` | authors hidden, "Under review as submission to TMLR" |
| Camera-ready | `\usepackage[accepted]{tmlr}` | also requires `\def\month`, `\def\year`, `\def\openreview` or the build fails |
| Preprint | `\usepackage[preprint]{tmlr}` | de-anonymised, no TMLR branding |

Before de-anonymising, replace the `\author{...}` block — it currently holds a
placeholder, since submission mode never expands it.

## Layout

```
main.tex                  manuscript
main.bib                  41 references, verified against DBLP/publisher records
appendix_repro.tex        Appendix A — reproducibility, claim-to-evidence map
appendix_backends.tex     Appendix C — Triton and Metal
figures/fig_dataflow.tex  Fig 1  HBM dataflow, baseline vs fused (TikZ)
results/*.tex             generated tables — do not edit by hand
scripts/                  measurement harness
```

The manuscript currently `\input`s only `figures/fig_dataflow.tex`,
`results/tab_rfmatch.tex` and `results/tab_backends.tex`. The other generated
fragments — `figures/fig_roofline.tex`, `figures/fig_convergence.tex`,
`results/tab_context.tex`, `results/tab_memory.tex`,
`results/tab_correctness.tex` — are kept because the harness regenerates them,
but they belong to the Experiments and Portability sections that were dropped
from `main.tex`, so nothing references them. Re-adding either section means
re-adding the corresponding `\input`.

## Regenerating the numbers

No number is typed into the manuscript. `results/*.tex` is generated, and the
paper `\input`s it.

Right now those tables come from `scripts/seed_from_report.py`, which encodes
the original measurement campaign (RTX A6000 / M3 Pro / L40S). To replace them
with fresh measurements on your own hardware:

```bash
cd scripts
python bench.py       --device cuda --mode both --out ../results/bench_cuda.json
python correctness.py --device cuda            --out ../results/correctness.json
python make_tables.py --bench ../results/bench_cuda.json \
                      --correctness ../results/correctness.json
```

`make_tables.py` overwrites the same filenames, so the manuscript picks the new
values up on the next `pdflatex`. It also emits `env.tex` (hardware/software
versions, captured automatically) and `macros.tex` (headline numbers).

### What still needs a GPU run

- **`results/tab_correctness.tex` is a placeholder.** Its structural rows are
  exact and machine-independent; the numerical deviation columns are `---`
  until `correctness.py` runs on a CUDA device. It is not currently `\input` by
  the manuscript (see the layout note above).

## Scope note

`bench.py` compares five methods: `depthwise`, `dense`, `reference` (the
implementation of Finder et al.), `fused_cuda`, and `fused_triton`. Ratios are
always taken against `reference`, so higher is faster.
