# Wavelet Convolutions Are Memory-Bound — TMLR submission

Source for the paper. Builds to a 17-page `main.pdf` with TeX Live 2026.

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
figures/fig_roofline.tex  Fig 2  roofline placement (TikZ)
figures/fig_convergence.tex Fig 3 CIFAR-10 curves, cropped from wandb.pdf
results/*.tex             generated tables — do not edit by hand
scripts/                  measurement harness
```

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
  until `correctness.py` runs on a CUDA device. This is the one table in the
  paper that is not yet populated with measurements.
- The paper quotes a $12\%$ maximum deviation between the I/O model and the
  measured `fp32` speedup. `seed_from_report.py` computes that deviation row
  from the data, so it updates itself; the prose in `main.tex` §1, §6.3 and §10
  quotes it and must be checked if the measurements change.

## Scope note

`bench.py` compares five methods: `depthwise`, `dense`, `reference` (the
implementation of Finder et al.), `fused_cuda`, and `fused_triton`. Ratios are
always taken against `reference`, so higher is faster.
