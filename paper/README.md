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
figures/fig_dataflow.tex  Fig 1  HBM dataflow, baseline vs fused (TikZ)
results/*.tex             generated tables and macros — do not edit by hand
scripts/                  measurement harness
```

## Regenerating the numbers

The tables under `results/` are generated; the prose in `main.tex` is written by
hand against them.

```bash
cd scripts
python bench.py       --device cuda --mode both --out ../results/bench_cuda.json
python correctness.py --device cuda            --out ../results/correctness.json
python make_tables.py --bench ../results/bench_cuda.json \
                      --correctness ../results/correctness.json
```

`make_tables.py` also writes `results/macros.tex`, which the manuscript does not
`\input`. It is a convenience: it carries the min/max over the level sweep of
every ratio the prose quotes, so the hand-written sentences can be checked
against the measurements without re-deriving them from the JSON.

Kernel sizes are never hard-coded in the generator: it reads them from the
benchmark JSON's recorded `args`, so a caption always describes the protocol
that was actually measured.

`make_tables.py` overwrites the same filenames, so the manuscript picks the new
values up on the next `pdflatex`. It also emits `env.tex` (hardware/software
versions, captured automatically).

`results/bench_cuda_k3.json` is a retained earlier sweep at `k=3`, kept only as
an ablation; the paper reports `k=5`, the kernel size WTConvNeXt uses.

## Scope note

`bench.py` is CUDA-only and compares four methods: `depthwise`, `dense`,
`reference` (the implementation of Finder et al.) and `fused_cuda`. Ratios are
always taken against `reference`, so higher is faster.

Two protocols:

- **homogeneous** — every method at the same `k`, isolating the cost of the
  wavelet machinery.
- **dropin** — WTConv at `k=5` against plain convolutions at `k=7`, the
  depthwise convolution WTConv is proposed to replace in a ConvNeXt block.
  This is *not* a receptive-field match: WTConv at `k=5` over `L` levels spans
  `5·2^L` ≥ 10 pixels, so the plain convolutions see strictly less context.
  Table 3 of the manuscript states the gap.

The wavelet methods are measured under `homogeneous` only — the protocols differ
solely in the plain convolutions' kernel size — and every ratio is taken against
that one denominator.
