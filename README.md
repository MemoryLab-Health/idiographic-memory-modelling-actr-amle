# Idiographic Memory Modelling in ACT-R using Alternating Maximum Likelihood Estimation

Code and data accompanying the paper *Idiographic Memory Modelling in ACT-R using Alternating Maximum Likelihood Estimation* (van der Velde, Wilschut, Hake, van Rijn, & Stocco).

The repository contains two analysis pipelines, with scripts numbered sequentially across the whole project:

- **`simulation/`** (scripts 01–03) — a parameter-recovery study on synthetic data, establishing that the five participant-level ACT-R parameters and the fact-level offsets are jointly identifiable from realistic amounts of retrieval-practice data.
- **`example-application/`** (scripts 04–06) — an application to the longitudinal clinical data of [Hake et al. (2024)](https://www.medrxiv.org/content/10.1101/2024.03.15.24304345v1), fitting the model and asking which parameters distinguish mild cognitive impairment (MCI) from healthy controls (HC).

Both pipelines share a single implementation of the ACT-R likelihood, the simulator, and the AMLE fitting procedure, in [`R/`](./R).

## Repository structure

```
actr-amle/
├── Makefile                  One entry point for every step (see `make help`)
├── R/                        Shared model code
│   ├── sim-mle.R             Simulator + AMLE fitting / recovery functions
│   └── actr_ll.cpp           C++ likelihood (compiled on load)
├── data/
│   ├── raw/                  Undistributed source data (git-ignored)
│   └── processed/            Shipped: prepared data + cached model fits
│       ├── hake2024.csv
│       ├── AMLE_fit.csv
│       ├── AMLE_delta_alpha.csv
│       └── fits/             Cached `.rds` results for the expensive steps
├── simulation/
│   └── scripts/
│       ├── 01_simulate_data.Rmd
│       ├── 02_greedy_parameter_selection.Rmd
│       └── 03_parameter_recovery.Rmd
├── example-application/
│   └── scripts/
│       ├── 04_prepare_data.R
│       ├── 05_fit_model.Rmd
│       └── 06_classification.Rmd
└── output/                   All figures (named PNGs from ggsave)
```

## Reproducing the analyses

All steps run through the top-level `Makefile`. List the available targets with:

```bash
make help
```

### Simulation pipeline (scripts 01–03)

No raw data is required; everything is generated from a fixed seed.

```bash
make simulation        # runs 01 -> 02 -> 03
```

### Example application (scripts 04–06)

The processed data and cached fits are shipped, so the model-fitting and classification steps run without the raw source file:

```bash
make fit               # 05: fit the AMLE model -> data/processed/AMLE_fit.csv
make classify          # 06: parameter contributions to MCI vs HC classification
```

Rebuilding the processed data from scratch requires the raw file (see below):

```bash
make data              # 04: data/raw/hake2024.Rdata -> data/processed/hake2024.csv
```

Run everything end to end with `make all`.

## Output

Everything lands in a single [`output/`](./output) folder. Each notebook renders to a GitHub-viewable `.md` (with its `<name>_files/` figure folder alongside), and the named paper figures are also written there as PNGs via `ggsave()`.

Rendered analysis notebook:

- Example application: [model fitting](./output/05_fit_model.md)

`make clean` removes the rendered notebooks and figure folders; `make` regenerates them.

## Data availability

The processed retrieval-practice data and the cached model fits needed to reproduce the figures are included under [`data/processed/`](./data/processed).

The **raw** source data (`data/raw/hake2024.Rdata`) from Hake et al. (2024) is not distributed with this repository and is excluded via `.gitignore`. To regenerate the processed data, place that file in `data/raw/` and run `make data`.

## Notes

- All scripts resolve paths through `here::here()`, anchored at the repository root (`actr-amle.Rproj`/`.here`), so they can be run or knit from any working directory.
- The expensive fitting steps (greedy selection, recovery grids, classification) are guarded by `eval = FALSE` chunks and read cached results from `data/processed/fits/`. Use `make clean-cache` to force a full refit (slow).

## Citation

If you use this code, please cite the accompanying paper.

## Conflict of interest

MvdV, TW, HvR and AS are associated with Precision Cognition Labs, a commercial partner with an interest in the validation and implementation of the Seattle-Groningen Memory Assessment, which is based on earlier versions of the memory model described here.
