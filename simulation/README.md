# Simulation

Parameter-recovery study on synthetic data, corresponding to the section *Simulation: Recovering parameters from synthetic data* in the paper.

Synthetic retrieval-practice data are generated under the ACT-R memory model with known parameter values, and the AMLE procedure is used to recover them.
The pipeline establishes (1) which combination of parameters can be identified, and (2) how recovery depends on the amount of available data.

## Scripts

1. [`01_simulate_data.Rmd`](./scripts/01_simulate_data.Rmd) — simulate data for 50 learners and characterise the parameter distributions and their effect on behaviour.
2. [`02_greedy_parameter_selection.Rmd`](./scripts/02_greedy_parameter_selection.Rmd) — greedy forward selection of participant-level parameters by AIC.
3. [`03_parameter_recovery.Rmd`](./scripts/03_parameter_recovery.Rmd) — recovery accuracy as a function of repetitions per fact and number of facts.

## Running

From the repository root:

```bash
make simulation        # runs all three scripts
```

Individual steps: `make sim-data`, `make sim-greedy`, `make sim-recovery`.

The expensive fits in scripts 02 and 03 are cached in [`../data/processed/fits/`](../data/processed/fits); the corresponding chunks are `eval = FALSE` and read those caches so the notebooks run quickly.
Each script renders to a GitHub-viewable `.md` in the shared [`../output/`](../output) folder, alongside its figure folder and the named PNG figures.

## Rendered notebooks

- [01_simulate_data.md](../output/01_simulate_data.md)
- [02_greedy_parameter_selection.md](../output/02_greedy_parameter_selection.md)
- [03_parameter_recovery.md](../output/03_parameter_recovery.md)

(These appear once you run `make simulation`.)
