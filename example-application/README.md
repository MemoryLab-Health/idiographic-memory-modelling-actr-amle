# Example application

Application of the AMLE method to the longitudinal retrieval-practice data of [Hake et al. (2024)](https://www.medrxiv.org/content/10.1101/2024.03.15.24304345v1), corresponding to the section *Application: Interpreting parameters in a clinical sample* in the paper.

Participants completed weekly 8-minute learning sessions over about a year.
The sample included participants diagnosed with Mild Cognitive Impairment (MCI) and age-matched healthy control (HC) participants.
Hake et al. showed that clinical status could be recovered from the Speed of Forgetting alone; here the same data are fitted with more free parameters to ask whether group differences also surface elsewhere, and which parameters carry the diagnostic signal.

## Scripts

4. [`04_prepare_data.R`](./scripts/04_prepare_data.R) — clean and anonymise the raw Hake et al. (2024) data into `data/processed/hake2024.csv`.
5. [`05_fit_model.Rmd`](./scripts/05_fit_model.Rmd) — fit the AMLE model per session, producing `data/processed/AMLE_fit.csv` and the model-fit and parameter figures.
6. [`06_classification.Rmd`](./scripts/06_classification.Rmd) — greedy forward parameter selection by bootstrap-corrected AUC for MCI vs HC classification.

## Analysis notebook

- [Model fitting](../output/05_fit_model.md)

## Running

From the repository root:

```bash
make data              # 04 (requires the raw data; processed CSV is shipped)
make fit               # 05
make classify          # 06
```

Or run all three with `make application`.

The model fit in script 05 and the classification fits in script 06 are cached (`AMLE_fit.csv`, `data/processed/fits/greedy_auc_results*.rds`).
Set `refit_model <- TRUE` in script 05, or remove the cached `.rds` files, to refit from scratch.
Each script renders to a GitHub-viewable `.md` in the shared [`../output/`](../output) folder, alongside its figure folder and the named PNG figures.
