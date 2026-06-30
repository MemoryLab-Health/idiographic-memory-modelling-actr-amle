# ============================================================================
#  Idiographic Memory Modelling in ACT-R (AMLE)
#  One Makefile to reproduce the two analysis pipelines:
#    Scripts 01-03  simulation          (synthetic-data parameter recovery)
#    Scripts 04-06  example-application (clinical MCI vs HC + classification)
#
#  Each notebook renders to a GitHub-viewable .md in output/, alongside its
#  figure folder; the named paper figures are also written to output/ via
#  ggsave().
#
#  The processed data and cached model fits are shipped, so `make fit` and
#  `make classify` run out of the box. Only `make data` needs the raw source
#  file (see data/raw/.gitkeep).
#
#  Run `make help` for an overview of all targets.
# ============================================================================

# Use bash so we can filter a harmless pandoc stderr warning below.
SHELL   := /bin/bash

R       := Rscript
OUT     := output

# Render a notebook to a GitHub-viewable .md in output/.
# Recent pandoc prints "Deprecated: --highlight-style" (a flag rmarkdown still
# passes); it is harmless for github_document output, so we filter just that one
# line out of stderr. The Rscript exit status is preserved, so real errors still
# surface and stop the build.
RENDER   = $(R) -e "rmarkdown::render('$(1)', output_format = 'github_document', output_dir = '$(OUT)', quiet = TRUE)" \
             2> >(grep -v 'Deprecated: --highlight-style' >&2)

SIM_DIR := simulation/scripts
APP_DIR := example-application/scripts

RAW     := data/raw/hake2024.Rdata
PROC    := data/processed/hake2024.csv
FIT     := data/processed/AMLE_fit.csv

.PHONY: all simulation application \
        sim-data sim-greedy sim-recovery data fit classify \
        walkthrough help clean clean-cache
.DEFAULT_GOAL := help

## all          : run both pipelines end to end (including walkthrough)
all: walkthrough simulation application

## walkthrough  : 00 model walkthrough notebook (renders to output/)
walkthrough:
	$(call RENDER,00_model_walkthrough.Rmd)

# ---------------------------------------------------------------------------
# Simulation pipeline (synthetic data; self-contained, no raw data needed)
# ---------------------------------------------------------------------------

## simulation   : run the full simulation pipeline (01 -> 02 -> 03)
simulation: sim-data sim-greedy sim-recovery

## sim-data     : 01 simulate data + parameter/behaviour figure
sim-data:
	$(call RENDER,$(SIM_DIR)/01_simulate_data.Rmd)

## sim-greedy   : 02 greedy forward parameter selection (AIC)
sim-greedy:
	$(call RENDER,$(SIM_DIR)/02_greedy_parameter_selection.Rmd)

## sim-recovery : 03 parameter recovery as a function of available data
sim-recovery:
	$(call RENDER,$(SIM_DIR)/03_parameter_recovery.Rmd)

# ---------------------------------------------------------------------------
# Example application (real data from Hake et al., 2024)
# ---------------------------------------------------------------------------

## application  : run the full application pipeline (04 -> 05 -> 06)
application: data fit classify

## data         : 04 (re)build processed data from raw (needs data/raw/hake2024.Rdata)
data:
	@if test -f $(RAW); then \
	  $(R) $(APP_DIR)/04_prepare_data.R; \
	else \
	  echo ">> $(RAW) not found — see data/raw/.gitkeep."; \
	  echo ">> Processed data is already shipped in data/processed/; skipping."; \
	fi

## fit          : 05 fit the AMLE model -> AMLE_fit.csv + figures
fit:
	@test -f $(PROC) || { echo "ERROR: $(PROC) missing — run 'make data' first."; exit 1; }
	$(call RENDER,$(APP_DIR)/05_fit_model.Rmd)

## classify     : 06 parameter contributions to MCI vs HC classification
classify:
	@test -f $(FIT) || { echo "ERROR: $(FIT) missing — run 'make fit' first."; exit 1; }
	$(call RENDER,$(APP_DIR)/06_classification.Rmd)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

## clean        : remove rendered notebooks (.md + figure folders) and HTML
clean:
	find $(OUT) -type d -name '*_files' -prune -exec rm -rf {} +
	find $(OUT) -type f \( -name '*.md' -o -name '*.nb.html' -o -name '*.html' \) -delete

## clean-cache  : also remove cached model fits (forces full refit; slow!)
clean-cache: clean
	rm -f data/processed/fits/*.rds

## help         : show this message
help:
	@grep -E '^##' $(MAKEFILE_LIST) | sed -e 's/## //'
