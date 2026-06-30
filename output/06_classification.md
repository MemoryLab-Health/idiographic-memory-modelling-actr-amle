Example application: parameter contributions to MCI vs HC classification
================
Thomas Wilschut
Last updated: 2026-06-30

- [Setup and helpers](#setup-and-helpers)
  - [Helpers](#helpers)
- [Greedy forward selection with
  refitting](#greedy-forward-selection-with-refitting)
  - [Data preparation](#data-preparation)
  - [Greedy selection: person-average
    criterion](#greedy-selection-person-average-criterion)
  - [Plot: person-average greedy
    selection](#plot-person-average-greedy-selection)
  - [Single-session AUC from person-average selection
    path](#single-session-auc-from-person-average-selection-path)
  - [Greedy selection: single-session criterion (steps
    3–5)](#greedy-selection-single-session-criterion-steps-35)
  - [Plot: combined figure (person-average +
    single-session)](#plot-combined-figure-person-average--single-session)
  - [Summary statistics for paper](#summary-statistics-for-paper)
- [Session info](#session-info)

# Setup and helpers

The classification analysis below reports performance as a
bootstrap-corrected AUC, to guard against the optimism of evaluating a
model on the same data it was fit to. We first set up the parameter
lookups and the bootstrap helper used throughout.

## Helpers

A few display lookups (full parameter names and a consistent colour
palette) that are reused across the plots below.

``` r
# ── Shared constants ──────────────────────────────────────────────────────────
PARAM_COL <- c(phi = "phi", tau = "tau", s = "s", F = "lf", ter = "ter")

param_display <- c(
  phi = "Speed of Forgetting",
  tau = "Retrieval threshold",
  s   = "Activation noise",
  F   = "Latency factor",
  ter = "Non-retrieval time"
)

param_cols <- setNames(
  pal_aaas()(5),
  c("Speed of Forgetting", "Retrieval threshold", "Activation noise",
    "Latency factor", "Non-retrieval time")
)
```

The idea is to first compute the apparent AUC (the model evaluated on
the same data it was fit to, which is optimistically biased), then
estimate that optimism by repeatedly refitting on bootstrap resamples
and re-evaluating on the original data; the corrected AUC subtracts the
mean optimism. `bootstrap_auc_corrected()` implements this and is the
function used inside the greedy selection below.

``` r
# ── Bootstrap-corrected AUC (general, used in greedy selection) ───────────────
bootstrap_auc_corrected <- function(outcome, predictors_df,
                                    n_boot = 500, seed = 42) {
  set.seed(seed)
  data   <- data.frame(outcome = outcome, predictors_df)
  n      <- nrow(data)
  params <- names(predictors_df)

  fmla <- if (length(params) == 1) NULL
          else as.formula(paste("outcome ~", paste(params, collapse = " + ")))

  if (is.null(fmla)) {
    roc_app <- roc(data$outcome, data[[params]], direction = "auto", quiet = TRUE)
    p_app   <- data[[params]]
  } else {
    fit_full <- glm(fmla, data = data, family = binomial())
    p_app    <- predict(fit_full, type = "response")
    roc_app  <- roc(data$outcome, p_app, direction = "auto", quiet = TRUE)
  }
  auc_app <- as.numeric(auc(roc_app))

  boot_aucs <- replicate(n_boot, {
    idx    <- sample(n, n, replace = TRUE)
    b_data <- data[idx, ]
    if (length(unique(b_data$outcome)) < 2) return(NA_real_)
    if (is.null(fmla)) {
      p_orig <- data[[params]]
    } else {
      fit_b <- tryCatch(glm(fmla, data = b_data, family = binomial()),
                        error = function(e) NULL)
      if (is.null(fit_b)) return(NA_real_)
      p_orig <- predict(fit_b, newdata = data, type = "response")
    }
    tryCatch(
      as.numeric(auc(roc(data$outcome, p_orig, direction = "auto", quiet = TRUE))),
      error = function(e) NA_real_
    )
  })

  optimism      <- auc_app - mean(boot_aucs, na.rm = TRUE)
  auc_corrected <- auc_app - optimism

  list(
    auc_apparent  = auc_app,
    auc_corrected = auc_corrected,
    auc_lo        = quantile(boot_aucs, 0.025, na.rm = TRUE),
    auc_hi        = quantile(boot_aucs, 0.975, na.rm = TRUE)
  )
}
```

# Greedy forward selection with refitting

To see which parameters carry the diagnostic signal, we perform a greedy
forward selection: starting from no free parameters, at each step we add
the parameter that most improves classification. Crucially, for each
candidate parameter set the data are refit with the non-selected
parameters held fixed at their population defaults, so that each model
genuinely uses only the information in its selected parameters.

## Data preparation

For the refitting analysis we reload the raw per-trial data and apply
the same filtering rules as Hake et al. (drop fact sequences with fewer
than three encounters, or with response times below 300 ms or above 15
s), and code the outcome as MCI = 1 / HC = 0.

``` r
future::plan(future::multisession, workers = future::availableCores() - 1)

source(here::here("R", "sim-mle.R"))
d <- fread(here::here("data", "processed", "hake2024.csv"))

min_rt             <- 0.3
max_rt             <- 15
min_reps_per_fact  <- 3

fact_sequences <- split(d, by = c("user_id", "session_id", "fact_id"))
fact_sequences_filtered <- fact_sequences[
  map_lgl(fact_sequences, ~ nrow(.x) >= min_reps_per_fact &&
            min(.x$rt) >= min_rt && max(.x$rt) <= max_rt)
]

d_filtered <- rbindlist(fact_sequences_filtered)
setorder(d_filtered, user_id, start_time)

clinical_df <- unique(d_filtered[, .(user_id, clinical_status)])[
  , outcome := as.integer(clinical_status == "MCI")
]
```

## Greedy selection: person-average criterion

`greedy_select_auc()` performs the forward selection: starting from no
free parameters, at each step it refits every session with each
remaining candidate added to the already-selected set, aggregates the
session fits to person-level means, and keeps the candidate that yields
the highest bootstrap-corrected AUC. It returns both the per-step
summary and the underlying fits (the latter are reused below to evaluate
the single-session criterion without refitting).

``` r
# ── Aggregate session-level fits to person-level means ────────────────────────
person_means <- function(fits_dt, free_params, clinical_df) {
  cols <- PARAM_COL[free_params]
  cols <- cols[cols %in% names(fits_dt)]
  agg  <- fits_dt[, lapply(.SD, mean, na.rm = TRUE),
                  by = .(user_id), .SDcols = cols]
  merge(agg, clinical_df, by = "user_id")
}

# ── Greedy forward selection (person-average AUC criterion) ───────────────────
greedy_select_auc <- function(d_filtered,
                              all_params = c("phi", "tau", "s", "F", "ter"),
                              n_boot     = 500,
                              full_iter  = 100) {

  clinical_df <- unique(d_filtered[, .(user_id, clinical_status)])[
    , outcome := as.integer(clinical_status == "MCI")
  ]
  d_sessions <- split(d_filtered, by = "session_id")

  fit_sessions <- function(free_params) {
    message("  Fitting ", length(d_sessions), " sessions with free: {",
            paste(free_params, collapse = ", "), "} ...")
    results <- future_map(d_sessions, function(d_i) {
      fit_i <- fit_one_learner(d_i, free_params = free_params,
                               full_iter = full_iter)[.N]
      if (is.null(fit_i) || nrow(fit_i) == 0) return(data.table())
      fit_i[, `:=`(user_id         = d_i$user_id[1],
                   clinical_status = d_i$clinical_status[1],
                   session_id      = d_i$session_id[1])]
      fit_i
    }, .options = furrr_options(seed = TRUE), .progress = interactive())
    results <- results[!vapply(results, function(x) is.null(x) || nrow(x) == 0,
                               logical(1))]
    rbindlist(results, fill = TRUE)
  }

  selected  <- character(0)
  remaining <- all_params
  log_rows  <- list()
  all_fits  <- list()

  for (step in seq_along(all_params)) {
    message("\n=== Step ", step, ": testing ", length(remaining), " candidate(s) ===")
    step_rows <- list()

    for (candidate in remaining) {
      trial_params <- c(selected, candidate)
      message("  Candidate: {", paste(trial_params, collapse = ", "), "}")

      fits_dt <- fit_sessions(trial_params)
      if (nrow(fits_dt) == 0) { message("  WARNING: no fits returned"); next }

      pm        <- person_means(fits_dt, trial_params, clinical_df)
      pred_cols <- PARAM_COL[trial_params]
      pred_cols <- pred_cols[pred_cols %in% names(pm)]
      auc_res   <- bootstrap_auc_corrected(pm$outcome, pm[, ..pred_cols],
                                           n_boot = n_boot)

      message(sprintf("    AUC corrected = %.3f [%.3f-%.3f]",
                      auc_res$auc_corrected, auc_res$auc_lo, auc_res$auc_hi))

      step_rows[[candidate]] <- data.table(
        step = step, param_added = candidate, free_params = list(trial_params),
        auc_apparent = auc_res$auc_apparent, auc_corrected = auc_res$auc_corrected,
        auc_lo = auc_res$auc_lo, auc_hi = auc_res$auc_hi, selected = FALSE
      )

      key      <- paste(step, candidate, sep = "_")
      fit_copy <- copy(fits_dt)
      fit_copy[, `:=`(step = step, param_added = candidate,
                      free_params_str = paste(trial_params, collapse = ","))]
      all_fits[[key]] <- fit_copy
    }

    step_dt  <- rbindlist(step_rows)
    best_idx <- which.max(step_dt$auc_corrected)
    best_par <- step_dt$param_added[best_idx]
    step_dt[best_idx, selected := TRUE]

    message(sprintf("  >>> Selected: %s  (AUC = %.3f)",
                    best_par, step_dt$auc_corrected[best_idx]))

    log_rows[[step]] <- step_dt
    selected  <- c(selected, best_par)
    remaining <- setdiff(remaining, best_par)
  }

  list(summary = rbindlist(log_rows),
       learner_fits = rbindlist(all_fits, fill = TRUE))
}
```

``` r
# Expensive (refits every session for every candidate); cached and reloaded below.
greedy_auc <- greedy_select_auc(d_filtered)
saveRDS(greedy_auc, here::here("data", "processed", "fits", "greedy_auc_results.rds"))
```

## Plot: person-average greedy selection

This plots the person-average selection path (Figure 10A): the solid
line is the selected path, faint points are the non-selected candidates
at each step, and the shaded band is the 95% bootstrap CI on the
selected path. F is selected first and φ second, after which adding
further parameters does not improve classification.

``` r
greedy_auc <- readRDS(here::here("data", "processed", "fits", "greedy_auc_results.rds"))

# Backward compatibility: rename alpha -> phi in cached results
if ("alpha" %in% names(greedy_auc$learner_fits)) setnames(greedy_auc$learner_fits, "alpha", "phi")
if ("param_added" %in% names(greedy_auc$learner_fits)) greedy_auc$learner_fits[param_added == "alpha", param_added := "phi"]
if ("param_added" %in% names(greedy_auc$summary))      greedy_auc$summary[param_added == "alpha", param_added := "phi"]
if ("free_params" %in% names(greedy_auc$summary)) {
  greedy_auc$summary[, free_params := lapply(free_params, function(fp) replace(fp, fp == "alpha", "phi"))]
}

summary_dt <- copy(greedy_auc$summary)
summary_dt[, param_label := param_display[param_added]]

selection_path <- summary_dt[selected == TRUE][order(step)]
selection_path[, param_label := param_display[param_added]]

seg_data <- summary_dt[selected == FALSE]
seg_data[, y_from := selection_path$auc_corrected[
  match(step - 1, selection_path$step)
]]

p_auc <- ggplot() +
  geom_segment(
    data = seg_data[!is.na(y_from)],
    aes(x = step - 1, xend = step, y = y_from, yend = auc_corrected,
        color = 'black'),
    linewidth = 0.4, linetype = "dashed", alpha = 0.7
  ) +
  geom_line(data = selection_path,
            aes(x = step, y = auc_corrected),
            linewidth = 1.2, color = "black") +
  geom_ribbon(data = selection_path,
              aes(x = step, ymin = auc_lo, ymax = auc_hi),
              alpha = 0.15, fill = "grey40") +
  geom_point(data = summary_dt[selected == FALSE],
             aes(x = step, y = auc_corrected, color = param_label),
             size = 3, shape = 16, alpha = 0.6) +
  geom_point(data = selection_path,
             aes(x = step, y = auc_corrected, color = param_label),
             size = 5, shape = 17) +
  geom_text(
    data = summary_dt[selected == FALSE & auc_corrected < 0.6],
    aes(x = step + 0.15, y = 0.62,
        label = paste0(param_label, " (",
                       scales::percent(auc_corrected, accuracy = 1), ")"),
        color = 'black'),
    size = 4, hjust = 0, show.legend = FALSE
  ) +
  scale_color_manual(name = "Parameter", values = param_cols) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(limits = c(0.6, 1),
                     breaks = seq(0.6, 1, by = 0.05),
                     oob = scales::squish) +
  labs(x = "Step (parameter added)",
       y = "Classification AUC (bootstrap-corrected)") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none", panel.grid.minor = element_blank())

p_auc
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/06_classification_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## Single-session AUC from person-average selection path

Compute single-session AUCs for all candidate models fitted during the
person-average greedy selection (i.e., same fits, different evaluation
criterion).

``` r
summary_single <- map_dfr(
  split(greedy_auc$summary, seq_len(nrow(greedy_auc$summary))),
  function(s) {
    fp      <- s$free_params[[1]]
    step_i  <- s$step[[1]]
    param_i <- s$param_added[[1]]

    fits_i    <- greedy_auc$learner_fits[step == step_i & param_added == param_i]
    fits_i    <- merge(fits_i, clinical_df, by = "user_id")
    pred_cols <- unname(PARAM_COL[fp])
    pred_cols <- pred_cols[pred_cols %in% names(fits_i)]

    auc_res <- bootstrap_auc_corrected(
      fits_i$outcome, as.data.frame(fits_i[, ..pred_cols]), n_boot = 500
    )

    data.table(
      step = step_i, param_added = param_i, free_params = list(fp),
      auc_apparent = auc_res$auc_apparent, auc_corrected = auc_res$auc_corrected,
      auc_lo = auc_res$auc_lo, auc_hi = auc_res$auc_hi,
      selected = s$selected[[1]]
    )
  }
)

summary_single[, param_label := param_display[param_added]]
selection_single <- summary_single[selected == TRUE][order(step)]

seg_single <- summary_single[selected == FALSE]
seg_single[, y_from := selection_single$auc_corrected[
  match(step - 1, selection_single$step)
]]
```

## Greedy selection: single-session criterion (steps 3–5)

The selection path diverges from step 3 onward. Since steps 1–2 selected
the same parameters (F, then phi) under both criteria, we reuse those
fits and rerun only steps 3–5 with single-session AUC as the selection
criterion.

``` r
# ── Greedy selection from a given step (reusing earlier fits) ──────────────────
greedy_select_auc_from_step <- function(d_filtered,
                                        already_selected = c("F", "phi"),
                                        remaining_params = c("tau", "s", "ter"),
                                        start_step       = 3,
                                        n_boot           = 500,
                                        full_iter        = 100) {

  clinical_df <- unique(d_filtered[, .(user_id, clinical_status)])[
    , outcome := as.integer(clinical_status == "MCI")
  ]
  d_sessions <- split(d_filtered, by = "session_id")

  fit_sessions <- function(free_params) {
    message("  Fitting ", length(d_sessions), " sessions with free: {",
            paste(free_params, collapse = ", "), "} ...")
    results <- future_map(d_sessions, function(d_i) {
      fit_i <- fit_one_learner(d_i, free_params = free_params,
                               full_iter = full_iter)[.N]
      if (is.null(fit_i) || nrow(fit_i) == 0) return(data.table())
      fit_i[, `:=`(user_id         = d_i$user_id[1],
                   clinical_status = d_i$clinical_status[1],
                   session_id      = d_i$session_id[1])]
      fit_i
    }, .options = furrr_options(seed = TRUE), .progress = interactive())
    results <- results[!vapply(results, function(x) is.null(x) || nrow(x) == 0,
                               logical(1))]
    rbindlist(results, fill = TRUE)
  }

  selected  <- already_selected
  remaining <- remaining_params
  log_rows  <- list()
  all_fits  <- list()

  for (step in start_step:(start_step + length(remaining_params) - 1)) {
    message("\n=== Step ", step, ": testing ", length(remaining), " candidate(s) ===")
    step_rows <- list()

    for (candidate in remaining) {
      trial_params <- c(selected, candidate)
      message("  Candidate: {", paste(trial_params, collapse = ", "), "}")

      fits_dt <- fit_sessions(trial_params)
      if (nrow(fits_dt) == 0) next

      fits_with_outcome <- merge(fits_dt, clinical_df, by = "user_id")
      pred_cols <- unname(PARAM_COL[trial_params])
      pred_cols <- pred_cols[pred_cols %in% names(fits_with_outcome)]

      auc_res <- bootstrap_auc_corrected(
        fits_with_outcome$outcome,
        as.data.frame(fits_with_outcome[, ..pred_cols]),
        n_boot = n_boot
      )

      message(sprintf("    AUC corrected = %.3f [%.3f-%.3f]",
                      auc_res$auc_corrected, auc_res$auc_lo, auc_res$auc_hi))

      step_rows[[candidate]] <- data.table(
        step = step, param_added = candidate, free_params = list(trial_params),
        auc_apparent = auc_res$auc_apparent, auc_corrected = auc_res$auc_corrected,
        auc_lo = auc_res$auc_lo, auc_hi = auc_res$auc_hi, selected = FALSE
      )

      key      <- paste(step, candidate, sep = "_")
      fit_copy <- copy(fits_dt)
      fit_copy[, `:=`(step = step, param_added = candidate,
                      free_params_str = paste(trial_params, collapse = ","))]
      all_fits[[key]] <- fit_copy
    }

    step_dt  <- rbindlist(step_rows)
    best_idx <- which.max(step_dt$auc_corrected)
    best_par <- step_dt$param_added[best_idx]
    step_dt[best_idx, selected := TRUE]

    message(sprintf("  >>> Selected: %s  (AUC = %.3f)",
                    best_par, step_dt$auc_corrected[best_idx]))

    log_rows[[step - start_step + 1]] <- step_dt
    selected  <- c(selected, best_par)
    remaining <- setdiff(remaining, best_par)
  }

  list(summary = rbindlist(log_rows),
       learner_fits = rbindlist(all_fits, fill = TRUE))
}
```

``` r
new_steps <- greedy_select_auc_from_step(
  d_filtered,
  already_selected = c("F", "phi"),
  remaining_params = c("tau", "s", "ter"),
  start_step       = 3,
  n_boot           = 500
)

# Recompute single-session AUC for reused steps 1–2
reused_summary <- map_dfr(
  split(greedy_auc$summary[step %in% 1:2],
        seq_len(nrow(greedy_auc$summary[step %in% 1:2]))),
  function(s) {
    fp      <- s$free_params[[1]]
    step_i  <- s$step[[1]]
    param_i <- s$param_added[[1]]

    fits_i    <- greedy_auc$learner_fits[step == step_i & param_added == param_i]
    fits_i    <- merge(fits_i, clinical_df, by = "user_id")
    pred_cols <- unname(PARAM_COL[fp])
    pred_cols <- pred_cols[pred_cols %in% names(fits_i)]

    auc_res <- bootstrap_auc_corrected(
      fits_i$outcome, as.data.frame(fits_i[, ..pred_cols]), n_boot = 500
    )

    data.table(
      step = step_i, param_added = param_i, free_params = list(fp),
      auc_apparent = auc_res$auc_apparent, auc_corrected = auc_res$auc_corrected,
      auc_lo = auc_res$auc_lo, auc_hi = auc_res$auc_hi,
      selected = s$selected[[1]]
    )
  }
)

greedy_auc_single <- list(
  summary      = rbind(reused_summary, new_steps$summary),
  learner_fits = rbind(greedy_auc$learner_fits[step %in% 1:2],
                       new_steps$learner_fits)
)

# Expensive; cached and reloaded in the next chunk.
saveRDS(greedy_auc_single, here::here("data", "processed", "fits", "greedy_auc_results_single.rds"))
```

## Plot: combined figure (person-average + single-session)

``` r
greedy_auc_single <- readRDS(here::here("data", "processed", "fits", "greedy_auc_results_single.rds"))

# Backward compatibility: rename alpha -> phi in cached results
if ("alpha" %in% names(greedy_auc_single$learner_fits)) setnames(greedy_auc_single$learner_fits, "alpha", "phi")
if ("param_added" %in% names(greedy_auc_single$learner_fits)) greedy_auc_single$learner_fits[param_added == "alpha", param_added := "phi"]
if ("param_added" %in% names(greedy_auc_single$summary))      greedy_auc_single$summary[param_added == "alpha", param_added := "phi"]
if ("free_params" %in% names(greedy_auc_single$summary)) {
  greedy_auc_single$summary[, free_params := lapply(free_params, function(fp) replace(fp, fp == "alpha", "phi"))]
}

# ── Panel B: single-session selection path ────────────────────────────────────
summary_single_b <- copy(greedy_auc_single$summary)
summary_single_b[, param_label := param_display[param_added]]

selection_single_b <- summary_single_b[selected == TRUE][order(step)]
selection_single_b[, param_label := param_display[param_added]]

seg_single_b <- summary_single_b[selected == FALSE]
seg_single_b[, y_from := selection_single_b$auc_corrected[
  match(step - 1, selection_single_b$step)
]]

p_auc_single <- ggplot() +
  geom_segment(
    data = seg_single_b[!is.na(y_from)],
    aes(x = step - 1, xend = step, y = y_from, yend = auc_corrected,
        color = param_label),
    linewidth = 0.4, linetype = "dashed", alpha = 0.7
  ) +
  geom_line(data = selection_single_b,
            aes(x = step, y = auc_corrected),
            linewidth = 1.2, color = "black") +
  geom_ribbon(data = selection_single_b,
              aes(x = step, ymin = auc_lo, ymax = auc_hi),
              alpha = 0.15, fill = "grey40") +
  geom_point(data = summary_single_b[selected == FALSE],
             aes(x = step, y = auc_corrected, color = param_label),
             size = 3, shape = 16, alpha = 0.6) +
  geom_point(data = selection_single_b,
             aes(x = step, y = auc_corrected, color = param_label),
             size = 5, shape = 17) +
  geom_text(
    data = summary_single_b[selected == FALSE & auc_corrected < 0.6] %>%
      arrange(auc_corrected) %>%
      mutate(y_pos = 0.62 + seq(0, by = 0.008, length.out = n())),
    aes(x = step + 0.15, y = y_pos,
        label = paste0(param_label, " (",
                       scales::percent(auc_corrected, accuracy = 1), ")"),
        color = 'black'),
    size = 4, hjust = 0, show.legend = FALSE
  ) +
  scale_color_manual(name = "Parameter", values = param_cols) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(limits = c(0.6, 1),
                     breaks = seq(0.6, 1, by = 0.05),
                     oob = scales::squish) +
  labs(x = "Step (parameter added)",
       y = "Classification AUC (bootstrap-corrected)") +
  theme_bw(base_size = 12) +
  theme(legend.position = "right", panel.grid.minor = element_blank())

# change paramter order for the legend:
p_auc_single <- p_auc_single +
  scale_color_manual(name = "Parameter", values = param_cols,
                     breaks = names(param_cols))
# ── Combine panels ───────────────────────────────────────────────────────────
p_combined <- (p_auc + ggtitle("Participant average")) +
  (p_auc_single + ggtitle("Single session")) +
  plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 14, face = "bold"))

p_combined
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/06_classification_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
# Paper Figure 10: greedy parameter selection by bootstrap-corrected AUC (A: participant average, B: single session)
ggsave(here::here("output", "greedy_selection_auc_combined.png"), p_combined,
       width = 11, height = 6, dpi = 300)
```

## Summary statistics for paper

Finally, we print the selection order and the AUCs of all candidates at
each step, for both aggregation levels, as reported in Table 4 of the
paper.

``` r
cat("=== PANEL A: Person-average selection ===\n\n")
```

    ## === PANEL A: Person-average selection ===

``` r
cat("Selection order:\n")
```

    ## Selection order:

``` r
greedy_auc$summary[selected == TRUE][order(step),
  .(step, param_added,
    auc_corrected = round(auc_corrected, 3),
    auc_lo = round(auc_lo, 3),
    auc_hi = round(auc_hi, 3))] |> print()
```

    ##     step param_added auc_corrected auc_lo auc_hi
    ##    <int>      <char>         <num>  <num>  <num>
    ## 1:     1           F         0.917  0.917  0.917
    ## 2:     2         phi         0.944  0.934  0.949
    ## 3:     3         tau         0.930  0.893  0.941
    ## 4:     4         ter         0.926  0.867  0.948
    ## 5:     5           s         0.926  0.859  0.958

``` r
cat("\nAll candidates at each step:\n")
```

    ## 
    ## All candidates at each step:

``` r
greedy_auc$summary[order(step, -auc_corrected),
  .(step, param_added, selected,
    auc_corrected = round(auc_corrected, 3),
    auc_lo = round(auc_lo, 3),
    auc_hi = round(auc_hi, 3))] |> print()
```

    ##      step param_added selected auc_corrected auc_lo auc_hi
    ##     <int>      <char>   <lgcl>         <num>  <num>  <num>
    ##  1:     1           F     TRUE         0.917  0.917  0.917
    ##  2:     1         phi    FALSE         0.907  0.907  0.907
    ##  3:     1         ter    FALSE         0.877  0.877  0.877
    ##  4:     1           s    FALSE         0.756  0.756  0.756
    ##  5:     1         tau    FALSE         0.532  0.532  0.532
    ##  6:     2         phi     TRUE         0.944  0.934  0.949
    ##  7:     2         ter    FALSE         0.912  0.889  0.921
    ##  8:     2         tau    FALSE         0.897  0.877  0.904
    ##  9:     2           s    FALSE         0.891  0.860  0.907
    ## 10:     3         tau     TRUE         0.930  0.893  0.941
    ## 11:     3           s    FALSE         0.930  0.889  0.948
    ## 12:     3         ter    FALSE         0.926  0.895  0.937
    ## 13:     4         ter     TRUE         0.926  0.867  0.948
    ## 14:     4           s    FALSE         0.923  0.860  0.944
    ## 15:     5           s     TRUE         0.926  0.859  0.958

``` r
cat("\n=== PANEL B: Single-session selection ===\n\n")
```

    ## 
    ## === PANEL B: Single-session selection ===

``` r
cat("Selection order:\n")
```

    ## Selection order:

``` r
greedy_auc_single$summary[selected == TRUE][order(step),
  .(step, param_added,
    auc_corrected = round(auc_corrected, 3),
    auc_lo = round(auc_lo, 3),
    auc_hi = round(auc_hi, 3))] |> print()
```

    ##     step param_added auc_corrected auc_lo auc_hi
    ##    <int>      <char>         <num>  <num>  <num>
    ## 1:     1           F         0.730  0.730  0.730
    ## 2:     2         phi         0.815  0.814  0.816
    ## 3:     3         ter         0.831  0.829  0.833
    ## 4:     4           s         0.837  0.834  0.838
    ## 5:     5         tau         0.810  0.806  0.813

``` r
cat("\nAll candidates at each step:\n")
```

    ## 
    ## All candidates at each step:

``` r
greedy_auc_single$summary[order(step, -auc_corrected),
  .(step, param_added, selected,
    auc_corrected = round(auc_corrected, 3),
    auc_lo = round(auc_lo, 3),
    auc_hi = round(auc_hi, 3))] |> print()
```

    ##      step param_added selected auc_corrected auc_lo auc_hi
    ##     <int>      <char>   <lgcl>         <num>  <num>  <num>
    ##  1:     1           F     TRUE         0.730  0.730  0.730
    ##  2:     1         phi    FALSE         0.722  0.722  0.722
    ##  3:     1           s    FALSE         0.628  0.628  0.628
    ##  4:     1         ter    FALSE         0.618  0.618  0.618
    ##  5:     1         tau    FALSE         0.556  0.556  0.556
    ##  6:     2         phi     TRUE         0.815  0.814  0.816
    ##  7:     2           s    FALSE         0.755  0.753  0.756
    ##  8:     2         tau    FALSE         0.748  0.746  0.749
    ##  9:     2         ter    FALSE         0.745  0.742  0.746
    ## 10:     3         ter     TRUE         0.831  0.829  0.833
    ## 11:     3           s    FALSE         0.811  0.808  0.812
    ## 12:     3         tau    FALSE         0.802  0.799  0.803
    ## 13:     4           s     TRUE         0.837  0.834  0.838
    ## 14:     4         tau    FALSE         0.815  0.810  0.818
    ## 15:     5         tau     TRUE         0.810  0.806  0.813

# Session info

``` r
sessionInfo()
```

    ## R version 4.5.1 (2025-06-13)
    ## Platform: aarch64-apple-darwin20
    ## Running under: macOS Sequoia 15.2
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRblas.0.dylib 
    ## LAPACK: /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.1
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## time zone: Europe/Amsterdam
    ## tzcode source: internal
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] Rcpp_1.1.1-1.1    patchwork_1.3.2   ggsci_4.0.0       pROC_1.19.0.1    
    ##  [5] furrr_0.3.1       future_1.67.0     lubridate_1.9.4   forcats_1.0.1    
    ##  [9] stringr_1.5.2     dplyr_1.1.4       purrr_1.1.0       readr_2.1.5      
    ## [13] tidyr_1.3.1       tibble_3.3.0      ggplot2_4.0.0     tidyverse_2.0.0  
    ## [17] here_1.0.2        data.table_1.17.8
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] generics_0.1.4     stringi_1.8.7      listenv_0.9.1      hms_1.1.3         
    ##  [5] digest_0.6.37      magrittr_2.0.4     evaluate_1.0.5     grid_4.5.1        
    ##  [9] timechange_0.3.0   RColorBrewer_1.1-3 fastmap_1.2.0      rprojroot_2.1.1   
    ## [13] scales_1.4.0       textshaping_1.0.4  codetools_0.2-20   cli_3.6.5         
    ## [17] rlang_1.1.6        parallelly_1.45.1  withr_3.0.2        yaml_2.3.10       
    ## [21] tools_4.5.1        parallel_4.5.1     tzdb_0.5.0         globals_0.18.0    
    ## [25] vctrs_0.6.5        R6_2.6.1           lifecycle_1.0.4    ragg_1.5.0        
    ## [29] pkgconfig_2.0.3    pillar_1.11.1      gtable_0.3.6       glue_1.8.0        
    ## [33] systemfonts_1.3.1  xfun_0.53          tidyselect_1.2.1   rstudioapi_0.17.1 
    ## [37] knitr_1.50         farver_2.1.2       htmltools_0.5.8.1  rmarkdown_2.30    
    ## [41] compiler_4.5.1     S7_0.2.0
