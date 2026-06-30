Simulation: parameter recovery as a function of available data
================
Thomas Wilschut
Last updated: 2026-06-30

- [Setup](#setup)
- [Simulate a larger dataset for
  recovery](#simulate-a-larger-dataset-for-recovery)
- [Run recovery across grid](#run-recovery-across-grid)
- [Load results (if saved earlier)](#load-results-if-saved-earlier)
- [Merge true parameters](#merge-true-parameters)
- [Recovery (Figure 5)](#recovery-figure-5)
- [Fact-level Δφ recovery (Figure 6)](#fact-level-δφ-recovery-figure-6)
- [Correlations of the true
  parameters](#correlations-of-the-true-parameters)

This markdown file is can be used to reproduce the results reported in
the paper: Idiographic Memory Modelling in ACT-R using Alternating
Maximum Likelihood Estimation

This script is used for the recovery analysis, reported in the section:
Simulation: Recovering parameters from synthetic data.

# Setup

``` r
library(data.table)
library(here)
library(ggplot2)
library(furrr)
library(patchwork)
library(ggsci)
library(Rcpp)
library(GGally)
library(tidyverse)

# Parallel backend
plan(multisession, workers = availableCores() - 1)

# Source simulation + fitting functions (which also compile actr_ll.cpp)
source(here::here("R", "sim-mle.R"))

# Plot aesthetics
col_blue <- "#0571b0"
col_red  <- "#ca0020"
plottheme <- theme_bw(base_size = 12) 
```

# Simulate a larger dataset for recovery

For the recovery analysis we generate a single large dataset (50
repetitions per fact) once, and then subset it down to smaller numbers
of repetitions and facts in the grid below. Generating the maximum
amount of data up front and subsetting it keeps the underlying learners
and facts identical across all data-quantity conditions, so differences
in recovery reflect the amount of data rather than a different random
draw.

``` r
sim <- simulate_fact_learner_data(
  n_learners    = 50,
  n_facts       = 15,
  n_repetitions = 50,   # generate plenty; we'll subset later
  seed          = 999
)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |=                                                                     |   2%  |                                                                              |===                                                                   |   4%  |                                                                              |====                                                                  |   6%  |                                                                              |======                                                                |   8%  |                                                                              |=======                                                               |  10%  |                                                                              |========                                                              |  12%  |                                                                              |==========                                                            |  14%  |                                                                              |===========                                                           |  16%  |                                                                              |=============                                                         |  18%  |                                                                              |==============                                                        |  20%  |                                                                              |===============                                                       |  22%  |                                                                              |=================                                                     |  24%  |                                                                              |==================                                                    |  26%  |                                                                              |====================                                                  |  28%  |                                                                              |=====================                                                 |  30%  |                                                                              |======================                                                |  32%  |                                                                              |========================                                              |  34%  |                                                                              |=========================                                             |  36%  |                                                                              |===========================                                           |  38%  |                                                                              |============================                                          |  40%  |                                                                              |=============================                                         |  42%  |                                                                              |===============================                                       |  44%  |                                                                              |================================                                      |  46%  |                                                                              |==================================                                    |  48%  |                                                                              |===================================                                   |  50%  |                                                                              |====================================                                  |  52%  |                                                                              |======================================                                |  54%  |                                                                              |=======================================                               |  56%  |                                                                              |=========================================                             |  58%  |                                                                              |==========================================                            |  60%  |                                                                              |===========================================                           |  62%  |                                                                              |=============================================                         |  64%  |                                                                              |==============================================                        |  66%  |                                                                              |================================================                      |  68%  |                                                                              |=================================================                     |  70%  |                                                                              |==================================================                    |  72%  |                                                                              |====================================================                  |  74%  |                                                                              |=====================================================                 |  76%  |                                                                              |=======================================================               |  78%  |                                                                              |========================================================              |  80%  |                                                                              |=========================================================             |  82%  |                                                                              |===========================================================           |  84%  |                                                                              |============================================================          |  86%  |                                                                              |==============================================================        |  88%  |                                                                              |===============================================================       |  90%  |                                                                              |================================================================      |  92%  |                                                                              |==================================================================    |  94%  |                                                                              |===================================================================   |  96%  |                                                                              |===================================================================== |  98%  |                                                                              |======================================================================| 100%

``` r
sim_test <- sim[is_study == FALSE]
```

# Run recovery across grid

We vary the amount of available data along two axes: the number of
repetitions per fact (holding facts fixed at 15) and the number of facts
per learner (holding repetitions fixed at 10). For each grid point we
refit all 50 learners and record the recovered parameters. As with the
greedy step, this is expensive, so the chunk is `eval = FALSE` and the
results are cached to `data/processed/fits/`; the next chunk reloads
them.

``` r
rep_levels  <- c(1, 2, 4, 6, 8, 10, 15, 20, 30)
fact_levels <- c(1, 2, 4, 6, 8, 10, 15, 20, 30)

fits_vary_reps <- 
  map_dfr(rep_levels, function(nr) {
    cat(sprintf("\n=== Fitting n_reps = %d ===\n", nr))
    res <- run_recovery_grid(sim, n_reps = nr, n_facts = 15)
    res$n_repetitions <- nr
    res$n_facts_grid  <- 15
    res
  })

fits_vary_facts <- 
  map_dfr(fact_levels, function(nf) {
    cat(sprintf("\n=== Fitting n_facts = %d ===\n", nf))
    res <- run_recovery_grid(sim, n_reps = 10, n_facts = nf)
    res$n_repetitions <- 10
    res$n_facts_grid  <- nf
    res
  })

saveRDS(fits_vary_reps,  here::here("data", "processed", "fits", "results_recovery_vary_reps.rds"))
saveRDS(fits_vary_facts, here::here("data", "processed", "fits", "results_recovery_vary_facts.rds"))
```

# Load results (if saved earlier)

``` r
fits_vary_reps  <- readRDS(here::here("data", "processed", "fits", "results_recovery_vary_reps.rds"))
fits_vary_facts <- readRDS(here::here("data", "processed", "fits", "results_recovery_vary_facts.rds"))

# Backward compatibility: rename old column names from pre-phi cached fits
if ("alpha"           %in% names(fits_vary_reps))  setnames(fits_vary_reps,  "alpha",           "phi")
if ("delta_alpha_est" %in% names(fits_vary_reps))  setnames(fits_vary_reps,  "delta_alpha_est", "delta_phi_est")
if ("alpha"           %in% names(fits_vary_facts)) setnames(fits_vary_facts, "alpha",           "phi")
if ("delta_alpha_est" %in% names(fits_vary_facts)) setnames(fits_vary_facts, "delta_alpha_est", "delta_phi_est")

names(fits_vary_reps)
```

    ##  [1] "n_repetitions" "n_facts_grid"  "user_id"       "iteration"    
    ##  [5] "phi"           "tau"           "s"             "lf"           
    ##  [9] "ter"           "ll"            "n_iter"        "delta_phi_est"

# Merge true parameters

To assess recovery we need the ground-truth parameter values alongside
the recovered estimates. Here we extract the true person-level
parameters (one row per learner) and join them onto the recovered fits
from both grids.

``` r
true_params <- sim_test[, .(
  true_phi = true_phi[1],
  true_tau = true_tau[1],
  true_s   = true_s[1],
  true_F   = true_lf[1],
  true_ter = true_ter[1]
), by = user_id]

recovery_reps  <- merge(fits_vary_reps,  true_params, by = "user_id")
recovery_facts <- merge(fits_vary_facts, true_params, by = "user_id")
```

# Recovery (Figure 5)

This builds the combined recovery figure (Figure 5 in the paper). Panel
A shows true-vs-recovered scatter plots at the reference data quantity
(10 repetitions, 15 facts); panels B–E show how recovery (correlation)
and mean absolute error change as a function of repetitions per fact and
number of facts, with the reference condition highlighted; panel F shows
the pairwise correlations between the recovered estimates, which reveal
the trade-offs between parameters with partly overlapping behavioural
effects. The helper functions `make_cor_summary`/`make_error_summary`
and `plot_cor`/`plot_error` are defined here to avoid repeating the same
summarising and plotting code for each panel.

``` r
# ── Shared aesthetics (consistent with other plots) ───────────────────────────
param_levels <- c(
  "Speed of Forgetting", "Retrieval threshold", "Activation noise",
  "Latency factor", "Non-retrieval time"
)

param_colors    <- setNames(pal_aaas()(5), param_levels)
param_shapes    <- setNames(c(16, 17, 15, 18, 8),                              param_levels)
param_linetypes <- setNames(c("solid", "dashed", "dotdash", "longdash", "twodash"), param_levels)

# ── Panel A: Scatter plots ────────────────────────────────────────────────────
subset_10 <- recovery_reps[n_repetitions == 10]

plot_df <- subset_10 %>%
  pivot_longer(cols = c(phi, tau, s, lf, ter),
               names_to = "param", values_to = "estimate") %>%
  mutate(
    true_value = case_when(
      param == "phi" ~ true_phi,
      param == "tau" ~ true_tau,
      param == "s"   ~ true_s,
      param == "lf"  ~ true_F,
      param == "ter" ~ true_ter
    ),
    param_label = recode(param,
      "phi" = "Speed of Forgetting",
      "tau" = "Retrieval threshold",
      "s"   = "Activation noise",
      "lf"  = "Latency factor",
      "ter" = "Non-retrieval time"
    ),
    param_label = factor(param_label, levels = param_levels)
  )

learner_corrs <- plot_df %>%
  group_by(param_label) %>%
  summarise(r = cor(estimate, true_value, use = "complete.obs"), .groups = "drop") %>%
  mutate(r_lab = paste0("r = ", format(round(r, 2), nsmall = 2)))

p_recovery <- ggplot(plot_df, aes(x = true_value, y = estimate, color = param_label)) +
  geom_point(alpha = 0.85, size = 2) +
  geom_smooth(method = "lm", se = TRUE, colour = "black") +
  geom_text(data = learner_corrs,
            aes(x = -Inf, y = Inf, label = r_lab),
            inherit.aes = FALSE, hjust = -0.1, vjust = 2, size = 4) +
  facet_wrap(~ param_label, scales = "free", ncol = 5) +
  coord_cartesian(clip = "off") +
  scale_color_manual(values = param_colors) +
  plottheme +
  theme(
    legend.position  = "none",
    strip.background = element_rect(fill = "grey85", colour = "grey60"),
    strip.text       = element_text(size = 9)
  ) +
  labs(x = "Original value", y = "Recovered value")

# ── Helper functions ──────────────────────────────────────────────────────────
make_cor_summary <- function(df, group_var) {
  df %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      cor_phi = cor(true_phi, phi, use = "complete.obs"),
      cor_tau = cor(true_tau, tau, use = "complete.obs"),
      cor_s   = cor(true_s,   s,   use = "complete.obs"),
      cor_F   = cor(true_F,   lf,  use = "complete.obs"),
      cor_ter = cor(true_ter, ter, use = "complete.obs"),
      .groups = "drop"
    ) %>%
    pivot_longer(starts_with("cor_"), names_to = "parameter", values_to = "correlation") %>%
    mutate(
      parameter   = sub("cor_", "", parameter),
      param_label = recode(parameter,
        "phi" = "Speed of Forgetting",
        "tau" = "Retrieval threshold",
        "s"   = "Activation noise",
        "F"   = "Latency factor",
        "ter" = "Non-retrieval time"
      ),
      param_label = factor(param_label, levels = param_levels)
    )
}

make_error_summary <- function(df, group_var) {
  df %>%
    mutate(
      phi_error = abs(true_phi - phi),
      tau_error = abs(true_tau - tau),
      s_error   = abs(true_s   - s),
      F_error   = abs(true_F   - lf),
      ter_error = abs(true_ter - ter)
    ) %>%
    pivot_longer(ends_with("_error"), names_to = "parameter", values_to = "abs_error") %>%
    group_by(parameter, !!sym(group_var)) %>%
    summarise(mean_error = mean(abs_error, na.rm = TRUE),
              se_error   = sd(abs_error,   na.rm = TRUE) / sqrt(n()),
              .groups = "drop") %>%
    mutate(
      param_label = recode(parameter,
        "phi_error" = "Speed of Forgetting",
        "tau_error" = "Retrieval threshold",
        "s_error"   = "Activation noise",
        "F_error"   = "Latency factor",
        "ter_error" = "Non-retrieval time"
      ),
      param_label = factor(param_label, levels = param_levels)
    )
}

# ── Shared scale layers ───────────────────────────────────────────────────────
scales_shared <- list(
  scale_color_manual(values = param_colors),
  scale_linetype_manual(values = param_linetypes),
  scale_shape_manual(values = param_shapes),
  guides(colour = guide_legend(override.aes = list(
    linetype = param_linetypes, shape = param_shapes)),
    linetype = "none", shape = "none")
)

plot_cor <- function(df, x_var, x_lab, highlight_x = NULL) {
  p <- ggplot(df, aes(x = factor(!!sym(x_var)), y = correlation,
                      colour = param_label, linetype = param_label,
                      shape = param_label, group = param_label))
  if (!is.null(highlight_x)) {
    p <- p + annotate("rect",
      xmin = which(levels(factor(df[[x_var]])) == as.character(highlight_x)) - 0.4,
      xmax = which(levels(factor(df[[x_var]])) == as.character(highlight_x)) + 0.4,
      ymin = -Inf, ymax = Inf, fill = "grey80", alpha = 0.4)
  }
  p + geom_line(linewidth = 1) + geom_point(size = 2) +
    scale_y_continuous(limits = c(0, 1)) +
    scales_shared +
    plottheme +
    theme(legend.position = "none") +
    labs(x = x_lab, y = "Recovery (Preason's r)")
}

plot_error <- function(df, x_var, x_lab, highlight_x = NULL) {
  p <- ggplot(df, aes(x = factor(!!sym(x_var)), y = mean_error,
                      colour = param_label, linetype = param_label,
                      shape = param_label, group = param_label))
  if (!is.null(highlight_x)) {
    p <- p + annotate("rect",
      xmin = which(levels(factor(df[[x_var]])) == as.character(highlight_x)) - 0.4,
      xmax = which(levels(factor(df[[x_var]])) == as.character(highlight_x)) + 0.4,
      ymin = -Inf, ymax = Inf, fill = "grey80", alpha = 0.4)
  }
  p + geom_line(linewidth = 1) + geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
                  width = 0.3, alpha = 0.6) +
    scales_shared +
    plottheme +
    theme(legend.position = "none") +
    labs(x = x_lab, y = "Mean absolute error")
}

# ── Panels B–E ────────────────────────────────────────────────────────────────
cor_reps   <- make_cor_summary(recovery_reps, "n_repetitions")
error_reps <- make_error_summary(recovery_reps, "n_repetitions")
cor_facts  <- make_cor_summary(recovery_facts, "n_facts_grid")
error_facts <- make_error_summary(recovery_facts, "n_facts_grid")

p_corr_reps  <- plot_cor(cor_reps  %>% filter(n_repetitions < 20), "n_repetitions", "Repetitions per fact (15 facts)", highlight_x = 10)
p_error_reps <- plot_error(error_reps %>% filter(n_repetitions < 20), "n_repetitions", "Repetitions per fact (15 facts)", highlight_x = 10)
p_corr_facts  <- plot_cor(cor_facts,  "n_facts_grid", "N facts (10 repetitions)", highlight_x = 15)
p_error_facts <- plot_error(error_facts, "n_facts_grid", "N facts (10 repetitions)", highlight_x = 15)

# ── Panel F: Pairwise correlations ────────────────────────────────────────────
pairs_df <- subset_10[, .(phi, tau, s, lf, ter)]


pairs_plot <- ggpairs(
  as.data.frame(pairs_df),
  upper = list(continuous = function(data, mapping, ...) {
    ggally_cor(data, mapping, ...)
  }),
  lower = list(continuous = function(data, mapping, ...) {
    ggplot(data, mapping) +
      geom_point(alpha = 0.3) +
      geom_smooth(method = "lm", se = TRUE, color = "black")
  }),
  columnLabels = c("Speed of\nForgetting", "Retrieval\nthreshold",
                 "Activation\nnoise", "Latency\nfactor", "Non-retrieval\ntime")
) +
  plottheme +
  theme(strip.background = element_rect(fill = "grey85", colour = "grey60"),
        strip.text       = element_text(size = 8),
        axis.text.x      = element_text(angle = 90, hjust = 1))


# ── Combined figure ───────────────────────────────────────────────────────────
pairs_grob <- GGally::ggmatrix_gtable(pairs_plot)
```

    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'

``` r
p_corr_reps <- p_corr_reps + theme(legend.position = "none")
p_error_reps <- p_error_reps + theme(legend.position = "none")
p_corr_facts <- p_corr_facts + theme(legend.position = "none")
p_error_facts <- p_error_facts + theme(legend.position = "bottom", # remeove legend title:
                                       legend.title = element_blank()
                                       )

combined <-
  ((p_corr_reps + p_error_reps) / (p_corr_facts + p_error_facts) |
     wrap_elements(pairs_grob)) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))

combined
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/03_parameter_recovery_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# Paper Figure 5: recovery of participant-level parameters as a function of available data
ggsave(here::here("output", "recovery_combined.png"), combined, width = 11, height = 6.5)
```

# Fact-level Δφ recovery (Figure 6)

Besides the person-level parameters, the model estimates a Speed of
Forgetting offset (Δφ) for each individual fact. Here we assess how well
those fact-level offsets are recovered (Figure 6 in the paper): panel A
is a true-vs-recovered scatter at 10 repetitions, and panels B and C
show recovery (correlation) and mean absolute error as a function of the
number of repetitions per fact. The recovered offsets are stored as a
named list-column in the fits, so `get_delta_phi()` unpacks them into a
long table keyed by fact before merging on the true values.

``` r
# Extract true fact offsets
true_offsets <- sim_test[, .(true_delta_phi = true_fact_offset[1]),
                         by = .(user_id, fact_id)]

# Extract recovered offsets from fits
get_delta_phi <- function(fits_df) {
  fits_df[, {
    da <- delta_phi_est[[1]]
    if (is.null(da) || length(da) == 0) {
      NULL
    } else {
      fact_ids <- as.integer(sub("^d_", "", names(da)))
      .(fact_id = fact_ids,
        delta_phi_est = as.numeric(da))
    }
  }, by = .(user_id, n_repetitions)]
}

fact_recovery <- get_delta_phi(fits_vary_reps) %>% filter(n_repetitions < 20)
fact_recovery <- merge(fact_recovery, true_offsets, by = c("user_id", "fact_id"))

# (A) Scatter at 10 reps
fact_10 <- fact_recovery[n_repetitions == 10]
r_val <- cor(fact_10$true_delta_phi, fact_10$delta_phi_est, use = "complete.obs")

p_fact_scatter <- ggplot(fact_10, aes(x = true_delta_phi, y = delta_phi_est)) +
  geom_point(alpha = 0.7, color = pal_aaas()(1)) +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("r = %.2f", r_val),
           hjust = -0.1, vjust = 2, size = 4) +
  coord_cartesian(clip = "off") +
  labs(x = expression("True "*Delta*phi),
       y = expression("Recovered "*Delta*phi)) +
  plottheme

# (B) Correlation over reps
fact_corr <- fact_recovery[, .(corr = cor(true_delta_phi, delta_phi_est, use = "complete.obs")),
                           by = n_repetitions]

p_fact_corr <- ggplot(fact_corr, aes(x = factor(n_repetitions), y = corr, group = 1)) +
  geom_line(linewidth = 1, color = pal_aaas()(1)) +
  geom_point(size = 2, color = pal_aaas()(1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Repetitions per fact", y = "Recovery (Preason's r)") +
  plottheme 

# (C) Error over reps
fact_error <- fact_recovery[, .(
  mean_error = mean(abs(true_delta_phi - delta_phi_est), na.rm = TRUE),
  se_error = sd(abs(true_delta_phi - delta_phi_est), na.rm = TRUE) / sqrt(.N)
), by = n_repetitions]

p_fact_error <- ggplot(fact_error, aes(x = factor(n_repetitions), y = mean_error, group = 1)) +
  geom_line(linewidth = 1, color = pal_aaas()(1)) +
  geom_point(size = 2, color = pal_aaas()(1)) +
  geom_errorbar(aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
                width = 0.3, alpha = 0.6, color = pal_aaas()(1)) +
  labs(x = "Repetitions per fact", y = "Mean absolute error") +
  plottheme

(p_fact_scatter + p_fact_corr + p_fact_error) +
  plot_annotation(tag_levels = "A") &
  theme(
    plot.tag = element_text(face = "bold", size = 14)
  )
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/03_parameter_recovery_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# Paper Figure 6: recovery of fact-level Speed of Forgetting offsets (delta-phi)
ggsave(here::here("output", "fact_level_recovery.png"), width = 10, height = 3.5)
```

    ## `geom_smooth()` using formula = 'y ~ x'

# Correlations of the true parameters

``` r
true_param_pairs <- sim_test[, .(true_phi, true_tau, true_s, true_lf, true_ter)]
pairs_true <- ggpairs(
  as.data.frame(true_param_pairs),
  upper = list(continuous = function(data, mapping, ...) {
    ggally_cor(data, mapping, ...)
  }),
  lower = list(continuous = function(data, mapping, ...) {
    ggplot(data, mapping) +
      geom_point(alpha = 0.3) +
      geom_smooth(method = "lm", se = TRUE, color = "black")
  }),
  columnLabels = c("Speed of\nForgetting", "Retrieval\nthreshold",
                 "Activation\nnoise", "Latency\nfactor", "Non-retrieval\ntime")
) +
  plottheme +
  theme(strip.background = element_rect(fill = "grey85", colour = "grey60"),
        strip.text       = element_text(size = 8),
        axis.text.x      = element_text(angle = 90, hjust = 1))
pairs_true
```

    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/03_parameter_recovery_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->
\# Session info

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
    ##  [1] lubridate_1.9.4   forcats_1.0.1     stringr_1.5.2     dplyr_1.1.4      
    ##  [5] purrr_1.1.0       readr_2.1.5       tidyr_1.3.1       tibble_3.3.0     
    ##  [9] tidyverse_2.0.0   GGally_2.4.0      Rcpp_1.1.1-1.1    ggsci_4.0.0      
    ## [13] patchwork_1.3.2   furrr_0.3.1       future_1.67.0     ggplot2_4.0.0    
    ## [17] here_1.0.2        data.table_1.17.8
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] generics_0.1.4     lattice_0.22-7     stringi_1.8.7      listenv_0.9.1     
    ##  [5] hms_1.1.3          digest_0.6.37      magrittr_2.0.4     timechange_0.3.0  
    ##  [9] evaluate_1.0.5     grid_4.5.1         RColorBrewer_1.1-3 fastmap_1.2.0     
    ## [13] Matrix_1.7-3       rprojroot_2.1.1    mgcv_1.9-3         scales_1.4.0      
    ## [17] textshaping_1.0.4  codetools_0.2-20   cli_3.6.5          rlang_1.1.6       
    ## [21] parallelly_1.45.1  splines_4.5.1      withr_3.0.2        yaml_2.3.10       
    ## [25] tools_4.5.1        parallel_4.5.1     tzdb_0.5.0         globals_0.18.0    
    ## [29] ggstats_0.11.0     vctrs_0.6.5        R6_2.6.1           lifecycle_1.0.4   
    ## [33] ragg_1.5.0         pkgconfig_2.0.3    pillar_1.11.1      gtable_0.3.6      
    ## [37] glue_1.8.0         systemfonts_1.3.1  xfun_0.53          tidyselect_1.2.1  
    ## [41] rstudioapi_0.17.1  knitr_1.50         farver_2.1.2       nlme_3.1-168      
    ## [45] htmltools_0.5.8.1  labeling_0.4.3     rmarkdown_2.30     compiler_4.5.1    
    ## [49] S7_0.2.0
