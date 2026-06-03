Simulation: greedy forward parameter selection
================
Thomas Wilschut
Last updated: 2026-06-03

- [Setup](#setup)
- [Simulate response data](#simulate-response-data)
- [Greedy forward parameter selection (C++
  likelihood)](#greedy-forward-parameter-selection-c-likelihood)
- [Greedy fits plot (Figure 4)](#greedy-fits-plot-figure-4)
- [AIC plot (Figure 3)](#aic-plot-figure-3)
- [Session info](#session-info)

This markdown file is can be used to reproduce the results reported in
the paper: Idiographic Memory Modelling in ACT-R using Alternating
Maximum Likelihood Estimation

This script is used for the greedy selection procedure, reported in the
section: Simulation: Recovering parameters from synthetic data.

# Setup

``` r
library(data.table)
library(here)
library(ggplot2)
library(furrr)
library(patchwork)
library(ggsci)
library(Rcpp)
library(ggh4x)
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

# Simulate response data

We generate a single synthetic dataset (50 learners, 15 facts, 10
repetitions per fact) with a fixed seed. The same `sim` object is reused
throughout this notebook, both as the input to the greedy selection and
as the source of the ground-truth parameter values for the recovery
plot.

``` r
sim <- simulate_fact_learner_data(
  n_learners    = 50,
  n_facts       = 15,
  n_repetitions = 10,
  seed          = 999
)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |=                                                                     |   2%  |                                                                              |===                                                                   |   4%  |                                                                              |====                                                                  |   6%  |                                                                              |======                                                                |   8%  |                                                                              |=======                                                               |  10%  |                                                                              |========                                                              |  12%  |                                                                              |==========                                                            |  14%  |                                                                              |===========                                                           |  16%  |                                                                              |=============                                                         |  18%  |                                                                              |==============                                                        |  20%  |                                                                              |===============                                                       |  22%  |                                                                              |=================                                                     |  24%  |                                                                              |==================                                                    |  26%  |                                                                              |====================                                                  |  28%  |                                                                              |=====================                                                 |  30%  |                                                                              |======================                                                |  32%  |                                                                              |========================                                              |  34%  |                                                                              |=========================                                             |  36%  |                                                                              |===========================                                           |  38%  |                                                                              |============================                                          |  40%  |                                                                              |=============================                                         |  42%  |                                                                              |===============================                                       |  44%  |                                                                              |================================                                      |  46%  |                                                                              |==================================                                    |  48%  |                                                                              |===================================                                   |  50%  |                                                                              |====================================                                  |  52%  |                                                                              |======================================                                |  54%  |                                                                              |=======================================                               |  56%  |                                                                              |=========================================                             |  58%  |                                                                              |==========================================                            |  60%  |                                                                              |===========================================                           |  62%  |                                                                              |=============================================                         |  64%  |                                                                              |==============================================                        |  66%  |                                                                              |================================================                      |  68%  |                                                                              |=================================================                     |  70%  |                                                                              |==================================================                    |  72%  |                                                                              |====================================================                  |  74%  |                                                                              |=====================================================                 |  76%  |                                                                              |=======================================================               |  78%  |                                                                              |========================================================              |  80%  |                                                                              |=========================================================             |  82%  |                                                                              |===========================================================           |  84%  |                                                                              |============================================================          |  86%  |                                                                              |==============================================================        |  88%  |                                                                              |===============================================================       |  90%  |                                                                              |================================================================      |  92%  |                                                                              |==================================================================    |  94%  |                                                                              |===================================================================   |  96%  |                                                                              |===================================================================== |  98%  |                                                                              |======================================================================| 100%

# Greedy forward parameter selection (C++ likelihood)

The greedy procedure starts with no free participant-level parameters
(only fact-level offsets Δα are estimated). At each step it adds the
candidate parameter that produces the largest AIC improvement, fitting
all learners in parallel using the compiled C++ likelihood from
`actr_ll.cpp`.

``` r
greedy_results <- greedy_select_params(
  sim_full   = sim,
  n_reps     = 10,
  n_facts    = 15,
  all_params = c("alpha", "tau", "s", "F", "ter")
)

# This step is expensive (it refits every learner at every candidate parameter),
# so the chunk is set to eval = FALSE; the result is cached and reloaded below.
# Set eval = TRUE (or delete the cached .rds) to rerun it from scratch.
saveRDS(greedy_results, here::here("data", "processed", "fits", "results_greedy_selection.rds"))
```

# Greedy fits plot (Figure 4)

We reload the cached greedy-selection result and build the recovery grid
(Figure 4 in the paper): one row per selection step, one column per
parameter. For every panel we plot the recovered value against the true
value, colouring the parameter that is freely estimated at that step
(selected) versus the ones still held at their defaults (not selected),
and annotate each panel with the true-vs-recovered correlation. This
shows how recovery of each parameter improves as more parameters are
freed.

``` r
greedy_results <- readRDS(here::here("data", "processed", "fits", "results_greedy_selection.rds"))

greedy_fits    <- greedy_results$learner_fits
greedy_summary <- greedy_results$summary

true_params <- sim[is_study == FALSE, .(
  true_alpha = true_alpha[1],
  true_tau   = true_tau[1],
  true_s     = true_s[1],
  true_F     = true_lf[1],
  true_ter   = true_ter[1]
), by = user_id]

greedy_fits_m <- merge(as.data.frame(greedy_fits), as.data.frame(true_params), by = "user_id")

selected_models <- as.data.frame(greedy_summary[selected == TRUE][order(step)])
selection_order <- selected_models$param_added

param_info <- data.frame(
  free_name = c("alpha", "tau", "s", "F", "ter"),
  fit_col   = c("alpha", "tau", "s", "lf", "ter"),
  true_col  = c("true_alpha", "true_tau", "true_s", "true_F", "true_ter"),
  label     = c("Speed of Forgetting", "Retrieval threshold",
                "Activation noise", "Latency factor", "Non-retrieval time"),
  stringsAsFactors = FALSE
)

# Full-name step labels — defined before the loop so plot_df uses them
name_lookup <- setNames(param_info$label, param_info$free_name)
slabels     <- paste0("Step ", selected_models$step, ": + ", name_lookup[selected_models$param_added])

col_order <- param_info$label[match(selection_order, param_info$free_name)]

# Build plot data
plot_rows <- list()
for (s in seq_along(slabels)) {
  sel_cand     <- selected_models$param_added[s]
  free_at_step <- selection_order[1:s]
  step_fits    <- greedy_fits_m[greedy_fits_m$step == s, ]
  candidates   <- unique(step_fits$param_added)

  for (i in seq_len(nrow(param_info))) {
    pname <- param_info$free_name[i]

    if (pname %in% free_at_step) {
      source_fits <- step_fits[step_fits$param_added == sel_cand, ]
      status <- "selected"
    } else if (pname %in% candidates) {
      source_fits <- step_fits[step_fits$param_added == pname, ]
      status <- "candidate"
    } else {
      next
    }

    plot_rows[[length(plot_rows) + 1]] <- data.frame(
      step       = s,
      step_label = slabels[s],
      param      = param_info$label[i],
      estimate   = source_fits[[param_info$fit_col[i]]],
      true_value = source_fits[[param_info$true_col[i]]],
      status     = status,
      user_id    = source_fits$user_id,
      stringsAsFactors = FALSE
    )
  }
}

plot_df <- do.call(rbind, plot_rows)

# Filter outliers
plot_df <- plot_df[plot_df$estimate <= 3, ]

# Compute correlations
corr_labels <- plot_df %>%
  group_by(param, step_label, status) %>%
  summarise(r = cor(estimate, true_value, use = "complete.obs"), .groups = "drop") %>%
  mutate(r_lab = paste0("r = ", format(round(r, 2), nsmall = 2)))

# Factor ordering
plot_df$step_label     <- factor(plot_df$step_label, levels = slabels)
plot_df$param          <- factor(plot_df$param, levels = col_order)
corr_labels$step_label <- factor(corr_labels$step_label, levels = slabels)
corr_labels$param      <- factor(corr_labels$param, levels = col_order)

# Define per-row strip styling: blank for step row, grey for param row
strip <- strip_themed(
  background_x = c(
    rep(list(element_blank()), 25),                                 # one per panel for step row
    rep(list(element_rect(fill = "grey85", colour = "grey60")), 25) # one per panel for param row
  )
)

p_grid <- ggplot(plot_df, aes(x = true_value, y = estimate)) +
  geom_point(aes(color = status), alpha = 0.6, size = 1.5) +
  scale_color_manual(
    values = c("selected" = "#0571b0", "candidate" = "grey60"),
    labels = c("selected" = "Selected", "candidate" = "Not selected"),
    name = ""
  ) +
  geom_smooth(method = "lm", se = TRUE, aes(linetype = status),
              color = "black", linewidth = 0.6, show.legend = FALSE) +
  scale_linetype_manual(values = c("selected" = "solid", "candidate" = "dashed")) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "grey40") +
  geom_text(data = corr_labels,
            aes(x = -Inf, y = Inf, label = r_lab, color = status),
            inherit.aes = FALSE, hjust = -0.1, vjust = 1.5, size = 3.5,
            show.legend = FALSE) +
  facet_wrap2(step_label ~ param, scales = "free", ncol = 5, strip = strip) +
  labs(x = "True value", y = "Recovered value") +
  plottheme +
  theme(
    strip.text      = element_text(size = 8),
    panel.spacing   = unit(0.3, "lines"),
    legend.position = "bottom"
  )

p_grid
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](/Users/thomaswilschut/Desktop/idiographic-memory-modeling-actr-amle/output/02_greedy_parameter_selection_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# Paper Figure 4: parameter recovery at each step of the greedy forward selection
ggsave(here::here("output", "greedy_recovery_grid.png"), p_grid, width = 10, height = 10)
```

    ## `geom_smooth()` using formula = 'y ~ x'

# AIC plot (Figure 3)

This reproduces Figure 3: the AIC at each step of the greedy forward
selection. The solid line traces the selected path (the lowest-AIC
candidate at each step), while the faint points show the alternative
candidates that were not selected. A signed square-root transform is
applied to the y-axis so that both the very large step-1 values and the
small differences at later steps remain legible in a single panel.

``` r
greedy_summary <- greedy_results$summary

param_display <- c(
  alpha = "Speed of Forgetting",
  tau   = "Retrieval threshold",
  s     = "Activation noise",
  F     = "Latency factor",
  ter   = "Non-retrieval time"
)

greedy_summary[, param_label := param_display[param_added]]
selection_path <- greedy_summary[selected == TRUE][order(step)]
selection_path[, param_label := param_display[param_added]]

# Match aaas colour order to the other plots
param_cols <- setNames(
  pal_aaas()(5),
  c("Speed of Forgetting", "Retrieval threshold", "Activation noise",
    "Latency factor", "Non-retrieval time")
)

seg_data <- greedy_summary[selected == FALSE]
seg_data[, y_from := selection_path$total_aic[match(step - 1, selection_path$step)]]

aic_vals <- greedy_summary$total_aic
sqrt_trans <- scales::trans_new(
  name      = "signed_sqrt",
  transform = function(x) sign(x) * sqrt(abs(x)),
  inverse   = function(x) sign(x) * x^2
)
aic_breaks <- sort(unique(c(
  round(min(aic_vals) / 500) * 500,
  seq(0, max(aic_vals), by = 2000),
  round(max(aic_vals) / 500) * 500
)))

p_aic <- ggplot() +
  geom_segment(data = seg_data[!is.na(y_from)],
               aes(x = step - 1, xend = step, y = y_from, yend = total_aic),
               linewidth = 0.4, color = "grey70", linetype = "dashed") +
  geom_line(data = selection_path,
            aes(x = step, y = total_aic),
            linewidth = 1.2, color = "black") +
  geom_point(data = greedy_summary[selected == FALSE],
             aes(x = step, y = total_aic, color = param_label),
             size = 3, shape = 16, alpha = 0.5) +
  geom_point(data = selection_path,
             aes(x = step, y = total_aic, color = param_label),
             size = 5, shape = 17) +
  scale_color_manual(name = "Parameter", values = param_cols) +
  scale_y_continuous(trans = sqrt_trans, breaks = aic_breaks) +
  scale_x_continuous(breaks = 1:5) +
  labs(x = "Step (parameter added)", y = "AIC") +
  theme_bw() +
  theme(legend.position = "right")

p_aic
```

![](/Users/thomaswilschut/Desktop/idiographic-memory-modeling-actr-amle/output/02_greedy_parameter_selection_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# Paper Figure 3: greedy forward parameter selection based on AIC
ggsave(here::here("output", "greedy_selection_aic.png"), p_aic, width = 10, height = 6)
```

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
    ##  [1] lubridate_1.9.4   forcats_1.0.1     stringr_1.5.2     dplyr_1.1.4      
    ##  [5] purrr_1.1.0       readr_2.1.5       tidyr_1.3.1       tibble_3.3.0     
    ##  [9] tidyverse_2.0.0   ggh4x_0.3.1       Rcpp_1.1.1-1.1    ggsci_4.0.0      
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
    ## [29] vctrs_0.6.5        R6_2.6.1           lifecycle_1.0.4    ragg_1.5.0        
    ## [33] pkgconfig_2.0.3    pillar_1.11.1      gtable_0.3.6       glue_1.8.0        
    ## [37] systemfonts_1.3.1  xfun_0.53          tidyselect_1.2.1   rstudioapi_0.17.1 
    ## [41] knitr_1.50         farver_2.1.2       nlme_3.1-168       htmltools_0.5.8.1 
    ## [45] labeling_0.4.3     rmarkdown_2.30     compiler_4.5.1     S7_0.2.0
