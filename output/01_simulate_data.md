Simulation: generating synthetic retrieval-practice data
================
Thomas Wilschut
Last updated: 2026-06-12

- [Setup](#setup)
- [Simulate data](#simulate-data)
- [Checks](#checks)
  - [Parameter distributions and their relationship to behavior (Figure
    2)](#parameter-distributions-and-their-relationship-to-behavior-figure-2)
  - [Check 2: Observed accuracy vs. predicted
    P(correct)](#check-2-observed-accuracy-vs-predicted-pcorrect)
  - [Check 3: Learning curves — accuracy and RT over
    repetitions](#check-3-learning-curves--accuracy-and-rt-over-repetitions)
  - [Check 4: Effect of fact-level offsets (Δα) on accuracy and
    RT](#check-4-effect-of-fact-level-offsets-δα-on-accuracy-and-rt)
  - [Check 5: Correct vs. incorrect RT
    distributions](#check-5-correct-vs-incorrect-rt-distributions)
- [Session info](#session-info)

This markdown file is can be used to reproduce the results reported in
the paper: Idiographic Memory Modelling in ACT-R using Alternating
Maximum Likelihood Estimation

This script is used to simulate the data, corresponding to the section:
Simulation: Recovering parameters from synthetic data.

# Setup

``` r
library(here)
library(data.table)
library(janitor)
library(ggplot2)
library(purrr)
library(furrr)
library(progressr)
library(tidyr)
library(jsonlite)
library(patchwork)
library(ggExtra)
library(lmerTest)
library(dplyr)
library(ggsci)

# Set up parallel processing
plan(multisession, workers = availableCores() - 1)

# Load the simulation, fitting, and recovery functions:
source(here::here("R", "sim-mle.R"))

# Define colours
col_blue <- "#0571b0"
col_red <- "#ca0020"
col_green <- "#1b9e77"
```

# Simulate data

Let’s first simulate data for 50 learners, each learning 15 facts, with
10 repetitions per fact. We set a seed for reproducibility. For more
details on the simulation procedure, see the `sim_mle.R` script.

Each learner is assigned a set of person-level parameters (drawn from
the truncated normal distributions in `sim-mle.R`), and each fact within
a learner gets a fact-level Speed of Forgetting offset (Δα). The
simulator then generates a study trial plus repeated retrieval trials
per fact, recording response accuracy and response time for every trial.
The first encounter of each fact is flagged with `is_study == TRUE` and
is excluded from likelihood evaluation later on.

``` r
sim <- simulate_fact_learner_data(
  n_learners    = 50,
  n_facts       = 15,
  n_repetitions = 10,
  seed          = 999
)
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |=                                                                     |   2%  |                                                                              |===                                                                   |   4%  |                                                                              |====                                                                  |   6%  |                                                                              |======                                                                |   8%  |                                                                              |=======                                                               |  10%  |                                                                              |========                                                              |  12%  |                                                                              |==========                                                            |  14%  |                                                                              |===========                                                           |  16%  |                                                                              |=============                                                         |  18%  |                                                                              |==============                                                        |  20%  |                                                                              |===============                                                       |  22%  |                                                                              |=================                                                     |  24%  |                                                                              |==================                                                    |  26%  |                                                                              |====================                                                  |  28%  |                                                                              |=====================                                                 |  30%  |                                                                              |======================                                                |  32%  |                                                                              |========================                                              |  34%  |                                                                              |=========================                                             |  36%  |                                                                              |===========================                                           |  38%  |                                                                              |============================                                          |  40%  |                                                                              |=============================                                         |  42%  |                                                                              |===============================                                       |  44%  |                                                                              |================================                                      |  46%  |                                                                              |==================================                                    |  48%  |                                                                              |===================================                                   |  50%  |                                                                              |====================================                                  |  52%  |                                                                              |======================================                                |  54%  |                                                                              |=======================================                               |  56%  |                                                                              |=========================================                             |  58%  |                                                                              |==========================================                            |  60%  |                                                                              |===========================================                           |  62%  |                                                                              |=============================================                         |  64%  |                                                                              |==============================================                        |  66%  |                                                                              |================================================                      |  68%  |                                                                              |=================================================                     |  70%  |                                                                              |==================================================                    |  72%  |                                                                              |====================================================                  |  74%  |                                                                              |=====================================================                 |  76%  |                                                                              |=======================================================               |  78%  |                                                                              |========================================================              |  80%  |                                                                              |=========================================================             |  82%  |                                                                              |===========================================================           |  84%  |                                                                              |============================================================          |  86%  |                                                                              |==============================================================        |  88%  |                                                                              |===============================================================       |  90%  |                                                                              |================================================================      |  92%  |                                                                              |==================================================================    |  94%  |                                                                              |===================================================================   |  96%  |                                                                              |===================================================================== |  98%  |                                                                              |======================================================================| 100%

# Checks

Next, we will inspect to see if the simulated data fits our
expectations.

## Parameter distributions and their relationship to behavior (Figure 2)

This reproduces Figure 2 of the paper. Panel A shows the sampled
distribution of each of the five person-level parameters across the 50
learners; panels B and C relate each parameter to aggregate accuracy and
median RT, so we can verify that the simulator induces the expected
behavioural signatures before fitting anything.

``` r
# ── Shared factor levels (controls column order) ──────────────────────────────
param_levels <- c(
  "Speed of Forgetting",
  "Retrieval threshold",
  "Activation noise",
  "Latency factor",
  "Non-retrieval time"
)

# ── A: Parameter distributions ────────────────────────────────────────────────
sim_test <- sim[is_study == FALSE]

param_long <- sim_test %>%
  distinct(user_id, true_alpha, true_tau, true_s, true_lf, true_ter) %>%
  pivot_longer(
    cols      = c(true_alpha, true_tau, true_s, true_lf, true_ter),
    names_to  = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    parameter = recode(parameter,
      "true_alpha" = "Speed of Forgetting",
      "true_tau"   = "Retrieval threshold",
      "true_s"     = "Activation noise",
      "true_lf"    = "Latency factor",
      "true_ter"   = "Non-retrieval time"
    ),
    parameter = factor(parameter, levels = param_levels)
  )

# ── B & C: Parameters vs behavioural outcomes (per learner) ───────────────────
learner_summary <- sim_test %>%
  group_by(user_id, true_alpha, true_tau, true_s, true_lf, true_ter) %>%
  summarise(
    accuracy = mean(correct),
    med_rt   = median(rt),
    .groups  = "drop"
  ) %>%
  pivot_longer(
    cols      = c(true_alpha, true_tau, true_s, true_lf, true_ter),
    names_to  = "parameter",
    values_to = "param_value"
  ) %>%
  mutate(
    parameter = recode(parameter,
      "true_alpha" = "Speed of Forgetting",
      "true_tau"   = "Retrieval threshold",
      "true_s"     = "Activation noise",
      "true_lf"    = "Latency factor",
      "true_ter"   = "Non-retrieval time"
    ),
    parameter = factor(parameter, levels = param_levels)
  )

# ── Shared x-range per parameter (aligns all three rows) ─────────────────────
param_ranges <- learner_summary %>%
  group_by(parameter) %>%
  summarise(xmin = min(param_value), xmax = max(param_value), .groups = "drop") %>%
  pivot_longer(c(xmin, xmax), names_to = "bound", values_to = "value") %>%
  select(parameter, value) %>%
  mutate(parameter = factor(parameter, levels = param_levels))

# ── Plot A: distributions ─────────────────────────────────────────────────────
distr <- ggplot(param_long, aes(x = value, fill = parameter)) +
  geom_histogram(bins = 12, color = "black", alpha = 0.7) +
  geom_blank(data = param_ranges, aes(x = value), inherit.aes = FALSE) +
  facet_wrap(~ parameter, scales = "free_x", ncol = 5) +
  labs(x = NULL, y = "Count") +
  theme_bw() +
  theme(
    legend.position  = "none",
    strip.background = element_rect(fill = "grey85", colour = "grey60"),
    strip.text       = element_text(size = 8, margin = margin(3, 3, 3, 3)),
    plot.margin      = margin(10, 20, 10, 20)
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 4)
  ) +
  scale_fill_aaas()

# ── Plot B: accuracy ──────────────────────────────────────────────────────────
acc <- ggplot(learner_summary, aes(x = param_value, y = accuracy, color = parameter)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  geom_blank(data = param_ranges, aes(x = value), inherit.aes = FALSE) +
  facet_wrap(~ parameter, scales = "free_x", ncol = 5) +
  labs(x = NULL, y = "Prop. correct") +
  theme_bw() +
  theme(
    legend.position  = "none",
    strip.background = element_blank(),
    strip.text       = element_blank(),
    plot.margin      = margin(10, 20, 10, 20)
  ) + 
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 4)
  ) +
  scale_color_aaas()

# ── Plot C: response time ─────────────────────────────────────────────────────
rt_plot <- ggplot(learner_summary, aes(x = param_value, y = med_rt, color = parameter)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  geom_blank(data = param_ranges, aes(x = value), inherit.aes = FALSE) +
  facet_wrap(~ parameter, scales = "free_x", ncol = 5) +
  labs(x = "Parameter value", y = "Median RT (s)") +
  theme_bw() +
  theme(
    legend.position  = "none",
    strip.background = element_blank(),
    strip.text       = element_blank(),
    plot.margin      = margin(10, 20, 10, 20)
  ) + 
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 4)
  ) +
  scale_color_aaas()

# ── Combine ───────────────────────────────────────────────────────────────────
distr / acc / rt_plot +
  plot_annotation(tag_levels = "A") &
  theme(
    plot.tag      = element_text(face = "bold", size = 14),
    plot.margin   = margin(1, 5, 1, 5),
    panel.spacing = unit(0.4, "lines")
  )
```

    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/01_simulate_data_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
# Paper Figure 2: parameter distributions and their relation to behaviour
ggsave(here::here("output", "parameter_distributions_and_behaviour.png"), width = 10, height = 6.5)
```

    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'

Panel A shows the distributions of the five simulated person-level
parameters. Panels B and C show how each parameter relates to aggregate
accuracy and median RT, respectively. Speed of forgetting (α) and
retrieval threshold (τ) should show the clearest relationships with
accuracy; latency factor (F) and non-retrieval time (t<sub>er</sub>)
should primarily affect RT.

## Check 2: Observed accuracy vs. predicted P(correct)

``` r
# Recompute activation for each TEST trial, using ALL traces (including study trials)
sim_test <- sim[is_study == FALSE]

sim_test[, activation := mapply(function(uid, fid, t) {
  # Get all prior traces for this fact, INCLUDING the study trial
  traces <- sim[user_id == uid & fact_id == fid & start_time < t, start_time]
  if (length(traces) == 0) return(NA_real_)
  calculate_activation_memory(t, traces, sim[user_id == uid & fact_id == fid, true_total_sof[1]])
}, user_id, fact_id, start_time)]

sim_act <- sim_test[is.finite(activation)]

# Per-trial predicted P(correct) using that trial's own parameters
sim_act[, pred_correct := correct_likelihood(activation, true_tau, true_s)]

# Bin by predicted P(correct)
sim_act[, pred_bin := cut(pred_correct,
                          breaks = seq(0, 1, by = 0.05),
                          include.lowest = TRUE)]

calib <- sim_act[, .(
  obs_acc  = mean(correct),
  pred_acc = mean(pred_correct),
  n = .N
), by = pred_bin][n >= 5]

p_calib <- ggplot(calib, aes(x = pred_acc, y = obs_acc)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(size = n), color = "#0571b0", alpha = 0.8) +
  scale_size_continuous(range = c(1.5, 8), name = "N trials") +
  labs(title = "Observed vs. predicted accuracy",
       subtitle = "Predicted P(correct) per trial with own τ and s",
       x = "Predicted P(correct)", y = "Observed P(correct)") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_bw()

# Right panel: accuracy vs activation with individual sigmoids
sigmoid_learners <- sim_act[, .(true_tau = true_tau[1], true_s = true_s[1]), by = user_id]
sigmoid_learners <- sigmoid_learners[order(true_tau)][seq(1, .N, length.out = 9)]

act_seq <- seq(min(sim_act$activation), max(sim_act$activation), length.out = 200)
sigmoid_curves <- sigmoid_learners[, .(
  act = act_seq,
  pred = correct_likelihood(act_seq, true_tau, true_s)
), by = .(user_id, true_tau)]
sigmoid_curves[, label := sprintf("τ=%.2f", true_tau)]

# Bin by activation (equal-width)
act_range <- range(sim_act$activation)
sim_act[, act_bin := cut(activation,
                         breaks = seq(act_range[1], act_range[2], length.out = 21),
                         include.lowest = TRUE)]
act_binned <- sim_act[, .(
  obs_acc  = mean(correct),
  mean_act = mean(activation),
  n = .N
), by = act_bin][n >= 5]

p_sigmoid <- ggplot() +
  geom_line(data = sigmoid_curves, aes(x = act, y = pred, group = user_id, color = label),
            alpha = 0.5, linewidth = 0.6) +
  geom_point(data = act_binned, aes(x = mean_act, y = obs_acc, size = n),
             color = "black", alpha = 0.7) +
  scale_size_continuous(range = c(1.5, 8), name = "N trials") +
  scale_color_discrete(name = "Learner") +
  labs(title = "Accuracy vs. activation",
       subtitle = "Lines = individual sigmoids (9 learners spanning τ range)",
       x = "Activation", y = "P(correct)") +
  theme_bw() +
  theme(legend.text = element_text(size = 8))

p_calib + p_sigmoid
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/01_simulate_data_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

The left panel is a calibration plot: if the simulation is internally
consistent, the binned points should fall close to the diagonal. The
right panel shows how individual retrieval thresholds (τ) shift the
sigmoid mapping from activation to response probability. The binned
observed accuracies (black dots) should track the spread of individual
curves.

## Check 3: Learning curves — accuracy and RT over repetitions

``` r
sim_learn <- sim[is_study == FALSE]

learn_curve <- sim_learn[, .(
  accuracy = mean(correct),
  median_rt = median(rt),
  q25_rt = quantile(rt, 0.25),
  q75_rt = quantile(rt, 0.75)
), by = rep]

p_acc_rep <- ggplot(learn_curve, aes(x = rep, y = accuracy)) +
  geom_line(linewidth = 0.8, color = "#0571b0") +
  geom_point(size = 2, color = "#0571b0") +
  ylim(c(NA, 1)) +
  labs(title = "Accuracy over repetitions", x = "Repetition", y = "Mean accuracy") +
  theme_bw()

p_rt_rep <- ggplot(learn_curve, aes(x = rep, y = median_rt)) +
  geom_ribbon(aes(ymin = q25_rt, ymax = q75_rt), alpha = 0.2, fill = "#ca0020") +
  geom_line(linewidth = 0.8, color = "#ca0020") +
  geom_point(size = 2, color = "#ca0020") +
  labs(title = "RT over repetitions", x = "Repetition", y = "Median RT (s)") +
  theme_bw()

p_acc_rep + p_rt_rep
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/01_simulate_data_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Accuracy should increase and RT should decrease over repetitions,
reflecting the strengthening of memory traces through practice. The IQR
ribbon around the RT curve shows the spread across learners.

## Check 4: Effect of fact-level offsets (Δα) on accuracy and RT

``` r
fact_summary <- sim[is_study == FALSE, .(
  accuracy = mean(correct),
  median_rt = median(rt)
), by = .(user_id, fact_id, true_fact_offset)]

p_da_acc <- ggplot(fact_summary, aes(x = true_fact_offset, y = accuracy)) +
  geom_point(alpha = 0.15, color = "#1b9e77") +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  labs(title = "Fact offset vs. accuracy", x = expression(Delta*alpha), y = "Mean accuracy") +
  theme_bw()

p_da_rt <- ggplot(fact_summary, aes(x = true_fact_offset, y = median_rt)) +
  geom_point(alpha = 0.15, color = "#1b9e77") +
  geom_smooth(method = "lm", color = "black", se = TRUE) +
  labs(title = "Fact offset vs. RT", x = expression(Delta*alpha), y = "Median RT (s)") +
  theme_bw()

p_da_acc + p_da_rt
```

    ## `geom_smooth()` using formula = 'y ~ x'
    ## `geom_smooth()` using formula = 'y ~ x'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/01_simulate_data_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Higher fact offsets (Δα) make a fact harder to retain, so accuracy
should decrease and RT should increase with larger offsets. Each dot is
one learner–fact combination.

## Check 5: Correct vs. incorrect RT distributions

``` r
sim_test <- sim[is_study == FALSE]

ggplot(sim_test, aes(x = rt, fill = correct)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("TRUE" = "#0571b0", "FALSE" = "#ca0020"),
                    labels = c("TRUE" = "Correct", "FALSE" = "Incorrect")) +
  labs(title = "RT distribution by response accuracy",
       x = "RT (s)", y = "Density", fill = "") +
  coord_cartesian(xlim = c(0, quantile(sim_test$rt, 0.99))) +
  theme_bw() +
  theme(legend.position = c(0.85, 0.85))
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/01_simulate_data_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Correct responses should be faster than incorrect ones, consistent with
the model’s assumption that higher activation produces both faster and
more accurate retrieval. The distributions should overlap partially,
with the incorrect-response distribution shifted towards longer RTs.

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
    ##  [1] Rcpp_1.1.1-1.1    ggsci_4.0.0       dplyr_1.1.4       lmerTest_3.1-3   
    ##  [5] lme4_2.0-1        Matrix_1.7-3      ggExtra_0.11.0    patchwork_1.3.2  
    ##  [9] jsonlite_2.0.0    tidyr_1.3.1       progressr_0.17.0  furrr_0.3.1      
    ## [13] future_1.67.0     purrr_1.1.0       ggplot2_4.0.0     janitor_2.2.1    
    ## [17] data.table_1.17.8 here_1.0.2       
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] gtable_0.3.6        xfun_0.53           lattice_0.22-7     
    ##  [4] numDeriv_2016.8-1.1 vctrs_0.6.5         tools_4.5.1        
    ##  [7] Rdpack_2.6.4        generics_0.1.4      parallel_4.5.1     
    ## [10] tibble_3.3.0        pkgconfig_2.0.3     RColorBrewer_1.1-3 
    ## [13] S7_0.2.0            lifecycle_1.0.4     compiler_4.5.1     
    ## [16] farver_2.1.2        stringr_1.5.2       textshaping_1.0.4  
    ## [19] codetools_0.2-20    snakecase_0.11.1    httpuv_1.6.16      
    ## [22] htmltools_0.5.8.1   yaml_2.3.10         nloptr_2.2.1       
    ## [25] pillar_1.11.1       later_1.4.4         MASS_7.3-65        
    ## [28] reformulas_0.4.4    boot_1.3-32         nlme_3.1-168       
    ## [31] mime_0.13           parallelly_1.45.1   tidyselect_1.2.1   
    ## [34] digest_0.6.37       stringi_1.8.7       listenv_0.9.1      
    ## [37] labeling_0.4.3      splines_4.5.1       rprojroot_2.1.1    
    ## [40] fastmap_1.2.0       grid_4.5.1          cli_3.6.5          
    ## [43] magrittr_2.0.4      withr_3.0.2         scales_1.4.0       
    ## [46] promises_1.4.0      lubridate_1.9.4     timechange_0.3.0   
    ## [49] rmarkdown_2.30      globals_0.18.0      otel_0.2.0         
    ## [52] ragg_1.5.0          shiny_1.11.1        evaluate_1.0.5     
    ## [55] knitr_1.50          rbibutils_2.3       miniUI_0.1.2       
    ## [58] mgcv_1.9-3          rlang_1.1.6         xtable_1.8-4       
    ## [61] glue_1.8.0          minqa_1.2.8         rstudioapi_0.17.1  
    ## [64] R6_2.6.1            systemfonts_1.3.1
