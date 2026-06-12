Example application: Modelling memory function in MCI and healthy ageing
================
Maarten van der Velde & Thomas Wilschut
Last updated: 2026-06-12

- [Overview](#overview)
- [Setup](#setup)
- [Load data](#load-data)
- [Inspect data](#inspect-data)
  - [Structure](#structure)
  - [Session information](#session-information)
- [Filter data](#filter-data)
- [Fit model](#fit-model)
- [Evaluate model fit](#evaluate-model-fit)
  - [How well does the model fit capture the
    data?](#how-well-does-the-model-fit-capture-the-data)
    - [Decile plots of predicted and observed
      RT](#decile-plots-of-predicted-and-observed-rt)
    - [Correlation between predicted and observed
      RT](#correlation-between-predicted-and-observed-rt)
    - [Visual comparison of full RT
      distributions](#visual-comparison-of-full-rt-distributions)
  - [Estimated parameters](#estimated-parameters)
    - [Parameter estimates per
      session](#parameter-estimates-per-session)
    - [Relationship to session
      properties](#relationship-to-session-properties)
    - [Stability over time](#stability-over-time)
    - [Average estimated parameters](#average-estimated-parameters)
    - [Correlation](#correlation)
  - [Fact-level offsets](#fact-level-offsets)

# Overview

In this notebook, we apply the AMLE method to an existing retrieval
practice data set from [Hake et
al. (2024)](https://www.medrxiv.org/content/10.1101/2024.03.15.24304345v1).
Participants completed multiple retrieval practice sessions over a
period of about a year, resulting in repeated longitudinal measurements
of memory performance. The sample included participants clinically
diagnosed with Mild Cognitive Impairment (MCI) and age-matched healthy
control (HC) participants. MCI is known to affect both memory and
non-memory cognitive processes, which is visible in participants’
performance on the task: Hake et al. found that participants’ clinical
status could be reliably determined from the Speed of Forgetting
($\alpha$) parameter estimated from the data. However, in that analysis,
other parameters of the memory model were held constant. Here, we fit
the same data set using the AMLE method while allowing more parameters
to vary, to see whether performance differences between individuals and
groups are also explained through differences in other parameters
(specifically: activation noise ($s$), retrieval threshold ($\tau$),
latency factor ($F$) and non-retrieval time ($t_{er}$)).

# Setup

``` r
library(here)
```

    ## here() starts at /Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle

``` r
library(data.table)
library(purrr)
```

    ## 
    ## Attaching package: 'purrr'

    ## The following object is masked from 'package:data.table':
    ## 
    ##     transpose

``` r
library(furrr)
```

    ## Loading required package: future

``` r
library(tidyr)
library(jsonlite)
```

    ## 
    ## Attaching package: 'jsonlite'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     flatten

``` r
library(lmerTest)
```

    ## Loading required package: lme4

    ## Warning: package 'lme4' was built under R version 4.5.2

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## 
    ## Attaching package: 'lmerTest'

    ## The following object is masked from 'package:lme4':
    ## 
    ##     lmer

    ## The following object is masked from 'package:stats':
    ## 
    ##     step

``` r
library(psych)
library(ggplot2)
```

    ## 
    ## Attaching package: 'ggplot2'

    ## The following objects are masked from 'package:psych':
    ## 
    ##     %+%, alpha

``` r
library(patchwork)
library(ggExtra)
library(GGally)
library(tidytext)

# Set up parallel processing
plan(multisession, workers = 8)

source(here("R", "sim-mle.R"))
```

    ## Warning: package 'Rcpp' was built under R version 4.5.2

``` r
set.seed(2026)

# Define colours
col_blue <- "#0571b0"
col_red <- "#ca0020"
col_green <- "#1b9e77"

# Define plotting theme
theme_set(theme_bw())

# Use cached version of model fit if available?
refit_model <- FALSE
```

# Load data

``` r
d <- fread(here("data", "processed", "hake2024.csv"))
```

# Inspect data

## Structure

Each row in the data corresponds to a single retrieval practice trial,
recording the context in which it happened and the participant’s
performance:

``` r
head(d)
```

    ##    user_id clinical_status   session_id lesson_id  fact_id repetition
    ##     <char>          <char>       <char>    <char>   <char>      <int>
    ## 1: user_01             MCI session_1192 lesson_07 fact_256          1
    ## 2: user_01             MCI session_1192 lesson_07 fact_256          2
    ## 3: user_01             MCI session_1192 lesson_07 fact_048          1
    ## 4: user_01             MCI session_1192 lesson_07 fact_048          2
    ## 5: user_01             MCI session_1192 lesson_07 fact_604          1
    ## 6: user_01             MCI session_1192 lesson_07 fact_701          1
    ##    start_time     rt correct session
    ##         <num>  <num>  <lgcl>   <int>
    ## 1: 1684597178 59.536    TRUE       1
    ## 2: 1684597238  4.657    TRUE       1
    ## 3: 1684597244  5.636    TRUE       1
    ## 4: 1684597250  3.290    TRUE       1
    ## 5: 1684597255  4.084    TRUE       1
    ## 6: 1684597259  6.830    TRUE       1

## Session information

Participants completed multiple sessions, each of which lasted about 8
minutes. The number of facts encountered, total number of trials, and
performance varied across sessions depending on performance, which may
be relevant when interpreting the fitted parameters later on.

``` r
session_stats <- d[, .(
  trials = .N,
  errors = sum(correct == FALSE),
  accuracy = mean(correct),
  median_rt = median(rt),
  facts = uniqueN(fact_id),
  reps_per_fact = .N / uniqueN(fact_id),
  duration_min = ((max(start_time) + rt[which.max(start_time)]) - min(start_time))/60),
  by = .(user_id, clinical_status, session, session_id, lesson_id)]

stats_by_clinical_status <- session_stats[, .(
  sessions = .N,
  participants = uniqueN(user_id),
  mean_trials_per_session = mean(trials),
  mean_duration_min = mean(duration_min),
  mean_reps_per_fact = mean(reps_per_fact),
  mean_n_facts = mean(facts),
  mean_accuracy = mean(accuracy),
  mean_median_rt = mean(median_rt)
), by = clinical_status]

stats_combined <- session_stats[, .(
  clinical_status = "Combined",
  sessions = .N,
  participants = uniqueN(user_id),
  mean_trials_per_session = mean(trials),
  mean_duration_min = mean(duration_min),
  mean_reps_per_fact = mean(reps_per_fact),
  mean_n_facts = mean(facts),
  mean_accuracy = mean(accuracy),
  mean_median_rt = mean(median_rt)
)]

rbind(stats_by_clinical_status, stats_combined)
```

    ##    clinical_status sessions participants mean_trials_per_session
    ##             <char>    <int>        <int>                   <num>
    ## 1:             MCI      900           24                62.04111
    ## 2:              HC     1248           27                98.30929
    ## 3:        Combined     2148           51                83.11313
    ##    mean_duration_min mean_reps_per_fact mean_n_facts mean_accuracy
    ##                <num>              <num>        <num>         <num>
    ## 1:          8.046848           6.135502     10.17111     0.8347306
    ## 2:          8.019310           6.859941     14.23397     0.9421364
    ## 3:          8.030849           6.556405     12.53166     0.8971340
    ##    mean_median_rt
    ##             <num>
    ## 1:       5.972667
    ## 2:       3.187801
    ## 3:       4.354644

The pairwise plots below show how these session characteristics relate
to each other, and how they differ between clinical groups. On average,
participants in the MCI group tended to complete fewer trials per
session, had fewer error-free sessions and lower accuracy, had longer
response times, and encountered fewer unique facts.

``` r
ggpairs(session_stats,
        columns = c("trials", "errors", "accuracy", "median_rt", "facts", "reps_per_fact", "duration_min"),
        columnLabels = c("Trials", "Errors", "Accuracy", "Median RT (s)", "Unique facts", "Repetitions per fact", "Duration (min)"),
        mapping = aes(colour = clinical_status, fill = clinical_status, alpha = .25),
        upper = list(continuous = wrap("cor", method = "pearson")),
        progress = FALSE) +
  scale_color_manual(values = c(col_blue, col_red)) +
  scale_fill_manual(values = c(col_blue, col_red))
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/session-stats-correlation-1.png)<!-- -->

# Filter data

For model fitting, we’ll apply the filtering rules used in Hake et al.’s
analysis of this data set:

- Exclude fact sequences containing a response time below 300 ms or
  above 15 s, which are likely to reflect accidental responding or
  disengagement.
- Exclude fact sequences with fewer than 3 repetitions per fact, which
  contain insufficient information for reliable parameter estimation.

``` r
# Filtering criteria
min_rt <- 0.3
max_rt <- 15
min_reps_per_fact <- 3

fact_sequences <- split(d, by = c("user_id", "session_id", "fact_id"))
fact_sequences_filtered <- fact_sequences[map_lgl(fact_sequences, ~ nrow(.x) >= min_reps_per_fact && min(.x$rt) >= min_rt && max(.x$rt) <= max_rt)]

d_filtered <- rbindlist(fact_sequences_filtered)
setorder(d_filtered, user_id, start_time)
```

Recalculate session statistics for the filtered set:

``` r
session_stats_filtered <- d_filtered[, .(
  trials = .N,
  errors = sum(correct == FALSE),
  accuracy = mean(correct),
  median_rt = median(rt),
  facts = uniqueN(fact_id),
  reps_per_fact = .N / uniqueN(fact_id),
  duration_min = ((max(start_time) + rt[which.max(start_time)]) - min(start_time))/60),
  by = .(user_id, clinical_status, session, session_id, lesson_id)]

stats_by_clinical_status_filtered <- session_stats_filtered[, .(
  sessions = .N,
  participants = uniqueN(user_id),
  mean_trials_per_session = mean(trials),
  mean_duration_min = mean(duration_min),
  mean_reps_per_fact = mean(reps_per_fact),
  mean_n_facts = mean(facts),
  mean_accuracy = mean(accuracy),
  mean_median_rt = mean(median_rt)
), by = clinical_status]

stats_combined_filtered <- session_stats_filtered[, .(
  clinical_status = "Combined",
  sessions = .N,
  participants = uniqueN(user_id),
  mean_trials_per_session = mean(trials),
  mean_duration_min = mean(duration_min),
  mean_reps_per_fact = mean(reps_per_fact),
  mean_n_facts = mean(facts),
  mean_accuracy = mean(accuracy),
  mean_median_rt = mean(median_rt)
)]

rbind(stats_by_clinical_status_filtered, stats_combined_filtered)
```

    ##    clinical_status sessions participants mean_trials_per_session
    ##             <char>    <int>        <int>                   <num>
    ## 1:             MCI      816           24                52.17402
    ## 2:              HC     1245           27                92.73655
    ## 3:        Combined     2061           51                76.67686
    ##    mean_duration_min mean_reps_per_fact mean_n_facts mean_accuracy
    ##                <num>              <num>        <num>         <num>
    ## 1:          7.474282           6.343031     8.053922     0.8527335
    ## 2:          7.897532           6.908353    13.110040     0.9466660
    ## 3:          7.729957           6.684528    11.108200     0.9094758
    ##    mean_median_rt
    ##             <num>
    ## 1:       4.835191
    ## 2:       3.080505
    ## 3:       3.775228

``` r
ggpairs(session_stats,
        columns = c("trials", "errors", "accuracy", "median_rt", "facts", "reps_per_fact", "duration_min"),
        columnLabels = c("Trials", "Errors", "Accuracy", "Median RT (s)", "Unique facts", "Repetitions per fact", "Duration (min)"),
        mapping = aes(colour = clinical_status, fill = clinical_status, alpha = .25),
        upper = list(continuous = wrap("cor", method = "pearson")),
        progress = FALSE) +
  scale_color_manual(values = c(col_blue, col_red)) +
  scale_fill_manual(values = c(col_blue, col_red))
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/session-stats-filtered-1.png)<!-- -->

# Fit model

The data set is split into sessions, each of which is fitted separately.

``` r
d_sessions <- split(d_filtered, by = c("session_id"))
```

If desired, load previously fitted model results; otherwise, fit the
model to each session and save the results for future use. (Note:
fitting takes a while!)

``` r
if (refit_model == FALSE &&
    file.exists(here("data", "processed", "AMLE_fit.csv")) &&
    file.exists(here("data", "processed", "AMLE_delta_alpha.csv"))) {
  fit_amle <- fread(here("data", "processed", "AMLE_fit.csv"))
  fit_delta_alpha <- fread(here("data", "processed", "AMLE_delta_alpha.csv"))
  
} else {

  # Fit model
  fit_amle <- future_map(d_sessions, function (d_i) {
    
    fit_i_cpp <- fit_one_learner(d_i, full_iter = 100)[.N]
    
    # If the fit failed and returned NULL, replace with an empty data.table
    if(is.null(fit_i_cpp)) {
      fit_i_cpp <- data.table()
    }
    # Add participant and session information
    fit_i_cpp[, user_id := d_i$user_id[1]]
    fit_i_cpp[, clinical_status := d_i$clinical_status[1]]
    fit_i_cpp[, session := d_i$session[1]]
    fit_i_cpp[, session_id := d_i$session_id[1]]
    fit_i_cpp[, lesson_id := d_i$lesson_id[1]]
    fit_i_cpp[, session_trials := nrow(d_i)]
    fit_i_cpp[, session_errors := sum(d_i$correct == FALSE)]
    fit_i_cpp[, session_accuracy := mean(d_i$correct)]
    
    return(fit_i_cpp)
  }, .progress = interactive())
  
  fit_amle <- rbindlist(fit_amle, fill = TRUE)
  
  # Extract delta_alpha estimates
  fit_delta_alpha <- fit_amle[, .(delta_alpha = delta_alpha_est), by = .(user_id, clinical_status, session, session_id, lesson_id, session_trials, session_errors, session_accuracy)]
  fit_delta_alpha[, delta_alpha := lapply(delta_alpha, function (delta_alpha) {
    as.data.table(delta_alpha, keep.rownames = "fact_id")
  })]
  fit_delta_alpha <- unnest(fit_delta_alpha, delta_alpha) |> as.data.table()
  
  # Save results
  fwrite(fit_amle, here("data", "processed", "AMLE_fit.csv"))
  fwrite(fit_delta_alpha, here("data", "processed", "AMLE_delta_alpha.csv"))
  
}
```

# Evaluate model fit

## How well does the model fit capture the data?

How well does the fitted model capture the distribution of response
times? Here, we use the fitted parameters to predict trial-level
response times.

``` r
fitted_session_params <- fit_amle[, .(session_id, alpha, tau, s, lf, ter)]
fitted_session_params <- fit_delta_alpha[fitted_session_params, on = .(session_id)]
fitted_session_params[, fact_sof := alpha + delta_alpha]
fitted_session_params[, fact_id := gsub("d_", "", fact_id)]

d_preds <- future_map(d_sessions, function (d_session) {
  
  session_params <- fitted_session_params[session_id == d_session$session_id[1]]
  
  # If session params is empty or parameters are NA, return NULL
  if (nrow(session_params) == 0 || any(is.na(session_params[, .(delta_alpha, alpha, tau, s, lf, ter)]))) {
    return (NULL)
  }
  
  fitted_ter <- session_params[1, ter]
  fitted_lf <- session_params[1, lf]
  fitted_tau <- session_params[1, tau]
  
  pred_rt_error <- fitted_ter + fitted_lf * exp(-fitted_tau)
  
  setorder(d_session, start_time)
  
  model_predictions <- map(seq_len(nrow(d_session)), function (trial) {
    current_trial <- d_session[trial]
    previous_trials_for_fact <- d_session[fact_id == current_trial$fact_id & start_time < current_trial$start_time]
    
    # If there are no previous trials for this fact, the activation is -Inf
    if (nrow(previous_trials_for_fact) == 0) {
      activation <- -Inf
    } else {
      activation <- calculate_activation_memory(t = current_trial$start_time,
                                                traces = previous_trials_for_fact$start_time, 
                                                sof = session_params[fact_id == current_trial$fact_id, fact_sof])
    }
    pred_rt_correct <- fitted_ter + fitted_lf * exp(-activation)
    
    return (list(
      activation = activation,
      pred_rt_correct = pred_rt_correct,
      pred_rt_error = pred_rt_error))
    
  }) |> rbindlist()
  
  return (cbind(d_session, model_predictions))
  
}) |> rbindlist()
```

### Decile plots of predicted and observed RT

The decile plots below compare each participant’s response time
distribution (on correct responses) as predicted from fitted model
parameters to their observed response time distribution. Agreement is
quantified per participant by a Pearson correlation, shown in the top of
each plot.

``` r
# Create simplified user IDs that are sorted by clinical status
user_ids <- unique(d_preds[, .(user_id, clinical_status)])
setorder(user_ids, clinical_status)
user_ids[, user_id_simple := sprintf("%02d", .I)]
d_preds <- user_ids[d_preds, on = .(user_id, clinical_status)]

# Only evaluate RT predictions on correct test trials
d_preds_correct <- d_preds[repetition != 1 & correct == TRUE]

# Test the correlation between predicted and observed RT per participant
d_preds_correct_by_user <- split(d_preds_correct, by = "user_id_simple")

cor_rt <- map(d_preds_correct_by_user, function (x) {
  cor_rt <- cor.test(x$pred_rt_correct, x$rt)
  return (list(
    user_id_simple = x$user_id_simple[1],
    clinical_status = x$clinical_status[1],
    r = cor_rt$estimate,
    p = cor_rt$p.value,
    n = nrow(x)
  ))
}) |> rbindlist()
cor_rt[, cor_label := paste0("r(", n, ") = ", sprintf("%.2f", r), ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", ""))))]

# Also compute decile points to plot on top (i.e., 10th decile of predicted RTs and 10th decile of observed RTs)
deciles <- d_preds_correct[, .(
  pred_rt_correct = quantile(pred_rt_correct, probs = seq(.1, .9, by = .1)),
  rt = quantile(rt, probs = seq(.1, .9, by = .1))
), by = .(user_id_simple, clinical_status)]

# Create plot
p_decile_rt <- ggplot(d_preds_correct, aes(x = pred_rt_correct, y = rt, colour = clinical_status)) +
  facet_wrap(~ user_id_simple, ncol = 9, labeller = labeller(user_id_simple = function(x) paste("Participant", x))) +
  geom_point(alpha = .01, size = .5, colour = "grey30") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_line(data = deciles, alpha = .5) +
  geom_point(data = deciles, alpha = .8) +
  geom_label(data = cor_rt, aes(x = 7.5, y = 15, label = cor_label), hjust = .5, vjust = 1, size = 2.5, fill = "white", show.legend = FALSE) +
  labs(x = "Predicted RT (s)", y = "Observed RT (s)", colour = "Clinical status") +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  coord_equal(ratio = 1, xlim = c(0, 15), ylim = c(0, 15)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1))) +
  theme(legend.position = c(.8375, .075), legend.direction = "horizontal")

# Paper Figure 8B: per-participant decile plots of predicted vs observed RT
ggsave(here("output", "predicted_vs_observed_rt.png"), plot = p_decile_rt, width = 10, height = 10)
p_decile_rt
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/decile-plots-1.png)<!-- -->

### Correlation between predicted and observed RT

The correlation coefficients between predicted and observed RTs (on
correct test trials) are distributed as follows:

``` r
cor_rt_dist <- copy(cor_rt)
cor_rt_dist[, clinical_status := factor(clinical_status, levels = c("MCI", "HC"))]

p_rt_corr <- ggplot(cor_rt_dist, aes(x = r, fill = clinical_status, y = clinical_status)) +
  # geom_violin(alpha = .5) +
  # ggdist::stat_halfeye(adjust = .5, justification = -.5, alpha = .5, scale = .5, .width = 0, point_colour = NA) +
  geom_boxplot(outlier.shape = NA, width = .2, alpha = .5) +
  geom_jitter(aes(colour = clinical_status), width = 0, height = .2, alpha = .25) +
  labs(x = "Correlation between predicted and observed RT", y = "Clinical status") +
  scale_x_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  guides(fill = "none", colour = "none")

# Paper Figure 8A: distribution of per-participant predicted-vs-observed RT correlations
ggsave(here("output", "predicted_vs_observed_rt_correlation_dist.png"), plot = p_rt_corr, width = 10, height = 4)
p_rt_corr
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/correlation-rt-1.png)<!-- -->

Combined version of these RT fit plots:

``` r
p_rt_corr + p_decile_rt  +
  plot_layout(ncol = 1, heights = c(1, 8)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/rt-fit-plots-1.png)<!-- -->

``` r
# Paper Figure 8: RT model fit, panels A (correlation distribution) + B (decile plots) combined
ggsave(here("output", "rt_fit_plots.png"), width = 10, height = 10)
```

### Visual comparison of full RT distributions

We’ll also compare the full RT distributions, including incorrect
responses. For incorrect RTs (predicted and observed) we flip the sign
so that they appear on the left side of the plot.

``` r
d_preds_both <- d_preds[repetition != 1]
d_preds_both[, pred_rt := ifelse(correct == TRUE, pred_rt_correct, pred_rt_error)]

# Flip the sign of error RTs
d_preds_both[, pred_rt := ifelse(correct == TRUE, pred_rt, -pred_rt)]
d_preds_both[, rt := ifelse(correct == TRUE, rt, -rt)]

ggplot(d_preds_both, aes(x = rt, fill = clinical_status)) +
  facet_wrap(~ user_id_simple, ncol = 9, labeller = labeller(user_id_simple = function(x) paste("Participant", x))) +
  geom_density(aes(alpha = "Observed")) +
  geom_density(aes(x = pred_rt, alpha = "Predicted")) +
  geom_vline(xintercept = 0, linetype = "dotted") +
  scale_x_continuous(limits = c(-15, 15)) +
  labs(x = "RT (s)", y = "Density", fill = "Clinical status") +
  scale_fill_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_alpha_manual(name = "Distribution", values = c("Observed" = .15, "Predicted" = .75), labels = c("Observed", "Predicted")) +
  theme(legend.position = c(.8375, .075), legend.direction = "horizontal") +
  guides(fill = guide_legend(override.aes = list(alpha = 1)),
         alpha = guide_legend(override.aes = list(fill = "grey20")))
```

    ## Warning: Removed 13 rows containing non-finite outside the scale range
    ## (`stat_density()`).

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/rt-distribution-full-1.png)<!-- -->

``` r
# Paper Figure 7: predicted vs observed RT density per participant (negative RTs = incorrect)
ggsave(here("output", "predicted_vs_observed_rt_density.png"), width = 10, height = 8.5)
```

    ## Warning: Removed 13 rows containing non-finite outside the scale range
    ## (`stat_density()`).

## Estimated parameters

### Parameter estimates per session

``` r
setorder(fit_amle, clinical_status, user_id, session, lesson_id)
fit_amle[, session_aligned := 1:.N, by = .(user_id)]
fit_amle_long <- melt(fit_amle, measure.vars = c("alpha", "tau", "s", "lf", "ter"))
fit_amle_long <- user_ids[fit_amle_long, on = .(user_id, clinical_status)]
```

The following plot shows each session’s parameter estimates, with a
horizontal line indicating the average parameter value across sessions
for each participant.

``` r
fit_amle_long_avg <- fit_amle_long[, .(value_mean = mean(value, na.rm = TRUE)), by = .(user_id_simple, clinical_status, variable)]

ggplot(fit_amle_long, aes(x = session_aligned, y = value, colour = clinical_status, group = user_id_simple)) +
  facet_wrap(user_id_simple ~ variable, ncol = 5, scale = "free_y", labeller = labeller(user_id_simple = function(x) paste("Participant", x))) +
  geom_point(alpha = .2) +
  geom_hline(data = fit_amle_long_avg, aes(yintercept = value_mean, colour = clinical_status), lwd = 1) +
  labs(x = "Session number (aligned)", y = "Parameter value", colour = "Clinical\nstatus") +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_x_continuous(breaks = seq(0, 60, by = 30)) +
  theme(legend.position = "bottom")
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-over-time-1.png)<!-- -->

### Relationship to session properties

How does the number of trials, number of unique facts, number of errors,
accuracy or RT in a session relate to the parameter estimates?

``` r
# Add session stats
fit_amle_long <- session_stats[fit_amle_long, on = .(user_id, clinical_status, session, session_id, lesson_id)]

ggplot(fit_amle_long, aes(x = session_trials, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Number of trials in session", y = "Parameter value", title = "Session trials")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-1.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = facts, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Number of unique facts in session", y = "Parameter value", title = "Unique facts")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-2.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = session_errors, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Number of errors in session", y = "Parameter value", title = "Session errors") 
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-3.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = session_accuracy, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Session accuracy", y = "Parameter value", title = "Session accuracy")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-4.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = median_rt, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Median RT in session (s)", y = "Parameter value", title = "Session median RT")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-5.png)<!-- -->

### Stability over time

Participants completed multiple sessions over time, and parameters are
estimated independently for each session. How consistent are parameter
estimates within each participant? As we would expect, there is
variation in estimated parameter values from session to session. This
variation can come from three sources:

1.  **Materials**: Part of this variation will be due to participants
    practicing different materials in each session, meaning that some
    sessions will end up being a bit more difficult than others.
2.  **Participants**: Part of this variation will also be
    within-participant variability, due to fluctuations in cognitive
    performance over time.
3.  **Noise**: Finally, part of this variation will be noise, e.g. due
    to measurement error or sources of variance that are otherwise not
    accounted for.

To find out how consistent estimates are within participants (more
precisely: what proportion of variance can be attributed to stable
differences between participants), we can compute intra-class
correlations (ICC). To factor out variance due to fact/lesson
differences and to account for missing data, we’ll calculate ICC from a
mixed-effects regression model with random intercepts for participants
and lessons.

$$ICC_{participant} = \frac{\sigma^2_{participant}}{\sigma^2_{participant} + \sigma^2_{residual}}$$

NB: this assumes that parameters are stationary (i.e., don’t
systematically change over time). If that isn’t the case, this method
underestimates the true consistency.

The result is shown per parameter under ICC3 in the tables below.

``` r
for (param in fit_amle_long[, unique(variable)]) {
  icc_data_param <- dcast(fit_amle_long[variable == param], user_id_simple ~ lesson_id, value.var = "value")
  icc_param <- ICC(icc_data_param[, -1, with = FALSE], lmer = TRUE)
  print(paste("Parameter:", param))
  print(icc_param$results)
}
```

    ## [1] "Parameter: alpha"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.2797114 23.52328  50 2907 5.932489e-176
    ## Single_random_raters     ICC2 0.2815072 29.45048  50 2850 1.018702e-217
    ## Single_fixed_raters      ICC3 0.3290957 29.45048  50 2850 1.018702e-217
    ## Average_raters_absolute ICC1k 0.9574889 23.52328  50 2907 5.932489e-176
    ## Average_random_raters   ICC2k 0.9578495 29.45048  50 2850 1.018702e-217
    ## Average_fixed_raters    ICC3k 0.9660447 29.45048  50 2850 1.018702e-217
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.2098333   0.3793249
    ## Single_random_raters      0.2106851   0.3818540
    ## Single_fixed_raters       0.2519361   0.4348774
    ## Average_raters_absolute   0.9390327   0.9725626
    ## Average_random_raters     0.9393257   0.9728475
    ## Average_fixed_raters      0.9512991   0.9780858
    ## [1] "Parameter: tau"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.1817292 13.88118  50 2907 8.139753e-101
    ## Single_random_raters     ICC2 0.1821611 14.42101  50 2850 5.797887e-105
    ## Single_fixed_raters      ICC3 0.1879140 14.42101  50 2850 5.797887e-105
    ## Average_raters_absolute ICC1k 0.9279600 13.88118  50 2907 8.139753e-101
    ## Average_random_raters   ICC2k 0.9281537 14.42101  50 2850 5.797887e-105
    ## Average_fixed_raters    ICC3k 0.9306567 14.42101  50 2850 5.797887e-105
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.1301612   0.2612152
    ## Single_random_raters      0.1306210   0.2615974
    ## Single_fixed_raters       0.1350336   0.2690142
    ## Average_raters_absolute   0.8966838   0.9535042
    ## Average_random_raters     0.8970588   0.9535919
    ## Average_fixed_raters      0.9005433   0.9552471
    ## [1] "Parameter: s"
    ##                          type        ICC        F df1  df2            p
    ## Single_raters_absolute   ICC1 0.05813699 4.580081  50 2907 1.420235e-23
    ## Single_random_raters     ICC2 0.05936011 4.978744  50 2850 9.428956e-27
    ## Single_fixed_raters      ICC3 0.06419531 4.978744  50 2850 9.428956e-27
    ## Average_raters_absolute ICC1k 0.78166323 4.580081  50 2907 1.420235e-23
    ## Average_random_raters   ICC2k 0.78541481 4.978744  50 2850 9.428956e-27
    ## Average_fixed_raters    ICC3k 0.79914615 4.978744  50 2850 9.428956e-27
    ##                         lower bound upper bound
    ## Single_raters_absolute   0.03644214  0.09511176
    ## Single_random_raters     0.03768556  0.09628864
    ## Single_fixed_raters      0.04086707  0.10375445
    ## Average_raters_absolute  0.68687210  0.85908175
    ## Average_random_raters    0.69431671  0.86072005
    ## Average_fixed_raters     0.71192219  0.87037243
    ## [1] "Parameter: lf"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.2819062 23.76939  50 2907 8.924254e-178
    ## Single_random_raters     ICC2 0.2823132 24.91031  50 2850 1.636827e-185
    ## Single_fixed_raters      ICC3 0.2919084 24.91031  50 2850 1.636827e-185
    ## Average_raters_absolute ICC1k 0.9579291 23.76939  50 2907 8.924254e-178
    ## Average_random_raters   ICC2k 0.9580100 24.91031  50 2850 1.636827e-185
    ## Average_fixed_raters    ICC3k 0.9598560 24.91031  50 2850 1.636827e-185
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.2116764   0.3818474
    ## Single_random_raters      0.2121007   0.3822091
    ## Single_fixed_raters       0.2200945   0.3932911
    ## Average_raters_absolute   0.9396640   0.9728467
    ## Average_random_raters     0.9398079   0.9728872
    ## Average_fixed_raters      0.9424228   0.9740917
    ## [1] "Parameter: ter"
    ##                          type       ICC        F df1  df2            p
    ## Single_raters_absolute   ICC1 0.1696807 12.85264  50 2907 2.190401e-92
    ## Single_random_raters     ICC2 0.1703644 13.63776  50 2850 1.325333e-98
    ## Single_fixed_raters      ICC3 0.1789094 13.63776  50 2850 1.325333e-98
    ## Average_raters_absolute ICC1k 0.9221950 12.85264  50 2907 2.190401e-92
    ## Average_random_raters   ICC2k 0.9225419 13.63776  50 2850 1.325333e-98
    ## Average_fixed_raters    ICC3k 0.9266742 13.63776  50 2850 1.325333e-98
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.1207038   0.2459081
    ## Single_random_raters      0.1214064   0.2465420
    ## Single_fixed_raters       0.1279314   0.2576597
    ## Average_raters_absolute   0.8884159   0.9497833
    ## Average_random_raters     0.8890688   0.9499460
    ## Average_fixed_raters      0.8948313   0.9526768

### Average estimated parameters

Aggregate estimates within participants to get an overall average. How
do these differ within and between groups?

``` r
fit_amle_long_avg <- fit_amle_long[, .(value_mean = mean(value, na.rm = TRUE)), by = .(user_id_simple, clinical_status, variable)]
```

Test: Is there a significant difference in parameters between
participants of different clinical status?

Yes: alpha, lf, ter. No: tau, s.

``` r
for (param in fit_amle_long[, unique(variable)]) {
  model <- lmer(value ~ clinical_status + (1 | user_id) + (1 | lesson_id), data = fit_amle_long[variable == param])
  summary_model <- summary(model)
  print(paste("Parameter:", param))
  print(summary_model)
}
```

    ## [1] "Parameter: alpha"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: -557
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.0822 -0.5925  0.0084  0.6064  3.5638 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.009834 0.09917 
    ##  user_id   (Intercept) 0.016955 0.13021 
    ##  Residual              0.039010 0.19751 
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)         0.44549    0.02892 70.26385  15.406   <2e-16 ***
    ## clinical_statusMCI  0.10009    0.03800 48.98441   2.634   0.0113 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.602
    ## [1] "Parameter: tau"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 6243.1
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.7545 -0.6355  0.1027  0.6479  3.1016 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.04336  0.2082  
    ##  user_id   (Intercept) 0.26237  0.5122  
    ##  Residual              1.11622  1.0565  
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)        -2.81020    0.10700 53.12112  -26.26   <2e-16 ***
    ## clinical_statusMCI  0.06301    0.15354 49.20540    0.41    0.683    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.648
    ## [1] "Parameter: s"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: -4440
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.7862 -0.6068 -0.0442  0.5618  3.2404 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance  Std.Dev.
    ##  lesson_id (Intercept) 0.0005470 0.02339 
    ##  user_id   (Intercept) 0.0004478 0.02116 
    ##  Residual              0.0062825 0.07926 
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                     Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)         0.307922   0.005637 65.220406  54.626   <2e-16 ***
    ## clinical_statusMCI -0.002177   0.007065 41.960155  -0.308     0.76    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.548
    ## [1] "Parameter: lf"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 4972.7
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.6399 -0.4359 -0.1434  0.2005  5.7041 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.02856  0.1690  
    ##  user_id   (Intercept) 0.21026  0.4585  
    ##  Residual              0.59503  0.7714  
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)         0.65700    0.09386 51.67009   6.999 5.12e-09 ***
    ## clinical_statusMCI  0.40363    0.13482 48.10432   2.994  0.00434 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.655
    ## [1] "Parameter: ter"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 3607.6
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.6768 -0.3481  0.0429  0.3994  6.5970 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.01886  0.1373  
    ##  user_id   (Intercept) 0.04955  0.2226  
    ##  Residual              0.31051  0.5572  
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)         1.13472    0.04940 53.84482   22.97  < 2e-16 ***
    ## clinical_statusMCI  0.27161    0.06842 44.21413    3.97 0.000261 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.620

### Correlation

Are parameters correlated?

``` r
fit_amle_avg <- dcast(fit_amle_long_avg, user_id_simple + clinical_status ~ variable, value.var = "value_mean")

fit_amle_avg_plot <- fit_amle_avg[, .(
  alpha = mean(alpha, na.rm = TRUE),
  tau = mean(tau, na.rm = TRUE),
  s = mean(s, na.rm = TRUE),
  lf = mean(lf, na.rm = TRUE),
  ter = mean(ter, na.rm = TRUE),
  n_sessions = sum(!is.na(alpha))),
  by = .(user_id = user_id_simple, clinical_status)]

param_cols <- setdiff(names(fit_amle_avg_plot), c("user_id", "clinical_status", "n_sessions"))

fit_amle_avg_plot[, (param_cols) := lapply(.SD, scale), .SDcols = param_cols]

fit_amle_avg_plot <- melt(
  fit_amle_avg_plot,
  id.vars = c("user_id", "clinical_status"),
  measure.vars = param_cols,
  variable.name = "parameter",
  value.name = "z"
)


# Custom ticks per parameter (raw values)
ticks_list <- list(
  alpha = c(0.1, 0.3, 0.5, 0.7, 0.9),
  tau   = c(-4, -3, -2, -1),
  s     = c(0.1, 0.2, 0.3, 0.4),
  lf    = c(0, 1, 2, 3),
  ter   = c(.5, 1, 1.5, 2, 2.5)
)

# Map ticks to a long data.table for annotation
axis_anno <- rbindlist(
  lapply(names(ticks_list), function(var) {
    data.table(
      variable = var,
      q = ticks_list[[var]]
    )
  })
)

# Compute z-scores for plotting
param_stats <- fit_amle_long_avg[, .(mean = mean(value_mean), sd = sd(value_mean)), by = .(variable)]
axis_anno <- merge(axis_anno, param_stats, by = "variable")
axis_anno[, z := (q - mean) / sd]

# Map variables to x-axis positions
axis_anno[, x := as.numeric(factor(variable, levels = param_cols))]

# Compute group means
group_means <- fit_amle_avg_plot[
  , .(z = mean(z)), 
  by = .(clinical_status, parameter)
]

# Map x-axis positions
group_means[, x := as.numeric(factor(parameter, levels = param_cols))]

# Add jitter
fit_amle_avg_plot[, x_num := as.numeric(factor(fit_amle_avg_plot$parameter, levels = param_cols))]
fit_amle_avg_plot[, x_jitter := runif(1, -0.05, 0.05), by = user_id]
fit_amle_avg_plot[, x := x_num + x_jitter]



# Numeric x positions for parameters
param_x <- seq_along(param_cols)

# Max y value to position labels above the plot
y_max <- max(c(fit_amle_avg_plot$z, group_means$z))

# Create a dataframe for top labels
param_titles <- list( alpha = "Speed of Forgetting", tau = "Retrieval threshold", s = "Activation noise", lf = "Latency factor", ter = "Non-retrieval time" )
param_symbols <- list( alpha = "alpha", tau = "tau", s = "s", lf = "F", ter = "t[er]" )

param_label_df <- data.frame(
  x = param_x,
  title = unlist(param_titles),
  symbol = unlist(param_symbols)
)

# Plot

p_param_estimates <- ggplot() +

  # Vertical axes
  geom_vline(xintercept = 1:length(param_cols), color = "grey80") +

  # Fake tick marks
  geom_segment(
    data = axis_anno,
    aes(x = x - 0.05, xend = x + 0.05,y = z,yend = z),
    inherit.aes = FALSE,
    color = "grey50"
  ) +

  # Individual participant lines (behind)
  geom_line(
    data = fit_amle_avg_plot,
    aes(x = x, y = z, group = user_id, color = factor(clinical_status)),
    alpha = 0.25
  ) +

  # Individual points
  geom_point(
    data = fit_amle_avg_plot,
    aes(x = x, y = z, group = user_id, color = factor(clinical_status)),
    alpha = 0.25,
    size = 1
  ) +

  # Group mean lines (on top)
  geom_line(
    data = group_means,
    aes(x = x, y = z, color = factor(clinical_status)),
    linewidth = 1
  ) +

  # Group mean points
  geom_point(
    data = group_means,
    aes(x = x, y = z, color = factor(clinical_status)),
    size = 2
  ) +
  
  # Fake tick labels on the right
  geom_label(
    data = axis_anno,
    aes(x = x + 0.1, y = z, label = q),
    size = 3,
    hjust = 0,
    alpha = .5,
    label.size = 0,
    label.padding = unit(c(0,0,0,0), "lines"),
    inherit.aes = FALSE
  ) +
  
  geom_text(
    data = axis_anno,
    aes(x = x + 0.1, y = z, label = q),
    size = 3,
    hjust = 0,
    inherit.aes = FALSE
  ) +

  # X-axis: parameter names along top
  geom_label(
    data = param_label_df,
    aes(x = x, y = 6, label = paste0("atop(bold('", title, "'), ", symbol, ")")),
    parse = TRUE,
    size = 3.5,
    vjust = 1,
    label.size = 0,
    label.padding = unit(c(.5, 0, .5, 0), "lines")
  ) +
  
  scale_x_continuous(
    expand = expansion(mult = .1),
    breaks = param_x
  ) +

  theme_minimal() +
  # Remove y-axis
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    strip.text.x = element_text(size = 10)) +

  # Labels
  labs(
    x = "Parameter",
    color = "Clinical status"
  ) +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red))
```

    ## Warning: The `label.size` argument of `geom_label()` is deprecated as of ggplot2 3.5.0.
    ## ℹ Please use the `linewidth` argument instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.

``` r
# Not in paper: per-participant parameter estimates (exploratory)
ggsave(here("output", "parameter_estimates_by_participant.png"), width = 10, height = 5)
```

Plot parameters separately:

``` r
p_parameter_estimates <- ggplot(fit_amle_long_avg, aes(x = variable, y = value_mean, fill = clinical_status)) +
  facet_wrap(~ variable, scales = "free", ncol = 5, labeller = labeller(variable = c(alpha = "Speed of Forgetting", tau = "Retrieval threshold", s = "Activation noise", lf = "Latency factor", ter = "Non-retrieval time"))) +
  geom_boxplot(outlier.shape = NA, alpha = .5) +
  geom_point(position = position_jitterdodge(jitter.height = 0), alpha = .5, aes(colour = clinical_status)) +
  labs(x = "Parameter", y = "Estimated value", colour = "Clinical status", fill = "Clinical status") +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_fill_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_x_discrete(labels = c(alpha = expression(alpha), tau = expression(tau), s = expression(s), lf = expression(F), ter = expression(t[er]))) +
  theme(legend.position = "bottom")

# Paper Figure 9A: estimated parameters by clinical status (boxplots)
ggsave(here("output", "parameter_estimates_boxplot.png"), plot = p_parameter_estimates, width = 10, height = 6)
p_parameter_estimates
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-boxplot-1.png)<!-- -->
Pairwise correlations between parameters:

``` r
p_pairwise_corrs <- ggpairs(
  fit_amle_avg,
  columns = param_cols,
  columnLabels = unlist(param_titles),
  mapping = aes(color = clinical_status, fill = clinical_status, alpha = 0.5),
  legend = c(1,1),
  upper = list(continuous = wrap("cor", digits = 2))
) +
  scale_color_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_fill_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  guides(alpha = "none",
         fill = guide_legend(override.aes = list(alpha = .5))) +
  labs(color = "Clinical status", fill = "Clinical status") +
    theme(legend.position = "bottom")

p_pairwise_corrs
```

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"
    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-correlations-1.png)<!-- -->

Combine into one plot:

``` r
p_combined <- p_parameter_estimates / wrap_elements(ggmatrix_gtable(p_pairwise_corrs)) +
  plot_layout(ncol = 1, heights = c(1, 4)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))
```

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"
    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Ignoring unknown labels:
    ## • fill : "Clinical status"

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's fill values.

    ## Warning: No shared levels found between `names(values)` of the manual scale and
    ## the data's colour values.

``` r
# Paper Figure 9: parameter estimates by clinical status (A) + pairwise correlations (B)
ggsave(here("output", "parameter_estimates_correlations.png"), plot = p_combined, width = 10, height = 12)

p_combined
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-combined-1.png)<!-- -->

## Fact-level offsets

The relative difficulty of facts was estimated through offsets to a
participant’s $\alpha$ parameter: $\Delta\alpha$. In the fitting
process, these offsets were continually re-centered on zero, meaning
that the average $\Delta\alpha$ across facts for each participant is
(approximately) zero, which makes it easier to compare them across
participants.

The plot below shows the mean $\Delta\alpha$ for each fact (+/- 1 SD),
averaged across participants who encountered it, and organised by
lesson.

``` r
fit_delta_alpha_avg <- fit_delta_alpha[, .(.N, delta_alpha_mean = mean(delta_alpha), delta_alpha_sd = sd(delta_alpha)), by = .(lesson_id, fact_id)]

ggplot(fit_delta_alpha_avg, aes(x = reorder_within(fact_id, delta_alpha_mean, lesson_id, min), y = delta_alpha_mean)) +
  geom_errorbar(aes(ymin = delta_alpha_mean - delta_alpha_sd, ymax = delta_alpha_mean + delta_alpha_sd), width = 0, alpha = .25) +
  geom_point(aes(alpha = N)) +
  facet_wrap(~ lesson_id, scales = "free_x", ncol = 8) +
  scale_x_reordered() +
  geom_hline(yintercept = 0, colour = col_green) +
  labs(x = "Fact", y = expression(Delta~alpha)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/delta-alpha-by-lesson-1.png)<!-- -->

Some facts/lessons were only encountered by a small number of
participants:

``` r
ggplot(fit_delta_alpha_avg, aes(x = N)) +
  geom_histogram(binwidth = 1) +
  labs(x = "Number of participants who encountered the fact", y = "Count")
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/num-participants-per-fact-1.png)<!-- -->
To quantify the agreement in relative difficulty across participants, we
can compute the intra-class correlation (ICC) of $\Delta\alpha$
estimates across facts. However, the fact that each participant’s delta
alpha estimates center around zero means that a mixed-effects model with
random intercepts for participants cannot be fitted to the data, as the
participant-level variance is effectively zero. We’ll calculate the net
alpha value for each fact by adding the participant’s overall alpha to
the fact-specific delta alpha:

``` r
user_alpha <- fit_amle_long[variable == "alpha", .(user_id, session_id, user_alpha = value)]
fit_delta_alpha <- user_alpha[fit_delta_alpha, on = .(user_id, session_id)]
fit_delta_alpha[, fact_alpha := user_alpha + delta_alpha]

ggplot(fit_delta_alpha, aes(x = reorder_within(fact_id, fact_alpha, lesson_id, mean), y = fact_alpha)) +
  geom_line(aes(group = user_id), alpha = .2) +
  facet_wrap(~ lesson_id, scales = "free_x", ncol = 8) +
  scale_x_reordered() +
  labs(x = "Fact", y = expression(Delta~alpha)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
```

![](/Users/thomaswilschut/Documents/GitHub/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/delta-alpha-by-lesson-2-1.png)<!-- -->

Then, compute ICC from a mixed-effects regression model with random
intercepts for participants and lessons:

``` r
delta_alpha_wide <- dcast(fit_delta_alpha, fact_id ~ user_id, value.var = "fact_alpha")
icc_delta_alpha <- ICC(delta_alpha_wide[, -1, with = FALSE], lmer = TRUE)
```

    ## Warning in pf(FJ, dfJ, dfE, log.p = TRUE): pbeta(*, log.p=TRUE) ->
    ## bpser(a=21725, b=25, x=0.690568,...) underflow to -Inf

``` r
print(icc_delta_alpha$results)
```

    ##                          type       ICC        F df1   df2 p lower bound
    ## Single_raters_absolute   ICC1 0.2198842 15.37491 869 43500 0   0.2032753
    ## Single_random_raters     ICC2 0.2235499 22.23856 869 43450 0   0.1972462
    ## Single_fixed_raters      ICC3 0.2940059 22.23856 869 43450 0   0.2742002
    ## Average_raters_absolute ICC1k 0.9349590 15.37491 869 43500 0   0.9286330
    ## Average_random_raters   ICC2k 0.9362389 22.23856 869 43450 0   0.9260973
    ## Average_fixed_raters    ICC3k 0.9550331 22.23856 869 43450 0   0.9506595
    ##                         upper bound
    ## Single_raters_absolute    0.2381760
    ## Single_random_raters      0.2515829
    ## Single_fixed_raters       0.3155190
    ## Average_raters_absolute   0.9409841
    ## Average_random_raters     0.9448848
    ## Average_fixed_raters      0.9591987
