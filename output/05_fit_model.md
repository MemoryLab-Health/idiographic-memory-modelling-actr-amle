Example application: Modelling memory function in MCI and healthy ageing
================
Maarten van der Velde & Thomas Wilschut
Last updated: 2026-10-01

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
    - [Does excluding the top RT decile improve model fit? (Reviewer 2,
      comment
      6)](#does-excluding-the-top-rt-decile-improve-model-fit-reviewer-2-comment-6)
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
($\phi$) parameter estimated from the data. However, in that analysis,
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

    ## here() starts at /Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle

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
d <- fread(here("data", "processed", "hake2026.csv"))
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
    ## 1: user_01              HC session_0542 lesson_27 fact_792          1
    ## 2: user_01              HC session_0542 lesson_27 fact_792          2
    ## 3: user_01              HC session_0542 lesson_27 fact_431          1
    ## 4: user_01              HC session_0542 lesson_27 fact_431          2
    ## 5: user_01              HC session_0542 lesson_27 fact_337          1
    ## 6: user_01              HC session_0542 lesson_27 fact_726          1
    ##    start_time     rt correct session
    ##         <num>  <num>  <lgcl>   <int>
    ## 1: 1684541713 10.262    TRUE       1
    ## 2: 1684541724  2.614    TRUE       1
    ## 3: 1684541728  2.559    TRUE       1
    ## 4: 1684541731  1.721    TRUE       1
    ## 5: 1684541734  2.687    TRUE       1
    ## 6: 1684541737  2.356    TRUE       1

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
    ## 1:              HC     1248           27                98.30929
    ## 2:             MCI      900           24                62.04111
    ## 3:        Combined     2148           51                83.11313
    ##    mean_duration_min mean_reps_per_fact mean_n_facts mean_accuracy
    ##                <num>              <num>        <num>         <num>
    ## 1:          8.019310           6.859941     14.23397     0.9421364
    ## 2:          8.046848           6.135502     10.17111     0.8347306
    ## 3:          8.030849           6.556405     12.53166     0.8971340
    ##    mean_median_rt
    ##             <num>
    ## 1:       3.187801
    ## 2:       5.972667
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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/session-stats-correlation-1.png)<!-- -->

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
    ## 1:              HC     1245           27                92.73655
    ## 2:             MCI      816           24                52.17402
    ## 3:        Combined     2061           51                76.67686
    ##    mean_duration_min mean_reps_per_fact mean_n_facts mean_accuracy
    ##                <num>              <num>        <num>         <num>
    ## 1:          7.897532           6.908353    13.110040     0.9466660
    ## 2:          7.474282           6.343031     8.053922     0.8527335
    ## 3:          7.729957           6.684528    11.108200     0.9094758
    ##    mean_median_rt
    ##             <num>
    ## 1:       3.080505
    ## 2:       4.835191
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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/session-stats-filtered-1.png)<!-- -->

Summary statistics of filtered session data:

``` r
# Number of sessions after filtering:
session_stats_filtered[, .N, by = clinical_status]
```

    ##    clinical_status     N
    ##             <char> <int>
    ## 1:              HC  1245
    ## 2:             MCI   816

``` r
# Number of participants after filtering:
session_stats_filtered[, .(N = uniqueN(user_id)), by = clinical_status]
```

    ##    clinical_status     N
    ##             <char> <int>
    ## 1:              HC    27
    ## 2:             MCI    24

``` r
# Number of sessions per participant after filtering:
session_stats_filtered[, .N, by = .(user_id, clinical_status)][, .(sessions_mean = mean(N), sessions_sd = sd(N)), by = clinical_status]
```

    ##    clinical_status sessions_mean sessions_sd
    ##             <char>         <num>       <num>
    ## 1:              HC      46.11111     6.02133
    ## 2:             MCI      34.00000    13.77143

``` r
# Number of trials per session after filtering:
session_stats_filtered[, .(trials_mean = mean(trials), trials_sd = sd(trials)), by = clinical_status]
```

    ##    clinical_status trials_mean trials_sd
    ##             <char>       <num>     <num>
    ## 1:              HC    92.73655   33.7705
    ## 2:             MCI    52.17402   29.4282

``` r
# Number of repetitions per fact after filtering:
session_stats_filtered[, .(reps_per_fact_mean = mean(reps_per_fact), reps_per_fact_sd = sd(reps_per_fact)), by = clinical_status]
```

    ##    clinical_status reps_per_fact_mean reps_per_fact_sd
    ##             <char>              <num>            <num>
    ## 1:              HC           6.908353         1.563904
    ## 2:             MCI           6.343031         1.413173

``` r
# Number of facts per session after filtering:
session_stats_filtered[, .(facts_mean = mean(facts), facts_sd = sd(facts)), by = clinical_status]
```

    ##    clinical_status facts_mean facts_sd
    ##             <char>      <num>    <num>
    ## 1:              HC  13.110040 2.984677
    ## 2:             MCI   8.053922 4.291471

``` r
# Response accuracy after filtering:
session_stats_filtered[, .(accuracy_mean = mean(accuracy), accuracy_sd = sd(accuracy)), by = clinical_status]
```

    ##    clinical_status accuracy_mean accuracy_sd
    ##             <char>         <num>       <num>
    ## 1:              HC     0.9466660  0.08464752
    ## 2:             MCI     0.8527335  0.16089820

# Fit model

The data set is split into sessions, each of which is fitted separately.

``` r
d_sessions <- split(d_filtered, by = c("session_id"))
```

If desired, load previously fitted model results; otherwise, fit the
model to each session and save the results for future use. (Note:
fitting takes a while!)

``` r
delta_phi_file <- if (file.exists(here("data", "processed", "AMLE_delta_phi.csv"))) {
  here("data", "processed", "AMLE_delta_phi.csv")
} else {
  here("data", "processed", "AMLE_delta_alpha.csv")
}

if (refit_model == FALSE &&
    file.exists(here("data", "processed", "AMLE_fit.csv")) &&
    file.exists(delta_phi_file)) {
  fit_amle <- fread(here("data", "processed", "AMLE_fit.csv"))
  fit_delta_phi <- fread(delta_phi_file)
  # Backward compatibility: rename old column names if present
  if ("delta_alpha" %in% names(fit_delta_phi)) setnames(fit_delta_phi, "delta_alpha", "delta_phi")
  if ("alpha" %in% names(fit_amle)) setnames(fit_amle, "alpha", "phi")

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
  
  # Extract delta_phi estimates
  fit_delta_phi <- fit_amle[, .(delta_phi = delta_phi_est), by = .(user_id, clinical_status, session, session_id, lesson_id, session_trials, session_errors, session_accuracy)]
  fit_delta_phi[, delta_phi := lapply(delta_phi, function (delta_phi) {
    as.data.table(delta_phi, keep.rownames = "fact_id")
  })]
  fit_delta_phi <- unnest(fit_delta_phi, delta_phi) |> as.data.table()
  
  # Save results
  fwrite(fit_amle, here("data", "processed", "AMLE_fit.csv"))
  fwrite(fit_delta_phi, here("data", "processed", "AMLE_delta_phi.csv"))
  
}
```

# Evaluate model fit

## How well does the model fit capture the data?

How well does the fitted model capture the distribution of response
times? Here, we use the fitted parameters to predict trial-level
response times.

``` r
fitted_session_params <- fit_amle[, .(session_id, phi, tau, s, lf, ter)]
fitted_session_params <- fit_delta_phi[fitted_session_params, on = .(session_id)]
fitted_session_params[, fact_sof := phi + delta_phi]
fitted_session_params[, fact_id := gsub("d_", "", fact_id)]

d_preds <- future_map(d_sessions, function (d_session) {
  
  session_params <- fitted_session_params[session_id == d_session$session_id[1]]
  
  # If session params is empty or parameters are NA, return NULL
  if (nrow(session_params) == 0 || any(is.na(session_params[, .(delta_phi, phi, tau, s, lf, ter)]))) {
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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/decile-plots-1.png)<!-- -->

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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/correlation-rt-1.png)<!-- -->

Combined version of these RT fit plots:

``` r
p_rt_corr + p_decile_rt  +
  plot_layout(ncol = 1, heights = c(1, 8)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))
```

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/rt-fit-plots-1.png)<!-- -->

``` r
# Paper Figure 8: RT model fit, panels A (correlation distribution) + B (decile plots) combined
ggsave(here("output", "rt_fit_plots.png"), width = 10, height = 10)
```

### Does excluding the top RT decile improve model fit? (Reviewer 2, comment 6)

Figure 8B shows that the top decile of the RT distribution (each
participant’s slowest 10% of responses) tends to deviate from the
diagonal more than the rest of the distribution. The reviewer asked
whether excluding this decile would improve the correspondence between
predicted and observed RT. To check this without re-fitting the model,
we recompute each participant’s predicted-vs-observed RT correlation (as
in `cor_rt` above) after excluding their own top decile of observed RTs,
and compare it to the correlation on their full set of correct trials.

``` r
# Per-participant predicted-vs-observed RT correlation, with vs without their
# own top decile of observed RT excluded (uses the already-fitted parameters
# via pred_rt_correct computed above -- no re-fitting involved).
d_preds_correct[, rt_p90 := quantile(rt, 0.9), by = user_id]

fit_full_by_user <- d_preds_correct[, .(r_full = cor(pred_rt_correct, rt), n_full = .N), by = user_id]

fit_trimmed <- d_preds_correct[rt <= rt_p90, .(
  r_trimmed = cor(pred_rt_correct, rt),
  n_trimmed = .N
), by = user_id]

fit_comparison <- merge(fit_full_by_user, fit_trimmed, by = "user_id")
fit_comparison[, delta_r := r_trimmed - r_full]

fit_comparison[order(-delta_r)]
```

    ##     user_id    r_full n_full r_trimmed n_trimmed       delta_r
    ##      <char>     <num>  <int>     <num>     <int>         <num>
    ##  1: user_20 0.4071270     24 0.5656807        21  0.1585536633
    ##  2: user_51 0.6022564     48 0.6354212        43  0.0331647431
    ##  3: user_09 0.5780087   2361 0.6062179      2125  0.0282092105
    ##  4: user_28 0.6492239   1022 0.6594712       919  0.0102473501
    ##  5: user_14 0.6464465   2973 0.6552892      2675  0.0088427087
    ##  6: user_24 0.5338910   1060 0.5415171       954  0.0076261704
    ##  7: user_10 0.6086260   1789 0.6161275      1610  0.0075015250
    ##  8: user_04 0.6077687    916 0.6132272       824  0.0054584985
    ##  9: user_02 0.6374984   2920 0.6372197      2628 -0.0002787703
    ## 10: user_13 0.5885349    340 0.5881790       306 -0.0003559468
    ## 11: user_44 0.5866914    883 0.5837316       794 -0.0029598004
    ## 12: user_25 0.6098147   1301 0.6067291      1171 -0.0030856304
    ## 13: user_16 0.6654788    955 0.6542200       859 -0.0112588489
    ## 14: user_36 0.6423193   5443 0.6298134      4899 -0.0125058356
    ## 15: user_18 0.6852600   1165 0.6654976      1048 -0.0197624412
    ## 16: user_06 0.6587504   3646 0.6375895      3282 -0.0211609078
    ## 17: user_21 0.6886646   1718 0.6643735      1546 -0.0242910572
    ## 18: user_46 0.6141926   4660 0.5897921      4194 -0.0244004610
    ## 19: user_07 0.6103044   3524 0.5844271      3172 -0.0258773913
    ## 20: user_30 0.6344944   3097 0.6083222      2787 -0.0261721640
    ## 21: user_50 0.5752118   1612 0.5479448      1450 -0.0272670337
    ## 22: user_48 0.7071660   2321 0.6783753      2089 -0.0287907541
    ## 23: user_08 0.6529430   2574 0.6238154      2316 -0.0291275924
    ## 24: user_31 0.6333221   3340 0.6035081      3006 -0.0298140002
    ## 25: user_01 0.6983662   3716 0.6673614      3345 -0.0310047859
    ## 26: user_32 0.6310941   4628 0.5984536      4166 -0.0326404932
    ## 27: user_49 0.6681513   2570 0.6347846      2313 -0.0333667016
    ## 28: user_42 0.7089256    190 0.6720602       171 -0.0368653840
    ## 29: user_15 0.6301496   4078 0.5922436      3670 -0.0379059250
    ## 30: user_39 0.7131172   1852 0.6742220      1666 -0.0388952068
    ## 31: user_45 0.6484373    964 0.6084808       867 -0.0399564585
    ## 32: user_22 0.6802910   3246 0.6397725      2921 -0.0405184788
    ## 33: user_27 0.6367628   1370 0.5959586      1233 -0.0408041578
    ## 34: user_43 0.6375942   2976 0.5965186      2678 -0.0410755773
    ## 35: user_41 0.6073341    792 0.5659455       712 -0.0413886177
    ## 36: user_47 0.6499310   4035 0.6063503      3631 -0.0435807352
    ## 37: user_29 0.6427888   3150 0.5989362      2836 -0.0438525946
    ## 38: user_33 0.7209554   3770 0.6766442      3394 -0.0443111171
    ## 39: user_37 0.6418711   3550 0.5967924      3195 -0.0450787344
    ## 40: user_19 0.6284815    530 0.5783275       477 -0.0501539961
    ## 41: user_40 0.6163903   3795 0.5659153      3415 -0.0504750125
    ## 42: user_38 0.5924318   1356 0.5383840      1220 -0.0540478078
    ## 43: user_34 0.6476607   3967 0.5883818      3570 -0.0592788461
    ## 44: user_35 0.6281997   2417 0.5650161      2175 -0.0631836530
    ## 45: user_03 0.6809980    627 0.6136500       564 -0.0673480008
    ## 46: user_12 0.6507343   1343 0.5696723      1208 -0.0810619708
    ## 47: user_17 0.6597748   2460 0.5756796      2214 -0.0840951805
    ## 48: user_26 0.6612462   5056 0.5741138      4550 -0.0871324821
    ## 49: user_23 0.6996887   5106 0.6104032      4595 -0.0892854862
    ## 50: user_11 0.6426657   6068 0.5490136      5461 -0.0936521029
    ## 51: user_05 0.8463578     12 0.1451167        10 -0.7012410202
    ##     user_id    r_full n_full r_trimmed n_trimmed       delta_r

``` r
cat(sprintf("Mean r (full data):         %.3f\n", mean(fit_comparison$r_full)))
```

    ## Mean r (full data):         0.641

``` r
cat(sprintf("Mean r (top decile excl.):  %.3f\n", mean(fit_comparison$r_trimmed)))
```

    ## Mean r (top decile excl.):  0.600

``` r
cat(sprintf("Mean change (delta r):      %.3f\n", mean(fit_comparison$delta_r)))
```

    ## Mean change (delta r):      -0.041

``` r
cat(sprintf("Participants where fit improved after exclusion: %d / %d\n",
            sum(fit_comparison$delta_r > 0), nrow(fit_comparison)))
```

    ## Participants where fit improved after exclusion: 8 / 51

``` r
wilcox.test(fit_comparison$r_trimmed, fit_comparison$r_full, paired = TRUE)
```

    ## 
    ##  Wilcoxon signed rank test with continuity correction
    ## 
    ## data:  fit_comparison$r_trimmed and fit_comparison$r_full
    ## V = 129, p-value = 5.711e-07
    ## alternative hypothesis: true location shift is not equal to 0

Excluding the top RT decile does not improve the correspondence between
predicted and observed RT; on average it makes it slightly worse, and
the correlation decreases for the majority of participants. This is
consistent with a range-restriction effect: trimming based on the
criterion variable’s (RT’s) own extreme values reduces the variance
available to the correlation, rather than selectively removing poorly
predicted trials. We therefore do not think the deviation visible in the
top decile of Figure 8B reflects a correctable artefact of retaining too
much data, and did not pursue a full model re-fit excluding these
trials, since this check suggests it would not improve – and could
worsen – the correspondence between predicted and observed RT. The
residual tail deviation more likely reflects the heavier tail of the
observed RT distribution relative to the model’s assumed log-logistic
form (e.g., due to occasional attentional lapses).

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

    ## Warning: Removed 7 rows containing non-finite outside the scale range
    ## (`stat_density()`).

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/rt-distribution-full-1.png)<!-- -->

``` r
# Paper Figure 7: predicted vs observed RT density per participant (negative RTs = incorrect)
ggsave(here("output", "predicted_vs_observed_rt_density.png"), width = 10, height = 8.5)
```

    ## Warning: Removed 7 rows containing non-finite outside the scale range
    ## (`stat_density()`).

## Estimated parameters

### Parameter estimates per session

``` r
setorder(fit_amle, clinical_status, user_id, session, lesson_id)
fit_amle[, session_aligned := 1:.N, by = .(user_id)]
fit_amle_long <- melt(fit_amle, measure.vars = c("phi", "tau", "s", "lf", "ter"))
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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-over-time-1.png)<!-- -->

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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-1.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = facts, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Number of unique facts in session", y = "Parameter value", title = "Unique facts")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-2.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = session_errors, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Number of errors in session", y = "Parameter value", title = "Session errors") 
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-3.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = session_accuracy, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Session accuracy", y = "Parameter value", title = "Session accuracy")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-4.png)<!-- -->

``` r
ggplot(fit_amle_long, aes(x = median_rt, y = value)) +
  facet_wrap(clinical_status ~ variable, scales = "free_y", ncol = 5) +
  geom_point(alpha = .2) +
  geom_smooth(method = "gam") +
  labs(x = "Median RT in session (s)", y = "Parameter value", title = "Session median RT")
```

    ## `geom_smooth()` using formula = 'y ~ s(x, bs = "cs")'

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-session-stats-5.png)<!-- -->

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

    ## [1] "Parameter: phi"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.3542931 32.82403  50 2907 1.072372e-241
    ## Single_random_raters     ICC2 0.3558425 41.87085  50 2850 1.561222e-298
    ## Single_fixed_raters      ICC3 0.4133761 41.87085  50 2850 1.561222e-298
    ## Average_raters_absolute ICC1k 0.9695345 32.82403  50 2907 1.072372e-241
    ## Average_random_raters   ICC2k 0.9697337 41.87085  50 2850 1.561222e-298
    ## Average_fixed_raters    ICC3k 0.9761170 41.87085  50 2850 1.561222e-298
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.2739784   0.4622514
    ## Single_random_raters      0.2741194   0.4648007
    ## Single_fixed_raters       0.3270933   0.5241126
    ## Average_raters_absolute   0.9563079   0.9803371
    ## Average_random_raters     0.9563375   0.9805337
    ## Average_fixed_raters      0.9657455   0.9845864
    ## [1] "Parameter: tau"
    ##                          type       ICC        F df1  df2            p
    ## Single_raters_absolute   ICC1 0.1368192 10.19334  50 2907 2.700483e-70
    ## Single_random_raters     ICC2 0.1369941 10.33404  50 2850 2.420844e-71
    ## Single_fixed_raters      ICC3 0.1386229 10.33404  50 2850 2.420844e-71
    ## Average_raters_absolute ICC1k 0.9018967 10.19334  50 2907 2.700483e-70
    ## Average_random_raters   ICC2k 0.9020276 10.33404  50 2850 2.420844e-71
    ## Average_fixed_raters    ICC3k 0.9032325 10.33404  50 2850 2.420844e-71
    ##                         lower bound upper bound
    ## Single_raters_absolute   0.09527075   0.2032247
    ## Single_random_raters     0.09546071   0.2033761
    ## Single_fixed_raters      0.09664512   0.2056130
    ## Average_raters_absolute  0.85930508   0.9366825
    ## Average_random_raters    0.85957108   0.9367379
    ## Average_fixed_raters     0.86120962   0.9375479
    ## [1] "Parameter: s"
    ##                          type        ICC        F df1  df2            p
    ## Single_raters_absolute   ICC1 0.04414278 3.678518  50 2907 1.535741e-16
    ## Single_random_raters     ICC2 0.04528350 3.966058  50 2850 9.950082e-19
    ## Single_fixed_raters      ICC3 0.04865098 3.966058  50 2850 9.950082e-19
    ## Average_raters_absolute ICC1k 0.72815141 3.678518  50 2907 1.535741e-16
    ## Average_random_raters   ICC2k 0.73340574 3.966058  50 2850 9.950082e-19
    ## Average_fixed_raters    ICC3k 0.74786050 3.966058  50 2850 9.950082e-19
    ##                         lower bound upper bound
    ## Single_raters_absolute   0.02627292  0.07495194
    ## Single_random_raters     0.02743643  0.07604736
    ## Single_fixed_raters      0.02953588  0.08148325
    ## Average_raters_absolute  0.61012806  0.82454431
    ## Average_random_raters    0.62066675  0.82680325
    ## Average_fixed_raters     0.63836493  0.83727356
    ## [1] "Parameter: lf"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.3865688 37.55014  50 2907 8.501447e-273
    ## Single_random_raters     ICC2 0.3869407 39.83471  50 2850 5.879037e-286
    ## Single_fixed_raters      ICC3 0.4010412 39.83471  50 2850 5.879037e-286
    ## Average_raters_absolute ICC1k 0.9733689 37.55014  50 2907 8.501447e-273
    ## Average_random_raters   ICC2k 0.9734096 39.83471  50 2850 5.879037e-286
    ## Average_fixed_raters    ICC3k 0.9748963 39.83471  50 2850 5.879037e-286
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.3027408   0.4964389
    ## Single_random_raters      0.3030866   0.4967933
    ## Single_fixed_raters       0.3158246   0.5114663
    ## Average_raters_absolute   0.9618070   0.9828119
    ## Average_random_raters     0.9618672   0.9828358
    ## Average_fixed_raters      0.9639946   0.9837985
    ## [1] "Parameter: ter"
    ##                          type       ICC        F df1  df2             p
    ## Single_raters_absolute   ICC1 0.1775404 13.52018  50 2907  7.263695e-98
    ## Single_random_raters     ICC2 0.1781657 14.28673  50 2850 7.085058e-104
    ## Single_fixed_raters      ICC3 0.1863843 14.28673  50 2850 7.085058e-104
    ## Average_raters_absolute ICC1k 0.9260365 13.52018  50 2907  7.263695e-98
    ## Average_random_raters   ICC2k 0.9263289 14.28673  50 2850 7.085058e-104
    ## Average_fixed_raters    ICC3k 0.9300050 14.28673  50 2850 7.085058e-104
    ##                         lower bound upper bound
    ## Single_raters_absolute    0.1268651   0.2559141
    ## Single_random_raters      0.1275116   0.2564884
    ## Single_fixed_raters       0.1338242   0.2670922
    ## Average_raters_absolute   0.8939252   0.9522627
    ## Average_random_raters     0.8944761   0.9523995
    ## Average_fixed_raters      0.8996085   0.9548264

### Average estimated parameters

Aggregate estimates within participants to get an overall average. How
do these differ within and between groups?

``` r
fit_amle_long_avg <- fit_amle_long[, .(value_mean = mean(value, na.rm = TRUE)), by = .(user_id_simple, clinical_status, variable)]
```

Test: Is there a significant difference in parameters between
participants of different clinical status?

Yes: phi, lf, ter. No: tau, s.

``` r
for (param in fit_amle_long[, unique(variable)]) {
  model <- lmer(value ~ clinical_status + (1 | user_id) + (1 | lesson_id), data = fit_amle_long[variable == param])
  summary_model <- summary(model)
  print(paste("Parameter:", param))
  print(summary_model)
}
```

    ## [1] "Parameter: phi"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: -2255.9
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.7731 -0.5145  0.0588  0.5711  4.8737 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.004656 0.06823 
    ##  user_id   (Intercept) 0.011197 0.10582 
    ##  Residual              0.016896 0.12999 
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)         0.33378    0.02262 64.70634  14.753   <2e-16 ***
    ## clinical_statusMCI  0.05926    0.03050 48.71138   1.943   0.0579 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.622
    ## [1] "Parameter: tau"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 1462.9
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -1.6908 -0.4357 -0.2300 -0.0554  5.5142 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.001534 0.03917 
    ##  user_id   (Intercept) 0.017490 0.13225 
    ##  Residual              0.111825 0.33440 
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)        -1.85616    0.02773 49.40908 -66.937   <2e-16 ***
    ## clinical_statusMCI  0.06003    0.04070 49.31438   1.475    0.147    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.656
    ## [1] "Parameter: s"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: -4525.8
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.9917 -0.6217 -0.0680  0.5644  3.0984 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance  Std.Dev.
    ##  lesson_id (Intercept) 0.0004744 0.02178 
    ##  user_id   (Intercept) 0.0002878 0.01696 
    ##  Residual              0.0060791 0.07797 
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                     Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)         0.289396   0.004935 68.710244  58.640   <2e-16 ***
    ## clinical_statusMCI  0.009119   0.006033 44.396338   1.512    0.138    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.528
    ## [1] "Parameter: lf"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 4505.7
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.2252 -0.4193 -0.1252  0.1772  5.9236 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.02851  0.1688  
    ##  user_id   (Intercept) 0.24813  0.4981  
    ##  Residual              0.46811  0.6842  
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)          0.9005     0.1005 50.8905   8.958 4.88e-12 ***
    ## clinical_statusMCI   0.5343     0.1444 47.6637   3.699 0.000559 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.660
    ## [1] "Parameter: ter"
    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ clinical_status + (1 | user_id) + (1 | lesson_id)
    ##    Data: fit_amle_long[variable == param]
    ## 
    ## REML criterion at convergence: 3791.4
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3508 -0.3877  0.0749  0.4269  6.5280 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  lesson_id (Intercept) 0.01910  0.1382  
    ##  user_id   (Intercept) 0.06319  0.2514  
    ##  Residual              0.33887  0.5821  
    ## Number of obs: 2061, groups:  lesson_id, 58; user_id, 51
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error       df t value Pr(>|t|)    
    ## (Intercept)         1.01385    0.05453 51.32164  18.592  < 2e-16 ***
    ## clinical_statusMCI  0.23479    0.07642 43.70536   3.072  0.00365 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## clncl_stMCI -0.630

### Correlation

Are parameters correlated?

``` r
fit_amle_avg <- dcast(fit_amle_long_avg, user_id_simple + clinical_status ~ variable, value.var = "value_mean")

fit_amle_avg_plot <- fit_amle_avg[, .(
  phi = mean(phi, na.rm = TRUE),
  tau = mean(tau, na.rm = TRUE),
  s = mean(s, na.rm = TRUE),
  lf = mean(lf, na.rm = TRUE),
  ter = mean(ter, na.rm = TRUE),
  n_sessions = sum(!is.na(phi))),
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
  phi = c(0.1, 0.3, 0.5, 0.7, 0.9),
  tau = c(-4, -3, -2, -1),
  s   = c(0.1, 0.2, 0.3, 0.4),
  lf  = c(0, 1, 2, 3),
  ter = c(.5, 1, 1.5, 2, 2.5)
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
param_titles <- list( phi = "Speed of Forgetting", tau = "Retrieval threshold", s = "Activation noise", lf = "Latency factor", ter = "Non-retrieval time" )
param_symbols <- list( phi = "phi", tau = "tau", s = "s", lf = "F", ter = "t[er]" )

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
    ## This warning is displayed once per session.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.

``` r
# Not in paper: per-participant parameter estimates (exploratory)
ggsave(here("output", "parameter_estimates_by_participant.png"), width = 10, height = 5)
```

Plot parameters separately:

``` r
p_parameter_estimates <- ggplot(fit_amle_long_avg, aes(x = variable, y = value_mean, fill = clinical_status)) +
  facet_wrap(~ variable, scales = "free", ncol = 5, labeller = labeller(variable = c(phi = "Speed of Forgetting", tau = "Retrieval threshold", s = "Activation noise", lf = "Latency factor", ter = "Non-retrieval time"))) +
  geom_boxplot(outlier.shape = NA, alpha = .5) +
  geom_point(position = position_jitterdodge(jitter.height = 0), alpha = .5, aes(colour = clinical_status)) +
  labs(x = "Parameter", y = "Estimated value", colour = "Clinical status", fill = "Clinical status") +
  scale_colour_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  scale_fill_manual(values = c("HC" = col_blue, "MCI" = col_red)) +
  theme(legend.position = "bottom",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Paper Figure 9A: estimated parameters by clinical status (boxplots), Reviewer 2 minor comment 8 (enlarged x-labels)
ggsave(here("output", "parameter_estimates_boxplot.png"), plot = p_parameter_estimates, width = 10, height = 6)
p_parameter_estimates
```

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-boxplot-1.png)<!-- -->
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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-correlations-1.png)<!-- -->

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

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/parameter-estimates-combined-1.png)<!-- -->

## Fact-level offsets

The relative difficulty of facts was estimated through offsets to a
participant’s $\phi$ parameter: $\Delta\phi$. In the fitting process,
these offsets were continually re-centered on zero, meaning that the
average $\Delta\phi$ across facts for each participant is
(approximately) zero, which makes it easier to compare them across
participants.

The plot below shows the mean $\Delta\phi$ for each fact (+/- 1 SD),
averaged across participants who encountered it, and organised by
lesson.

``` r
fit_delta_phi_avg <- fit_delta_phi[, .(.N, delta_phi_mean = mean(delta_phi), delta_phi_sd = sd(delta_phi)), by = .(lesson_id, fact_id)]

ggplot(fit_delta_phi_avg, aes(x = reorder_within(fact_id, delta_phi_mean, lesson_id, min), y = delta_phi_mean)) +
  geom_errorbar(aes(ymin = delta_phi_mean - delta_phi_sd, ymax = delta_phi_mean + delta_phi_sd), width = 0, alpha = .25) +
  geom_point(aes(alpha = N)) +
  facet_wrap(~ lesson_id, scales = "free_x", ncol = 8) +
  scale_x_reordered() +
  geom_hline(yintercept = 0, colour = col_green) +
  labs(x = "Fact", y = expression(Delta~phi)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
```

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/delta-phi-by-lesson-1.png)<!-- -->

Some facts/lessons were only encountered by a small number of
participants:

``` r
ggplot(fit_delta_phi_avg, aes(x = N)) +
  geom_histogram(binwidth = 1) +
  labs(x = "Number of participants who encountered the fact", y = "Count")
```

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/num-participants-per-fact-1.png)<!-- -->
To quantify the agreement in relative difficulty across participants, we
can compute the intra-class correlation (ICC) of $\Delta\phi$ estimates
across facts. However, the fact that each participant’s delta phi
estimates center around zero means that a mixed-effects model with
random intercepts for participants cannot be fitted to the data, as the
participant-level variance is effectively zero. We’ll calculate the net
phi value for each fact by adding the participant’s overall phi to the
fact-specific delta phi:

``` r
user_phi <- fit_amle_long[variable == "phi", .(user_id, session_id, user_phi = value)]
fit_delta_phi <- user_phi[fit_delta_phi, on = .(user_id, session_id)]
fit_delta_phi[, fact_phi := user_phi + delta_phi]

ggplot(fit_delta_phi, aes(x = reorder_within(fact_id, fact_phi, lesson_id, mean), y = fact_phi)) +
  geom_line(aes(group = user_id), alpha = .2) +
  facet_wrap(~ lesson_id, scales = "free_x", ncol = 8) +
  scale_x_reordered() +
  labs(x = "Fact", y = expression(Delta~phi)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())
```

![](/Users/maarten/Documents/projects/PCL/amle-gh/idiographic-memory-modelling-actr-amle/output/05_fit_model_files/figure-gfm/delta-phi-by-lesson-2-1.png)<!-- -->

Then, compute ICC from a mixed-effects regression model with random
intercepts for participants and lessons:

``` r
delta_phi_wide <- dcast(fit_delta_phi, fact_id ~ user_id, value.var = "fact_phi")
icc_delta_phi <- ICC(delta_phi_wide[, -1, with = FALSE], lmer = TRUE)
```

    ## Warning in pf(FJ, dfJ, dfE, log.p = TRUE): pbeta(*, log.p=TRUE) ->
    ## bpser(a=21725, b=25, x=0.722377,...) underflow to -Inf

``` r
print(icc_delta_phi$results)
```

    ##                          type       ICC        F df1   df2 p lower bound
    ## Single_raters_absolute   ICC1 0.2276406 16.03144 869 43500 0   0.2106535
    ## Single_random_raters     ICC2 0.2308647 22.16711 869 43450 0   0.2055881
    ## Single_fixed_raters      ICC3 0.2933069 22.16711 869 43450 0   0.2735269
    ## Average_raters_absolute ICC1k 0.9376226 16.03144 869 43500 0   0.9315556
    ## Average_random_raters   ICC2k 0.9386813 22.16711 869 43450 0   0.9295697
    ## Average_fixed_raters    ICC3k 0.9548881 22.16711 869 43450 0   0.9505004
    ##                         upper bound
    ## Single_raters_absolute    0.2463219
    ## Single_random_raters      0.2578422
    ## Single_fixed_raters       0.3147948
    ## Average_raters_absolute   0.9434010
    ## Average_random_raters     0.9465770
    ## Average_fixed_raters      0.9590672
