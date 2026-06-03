# =============================================================================
# MLE DATA SIMULATION 
#
# Generates synthetic retrieval practice data under the ACT-R memory model
# 
#
# The likelihood functions used for generating data are copied VERBATIM from
# fit_session_amle_df() in mle_greedy_selection.R, which is the reference
# implementation matching the paper.  Key specification choices:
#
#   correct_likelihood:  1 / (1 + exp(-(a - tau) / s))
#   rt_likelihood:       log-logistic with alpha = F*exp(-a), beta = 1/s
#                        (paper eq. 10: beta = 1/s, NOT sqrt(3)/(pi*s))
#   calculate_activation_memory:  adaptive decay with c = 0.25
#   RT for incorrect responses:   uses tau as activation (paper p.5)
#   Study trial:  first encounter per fact is recorded but not evaluated
#                 (matches includes_study_trials = TRUE in the fit)
#   Fact offsets:  sum-to-zero per learner (matches the AMLE E-step constraint)
#   Units:  seconds throughout (no ms conversion)
#
# NOTE ON beta PARAMETRISATION:
#   The paper (eq. 10) and fit_session_amle_df both use beta = 1/s.
#   The older sof_mle_em_model_funs.R uses beta = sqrt(3)/(pi*s).
#   This simulation follows the paper and fit_session_amle_df.
# Output columns match what fit_session_amle_df expects:
#   user_id, fact_id, start_time, rt, correct (logical)
#   + true parameter columns for recovery analysis
# =============================================================================

library(data.table)
library(furrr)
library(Rcpp)

# Compile and load C++ likelihood functions on the main process.
# Store the absolute path for worker initialisation.
.actr_ll_cpp_path <- here::here("R", "actr_ll.cpp")
Rcpp::sourceCpp(.actr_ll_cpp_path)

# Helper: compile C++ on a furrr worker (called once per worker).
# This should be called after plan() is set, e.g.:
#   init_workers()
# It ensures every worker has the compiled C++ functions available.
init_workers <- function() {
  cpp_path <- .actr_ll_cpp_path
  furrr::future_map(seq_len(future::nbrOfWorkers()), function(i) {
    Rcpp::sourceCpp(cpp_path)
    NULL
  }, .options = furrr::furrr_options(seed = NULL))
  invisible(NULL)
}

# =============================================================================
# MODEL FUNCTIONS — verbatim from fit_session_amle_df (mle_greedy_selection.R)
# =============================================================================

#' Trace odds (power-law decay), capped to avoid numerical overflow.
calculate_trace_odds <- function(t, te, d, cap = 10) {
  if (t > te) min(cap, (t - te)^-d) else 0
}

#' Base-level activation of a memory chunk at time t, given prior traces and
#' Speed of Forgetting (sof).  Uses adaptive decay (c = 0.25).
#' This is the version from fit_session_amle_df, which does NOT floor at 1e-20.
calculate_activation_memory <- function(t, traces, sof, c = 0.25, include_t = FALSE) {
  traces <- if (include_t) traces[traces <= t] else traces[traces < t]
  previous <- c()
  dis      <- c()
  for (i in seq_along(traces)) {
    new_trace <- traces[i]
    if (!length(previous)) {
      previous <- c(new_trace)
      dis      <- c(sof)
    } else {
      odds_t_trace <- 0
      for (j in seq_along(previous)) {
        odds_t_trace <- odds_t_trace + calculate_trace_odds(new_trace, previous[j], dis[j])
      }
      d_new    <- c * odds_t_trace + sof
      previous <- c(previous, new_trace)
      dis      <- c(dis, d_new)
    }
  }
  memory_odds <- 0
  for (i in seq_along(traces)) {
    memory_odds <- memory_odds + calculate_trace_odds(t, traces[i], dis[i])
  }
  log(memory_odds)
}

#' Probability of correct recall (logistic).
correct_likelihood <- function(a, tau, s) {
  1 / (1 + exp(-(a - tau) / s))
}

#' Probability density of reaction time (log-logistic).
#' Paper eq. 10:  alpha = F * exp(-a),  beta = 1/s
rt_likelihood <- function(rt, a, s, lf, t0) {
  alpha <- exp(-a) * lf
  beta  <- 1 / s
  t     <- rt - t0
  t[t < 0] <- 0
  (beta / alpha) * (t / alpha)^(beta - 1) / (1 + (t / alpha)^beta)^2
}

# =============================================================================
# SAMPLING HELPERS
# =============================================================================

#' Draw accuracy from a Bernoulli with p = correct_likelihood(a, tau, s).
sample_correct <- function(activation, tau, s) {
  p_correct <- correct_likelihood(activation, tau, s)
  as.logical(rbinom(1, 1, p_correct))
}

#' Draw RT from the log-logistic implied by rt_likelihood, via inverse CDF.
#'   alpha_param = lf * exp(-a)   (scale, matching rt_likelihood)
#'   beta_param  = 1 / s          (shape, matching rt_likelihood & paper eq. 10)
#' Inverse CDF:  t = alpha * (u / (1-u))^(1/beta)
#' Total RT = ter + t
#' Reject-resample if outside [rt_min, rt_max].
sample_rt <- function(activation, s, lf, ter,
                      rt_min = 0.0, rt_max = 20,
                      max_tries = 10000) {
  alpha_param <- exp(-activation) * lf
  beta_param  <- 1 / s
  
  for (k in seq_len(max_tries)) {
    u        <- runif(1)
    t_sample <- alpha_param * ((u / (1 - u))^(1 / beta_param))
    rt       <- ter + t_sample
    if (rt >= rt_min && rt <= rt_max) return(rt)
  }
  
  warning("sample_rt(): exceeded max_tries; clamping last draw.")
  return(min(max(rt, rt_min), rt_max))
}

# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

simulate_fact_learner_data <- function(
    # --- Dataset characteristics ---
  n_learners     = 50,
  n_facts        = 10,
  n_repetitions  = 10,       # test trials per fact (total encounters = n_rep + 1 study)
  fixed_delay    = 0.5,      # seconds between trials
  delay_jitter   = 0,        # uniform jitter (+/-) on delay
  
  # --- Parameter distributions (truncated normals) ---
  # Defaults based on the paper (Van der Velde, Wilschut et al., 2026):
  #   - Means match Algorithm 1 init / Figure 1 defaults
  #   - SDs chosen to give realistic spread around those means
  #   - Bounds reflect fit_session_amle_df constraints & observed data range
  sof_mean = 0.3,  sof_sd = 0.05, sof_min = 0.15, sof_max = 0.5,
  sof_fact_sd = 0.05,        # SD of fact-level offsets (delta_alpha)
  
  tau_mean = -0.8, tau_sd = 0.15, tau_min = -1.5, tau_max = 0.0,
  s_mean   = 0.1,  s_sd  = 0.03,  s_min  = 0.02,  s_max  = 0.25,
  lf_mean  = 1.0,  lf_sd = 0.2,   lf_min = 0.2,   lf_max = 2.0,
  ter_mean = 0.75, ter_sd = 0.3,  ter_min = 0.1,  ter_max = 2.0,
  
  # --- RT bounds (reject-resample outside these) ---
  rt_min = 0.2,
  rt_max = 20,
  
  # --- Random seed ---
  seed = 999
) {
  set.seed(seed)
  
  pb <- txtProgressBar(min = 0, max = n_learners, style = 3)
  on.exit(close(pb))
  
  # --- Draw true learner-level parameters (truncated normal) -----------------
  true_alpha <- pmin(pmax(rnorm(n_learners, sof_mean, sof_sd), sof_min), sof_max)
  true_tau   <- pmin(pmax(rnorm(n_learners, tau_mean, tau_sd), tau_min), tau_max)
  true_s     <- pmin(pmax(rnorm(n_learners, s_mean, s_sd),     s_min),   s_max)
  true_lf    <- pmin(pmax(rnorm(n_learners, lf_mean, lf_sd),   lf_min),  lf_max)
  true_ter   <- pmin(pmax(rnorm(n_learners, ter_mean, ter_sd), ter_min), ter_max)
  
  # --- Draw fact offsets & impose sum-to-zero per learner --------------------
  # This matches the constraint in the AMLE E-step:
  #   delta_alpha <- delta_alpha - mean(delta_alpha)
  true_fact_offset <- matrix(rnorm(n_learners * n_facts, 0, sof_fact_sd),
                             nrow = n_learners, ncol = n_facts)
  for (i in seq_len(n_learners)) {
    true_fact_offset[i, ] <- true_fact_offset[i, ] - mean(true_fact_offset[i, ])
  }
  
  # --- Pre-allocate results list ---------------------------------------------
  n_total <- n_learners * n_facts * (n_repetitions + 1L)  # study + test trials
  res     <- vector("list", n_total)
  row_idx <- 0L
  
  # --- Simulation loop -------------------------------------------------------
  for (uid in seq_len(n_learners)) {
    
    alpha <- true_alpha[uid]
    tau   <- true_tau[uid]
    s     <- true_s[uid]
    lf    <- true_lf[uid]
    ter   <- true_ter[uid]
    
    # Create randomised trial order:
    # n_repetitions + 1 encounters per fact (1 study + n_repetitions test)
    trials <- expand.grid(
      factId    = seq_len(n_facts),
      encounter = seq_len(n_repetitions + 1L)
    )
    trials <- trials[sample(nrow(trials)), ]
    
    # Per-fact trace history
    fact_traces <- vector("list", n_facts)
    
    current_time <- 0
    
    for (i in seq_len(nrow(trials))) {
      fid         <- trials$factId[i]
      fact_offset <- true_fact_offset[uid, fid]
      total_sof   <- alpha + fact_offset
      traces      <- fact_traces[[fid]]
      n_prev      <- length(traces)
      
      if (n_prev == 0) {
        # ---- Study trial (first encounter) ----
        # No accuracy/RT generated; the fit skips this trial
        # (includes_study_trials = TRUE -> start_index = 2).
        # Record the trace so subsequent encounters can compute activation.
        fact_traces[[fid]] <- current_time
        
        # Store study trial in output (rt and correct are NA)
        row_idx <- row_idx + 1L
        res[[row_idx]] <- list(
          user_id          = uid,
          fact_id          = fid,
          start_time       = current_time,
          rt               = NA_real_,
          correct          = NA,
          is_study         = TRUE,
          true_alpha       = alpha,
          true_fact_offset = fact_offset,
          true_total_sof   = total_sof,
          true_tau         = tau,
          true_s           = s,
          true_lf          = lf,
          true_ter         = ter
        )
        
        # Advance clock by a fixed study duration
        current_time <- current_time + fixed_delay +
          runif(1, -delay_jitter, delay_jitter)
        next
      }
      
      # ---- Test trial (subsequent encounters) ----
      # Activation is computed from all prior traces for this fact,
      # exactly as in the fit:
      #   a <- calculate_activation_memory(traces[i], traces[1:(i-1)], sof)
      activation <- calculate_activation_memory(
        t      = current_time,
        traces = traces,
        sof    = total_sof
      )
      
      # Sample accuracy
      correct <- sample_correct(activation, tau, s)
      
      # Sample RT: for incorrect responses, use tau as activation
      # (matches fit: rt_likelihood(rt[i], tau, s, lf, ter))
      a_for_rt     <- if (correct) activation else tau
      reactionTime <- sample_rt(a_for_rt, s, lf, ter, rt_min, rt_max)
      
      # Store result
      row_idx <- row_idx + 1L
      res[[row_idx]] <- list(
        user_id          = uid,
        fact_id          = fid,
        start_time       = current_time,
        rt               = reactionTime,
        correct          = correct,
        is_study         = FALSE,
        true_alpha       = alpha,
        true_fact_offset = fact_offset,
        true_total_sof   = total_sof,
        true_tau         = tau,
        true_s           = s,
        true_lf          = lf,
        true_ter         = ter
      )
      
      # Update fact history and advance clock
      fact_traces[[fid]] <- c(traces, current_time)
      current_time <- current_time + reactionTime + fixed_delay +
        runif(1, -delay_jitter, delay_jitter)
    }
    
    setTxtProgressBar(pb, uid)
  }
  
  # --- Assemble data.table ---------------------------------------------------
  data <- rbindlist(res[seq_len(row_idx)])
  
  # Add trial and repetition indices
  setorder(data, user_id, start_time)
  data[, trial := seq_len(.N) - 1L, by = user_id]
  data[, rep   := seq_len(.N) - 1L, by = .(user_id, fact_id)]
  
  return(data[])
}

# =============================================================================
# FIT FUNCTION — self-contained AMLE
#
# Based on fit_session_amle from the Rmd notebook, adapted to:
#   - Use the same model functions defined above in this file
#   - Accept all 5 parameters as free (configurable via free_params)
#   - Work directly on output from simulate_fact_learner_data()
#   - Not require columns like session_id, lesson_id, clinical_status
#
# Input: a data.table for ONE learner with columns:
#   user_id, fact_id, start_time, rt, correct, is_study
#
# Study trials (is_study == TRUE) are kept as the first trace per fact but
# are not evaluated in the likelihood (includes_study_trials = TRUE skips
# the first encounter).  Their rt/correct values are ignored.
# =============================================================================

fit_learner_amle <- function(
    d_learner,
    free_params = c("alpha", "tau", "s", "F", "ter"),
    init   = list(alpha = 0.3, tau = -0.8, s = 0.1, F = 1, ter = 0.3),
    max_iter    = 100,
    ll_epsilon  = 1e-5,
    rt_range    = c(0.1, 15),
    min_trials  = 3,
    bounds = list(
      alpha = c(0, 1),
      tau   = c(-2, 0.5),
      s     = c(1e-2, 0.5),
      F     = c(0.1, 5),
      ter   = c(0, 5)
    ),
    delta_upper = 1,
    includes_study_trials = TRUE
) {
  
  # --- Local likelihood functions ----------------------------------------------
  # The heavy computation (activation + likelihood) is handled by compiled C++
  # functions from actr_ll.cpp (loaded via Rcpp::sourceCpp at the top of the
  # script).  We only define thin R wrappers here.
  
  .objective_fun <- function(data, sof, tau, s, lf, ter) {
    val <- tryCatch(
      session_ll_cpp(data, sof, tau, s, lf, ter,
                     start_index = if (includes_study_trials) 2L else 1L),
      error = function(e) NA_real_
    )
    if (!is.finite(val)) val <- -1e10
    val
  }
  
  .damp_and_shrink <- function(old, new, rho = 0.5, lambda = 0.1) {
    damped <- (1 - rho) * old + rho * new
    (damped + lambda * old) / (1 + lambda)
  }
  
  # --- Preprocessing ----------------------------------------------------------
  d <- copy(d_learner)
  
  # For study trials, set dummy rt/correct so they survive filtering
  # (the likelihood skips them via start_index = 2)
  if ("is_study" %in% names(d)) {
    d[is_study == TRUE, `:=`(rt = 1, correct = TRUE)]
  }
  
  # Remove fact sequences containing fast or slow responses
  too_fast_or_slow <- d[!is.na(rt) & (rt < rt_range[1] | rt > rt_range[2])]
  if (nrow(too_fast_or_slow) > 0) {
    d <- d[!too_fast_or_slow[, .(user_id, fact_id)], on = .(user_id, fact_id)]
  }
  
  # Remove fact sequences with fewer than min_trials
  too_few <- d[, .N, by = .(user_id, fact_id)][N < min_trials]
  if (nrow(too_few) > 0) {
    d <- d[!too_few[, .(user_id, fact_id)], on = .(user_id, fact_id)]
  }
  
  if (nrow(d) == 0) return(NULL)
  
  # --- Format as fact-sequence lists ------------------------------------------
  d_facts <- split(d, by = "fact_id")
  d_facts_list <- lapply(d_facts, function(x) {
    list(
      N      = nrow(x),
      traces = x$start_time,
      correct = x$correct,
      rt     = x$rt
    )
  })
  
  # --- Initialise parameters --------------------------------------------------
  params <- init
  nF     <- length(d_facts_list)
  
  delta_alpha <- rep(0, nF)
  names(delta_alpha) <- paste0("d_", names(d_facts_list))
  
  uid <- d$user_id[1]
  
  # --- Iteration history ------------------------------------------------------
  fits <- data.table(
    user_id   = uid,
    iteration = 0L,
    delta_alpha = list(delta_alpha),
    alpha = params$alpha,
    tau   = params$tau,
    s     = params$s,
    lf    = params$F,
    ter   = params$ter,
    ll    = .objective_fun(d_facts_list,
                           sof = params$alpha + delta_alpha,
                           tau = params$tau, s = params$s,
                           lf = params$F, ter = params$ter)
  )
  
  # --- AMLE loop --------------------------------------------------------------
  alpha <- params$alpha
  tau   <- params$tau
  s     <- params$s
  lf    <- params$F
  ter   <- params$ter
  
  for (it in seq_len(max_iter)) {
    
    # ---- E-step: fact-level offsets ------------------------------------------
    fit_e <- optim(
      par = delta_alpha,
      fn = function(p) {
        -.objective_fun(d_facts_list,
                        sof = alpha + p,
                        tau = tau, s = s, lf = lf, ter = ter)
      },
      method = "L-BFGS-B",
      lower = rep(-alpha, nF),
      upper = rep(delta_upper, nF),
      control = list(maxit = 80, factr = 1e7, pgtol = 1e-3)
    )
    
    delta_alpha_new    <- fit_e$par
    delta_alpha_damped <- .damp_and_shrink(delta_alpha, delta_alpha_new)
    delta_alpha        <- delta_alpha_damped - mean(delta_alpha_damped)
    
    # ---- M-step: participant-level parameters --------------------------------
    par_names <- intersect(free_params, c("alpha", "tau", "s", "F", "ter"))
    
    if (length(par_names) > 0) {
      # Map param names to current values
      par_current <- c(alpha = alpha, tau = tau, s = s, F = lf, ter = ter)
      par0 <- par_current[par_names]
      lo   <- sapply(par_names, function(n) bounds[[n]][1])
      hi   <- sapply(par_names, function(n) bounds[[n]][2])
      
      fit_m <- optim(
        par = par0,
        fn = function(p) {
          a_  <- if ("alpha" %in% par_names) p[["alpha"]] else alpha
          t_  <- if ("tau"   %in% par_names) p[["tau"]]   else tau
          s_  <- if ("s"     %in% par_names) p[["s"]]     else s
          f_  <- if ("F"     %in% par_names) p[["F"]]     else lf
          te_ <- if ("ter"   %in% par_names) p[["ter"]]   else ter
          -.objective_fun(d_facts_list,
                          sof = a_ + delta_alpha,
                          tau = t_, s = s_, lf = f_, ter = te_)
        },
        method = "L-BFGS-B",
        lower = lo, upper = hi,
        control = list(maxit = 80, factr = 1e7, pgtol = 1e-3)
      )
      
      # Extract and damp
      if ("alpha" %in% par_names) alpha <- .damp_and_shrink(alpha, fit_m$par[["alpha"]])
      if ("tau"   %in% par_names) tau   <- .damp_and_shrink(tau,   fit_m$par[["tau"]])
      if ("s"     %in% par_names) s     <- .damp_and_shrink(s,     fit_m$par[["s"]])
      if ("F"     %in% par_names) lf    <- .damp_and_shrink(lf,    fit_m$par[["F"]])
      if ("ter"   %in% par_names) ter   <- .damp_and_shrink(ter,   fit_m$par[["ter"]])
    }
    
    # ---- Record iteration ----------------------------------------------------
    ll_now <- .objective_fun(d_facts_list,
                             sof = alpha + delta_alpha,
                             tau = tau, s = s, lf = lf, ter = ter)
    
    fits <- rbind(fits, list(
      user_id     = uid,
      iteration   = as.integer(it),
      delta_alpha = list(delta_alpha),
      alpha = alpha,
      tau   = tau,
      s     = s,
      lf    = lf,
      ter   = ter,
      ll    = ll_now
    ))
    
    # ---- Convergence check ---------------------------------------------------
    ll_change <- fits[iteration == it, ll] - fits[iteration == it - 1L, ll]
    if (abs(ll_change) < ll_epsilon) break
  }
  
  return(fits)
}

# =============================================================================
# RECOVERY HELPERS — parallel grid fitting with progress + ETA
#
# These replace the inline functions from the Rmd notebook.  Key improvements:
#   1. delta_alpha extraction is correct (from list column, not attr())
#      -> eliminates "Tried to assign NULL to column 'delta_alpha_est'" warnings
#   2. Progress with ETA via the {progressr} package
#   3. Multi-core via {future}/{furrr} (caller sets up the plan)
# =============================================================================

#' Fit one learner's data with fit_learner_amle (efficient multi-start).
#'
#' Two-phase strategy to avoid local optima without multiplying runtime:
#'   Phase 1 (screening): run n_starts fits for just a few iterations each
#'   Phase 2 (refinement): continue ONLY the best one to full convergence
#'
#' Typical overhead vs single-start: ~30-50% longer, not 5× longer.
#'
#' Returns a one-row data.table with final parameter estimates + delta_alpha.
#' If keep_history = TRUE, returns ALL iterations from the best start.
fit_one_learner <- function(d_learner, free_params = c("alpha", "tau", "s", "F", "ter"),
                            keep_history = FALSE, n_starts = 10,
                            screen_iter = 8, full_iter = 50) {
  
  # Ensure C++ functions work on this process. If init_workers() was called
  # after plan(), this is a no-op. Otherwise (e.g. running on main process
  # or init_workers was forgotten), recompile.
  if (!exists("session_ll_cpp", mode = "function") ||
      !tryCatch({ session_ll_cpp(list(list(traces = c(0, 1), correct = c(TRUE, TRUE),
                                           rt = c(1, 1))), 0.3, -0.8, 0.1, 1.0, 0.3, 2L)
                  TRUE }, error = function(e) FALSE)) {
    Rcpp::sourceCpp(.actr_ll_cpp_path)
  }
  
  # Default init (always included as start 1)
  default_init <- list(alpha = 0.3, tau = -0.8, s = 0.1, F = 1, ter = 0.3)
  
  # Generate additional starting points with wider coverage via Latin-hypercube-
  # style stratified sampling: divide each parameter range into n_starts-1
  # equal strata and draw one random point per stratum, then shuffle across
  # parameters so strata are not aligned.
  make_random_init <- function() {
    init <- list(alpha = 0.3, tau = -0.8, s = 0.1, F = 1, ter = 0.3)  # defaults
    if ("alpha" %in% free_params) init$alpha <- runif(1, 0.15, 0.50)
    if ("tau"   %in% free_params) init$tau   <- runif(1, -1.5, -0.3)
    if ("s"     %in% free_params) init$s     <- runif(1, 0.03, 0.25)
    if ("F"     %in% free_params) init$F     <- runif(1, 0.3,  2.5)
    if ("ter"   %in% free_params) init$ter   <- runif(1, 0.1,  1.5)
    init
  }
  
  inits <- c(list(default_init),
             lapply(seq_len(n_starts - 1), function(i) make_random_init()))
  
  # --- Phase 1: screen all starts for a few iterations ------------------------
  screen_results <- list()
  
  for (k in seq_along(inits)) {
    fit <- tryCatch(
      fit_learner_amle(
        d_learner,
        free_params = free_params,
        init = inits[[k]],
        includes_study_trials = TRUE,
        max_iter = screen_iter,
        ll_epsilon = -Inf  # don't stop early during screening
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    
    final_ll <- fit[nrow(fit), ll]
    if (is.finite(final_ll)) {
      screen_results[[length(screen_results) + 1]] <- list(ll = final_ll, init = inits[[k]])
    }
  }
  
  if (length(screen_results) == 0) return(NULL)
  
  # Sort by ll (best first) and keep top 2
  screen_lls <- vapply(screen_results, function(x) x$ll, numeric(1))
  top_idx    <- head(order(screen_lls, decreasing = TRUE), min(2, length(screen_results)))
  
  # --- Phase 2: refine the best candidates to full convergence ----------------
  best_fit <- NULL
  best_ll  <- -Inf
  
  for (idx in top_idx) {
    candidate_fit <- tryCatch(
      fit_learner_amle(
        d_learner,
        free_params = free_params,
        init = screen_results[[idx]]$init,
        includes_study_trials = TRUE,
        max_iter = full_iter,
        ll_epsilon = 1e-5
      ),
      error = function(e) NULL
    )
    if (is.null(candidate_fit)) next
    
    candidate_ll <- candidate_fit[nrow(candidate_fit), ll]
    if (is.finite(candidate_ll) && candidate_ll > best_ll) {
      best_ll  <- candidate_ll
      best_fit <- candidate_fit
    }
  }
  
  if (is.null(best_fit)) return(NULL)
  
  n_iter <- nrow(best_fit) - 1L
  
  if (keep_history) {
    history <- copy(best_fit)
    history[, n_iter := n_iter]
    history[, delta_alpha := NULL]
    return(history)
  }
  
  # Last iteration = final estimates
  final <- best_fit[nrow(best_fit), ]
  final[, n_iter := n_iter]
  
  da <- final[["delta_alpha"]][[1]]
  if (is.null(da)) da <- numeric(0)
  
  final[, delta_alpha := NULL]
  final[, delta_alpha_est := list(list(da))]
  
  final
}


#' Subset simulated data to n_reps repetitions and n_facts facts,
#' then fit each learner in parallel with progress reporting.
#'
#' @param sim_full  Full simulated data.table from simulate_fact_learner_data()
#' @param n_reps    Number of test repetitions per fact to keep
#' @param n_facts   Number of facts per learner to keep
#' @param free_params  Character vector of parameters to estimate
#' @param seed_sample  Seed for fact sampling (default 42)
#'
#' @details
#'   Progress reporting uses the {progressr} package.  To see progress + ETA
#'   in the console, wrap the call (or the whole script) in:
#'     progressr::with_progress({ ... })
#'   or set the global handler once:
#'     progressr::handlers("cli")   # or "txtprogressbar"
#'     progressr::handlers(global = TRUE)
run_recovery_grid <- function(sim_full, n_reps, n_facts,
                              free_params = c("alpha", "tau", "s", "F", "ter"),
                              seed_sample = 42,
                              batch_size = NULL) {
  
  # Subset: keep study trials + first n_reps test trials per fact
  sim_sub <- sim_full[, .SD[rep <= n_reps], by = .(user_id, fact_id)]
  
  # Sample n_facts facts per learner
  set.seed(seed_sample)
  fact_sample <- sim_sub[, .(fact_id = sample(unique(fact_id),
                                              min(n_facts, uniqueN(fact_id)))),
                         by = user_id]
  sim_sub <- sim_sub[fact_sample, on = .(user_id, fact_id)]
  
  # Split by learner
  learner_list <- split(sim_sub, by = "user_id")
  n_learners   <- length(learner_list)
  
  # Determine batch size: default = number of workers, so we get one
  # progress update each time a full round of workers finishes
  if (is.null(batch_size)) {
    n_workers  <- tryCatch(future::nbrOfWorkers(), error = function(e) 4L)
    batch_size <- min(n_workers, n_learners)
  }
  
  # Split learner list into batches
  batch_idx <- ceiling(seq_along(learner_list) / batch_size)
  batches   <- split(learner_list, batch_idx)
  
  results  <- vector("list", length(batches))
  t_start  <- Sys.time()
  done     <- 0L
  
  for (b in seq_along(batches)) {
    # Dispatch one batch in parallel
    batch_res <- furrr::future_map(
      batches[[b]],
      function(d) fit_one_learner(d, free_params = free_params),
      .options = furrr::furrr_options(seed = TRUE)
    )
    results[[b]] <- batch_res
    
    # Print progress from main thread (flushes immediately in RStudio)
    done    <- done + length(batches[[b]])
    elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
    eta     <- if (done < n_learners) elapsed / done * (n_learners - done) else 0
    fmt_time <- function(s) {
      s <- round(s)
      if (s < 60) sprintf("%ds", s) else sprintf("%dm %ds", s %/% 60, s %% 60)
    }
    cat(sprintf("  [%d/%d learners]  elapsed: %s  ETA: %s\n",
                done, n_learners, fmt_time(elapsed), fmt_time(eta)))
    flush.console()
  }
  
  # Flatten and combine
  results <- unlist(results, recursive = FALSE)
  results <- results[!vapply(results, is.null, logical(1))]
  if (length(results) == 0) return(data.table())
  rbindlist(results, fill = TRUE)
}


# =============================================================================
# GREEDY FORWARD PARAMETER SELECTION
#
# At each step, try adding each remaining parameter to the current set of free
# parameters, fit all learners in parallel, compute the summed AIC across
# learners, and select the parameter that gives the lowest AIC.
#
# AIC = -2 * LL + 2 * k
#   where k = number of free participant-level params + (n_facts - 1) for Δα
#   Since Δα is always estimated, the (n_facts - 1) term is constant across
#   models and only the participant-level params differ.
#
# Returns a data.table with one row per (step × candidate) combination,
# including columns: step, param_added, free_params (list), total_aic,
# total_ll, n_free, selected (logical).
# =============================================================================

#' Run greedy forward selection on simulated data.
#'
#' @param sim_full   Full simulated data from simulate_fact_learner_data()
#' @param n_reps     Number of test reps per fact to use
#' @param n_facts    Number of facts per learner to use
#' @param all_params Character vector of candidate parameters
#'                   (default: all 5 participant-level params)
#' @param seed_sample Seed for fact sampling
#'
#' @return data.table with columns: step, param_added, free_params,
#'         total_ll, n_free, n_facts_aic, total_aic, selected
greedy_select_params <- function(
    sim_full,
    n_reps     = 10,
    n_facts    = 15,
    all_params = c("alpha", "tau", "s", "F", "ter"),
    seed_sample = 42,
    keep_history = FALSE
) {
  
  # --- Prepare the subset once (same logic as run_recovery_grid) --------------
  sim_sub <- sim_full[, .SD[rep <= n_reps], by = .(user_id, fact_id)]
  
  set.seed(seed_sample)
  fact_sample <- sim_sub[, .(fact_id = sample(unique(fact_id),
                                              min(n_facts, uniqueN(fact_id)))),
                         by = user_id]
  sim_sub <- sim_sub[fact_sample, on = .(user_id, fact_id)]
  
  learner_list <- split(sim_sub, by = "user_id")
  n_learners   <- length(learner_list)
  
  # Number of facts per learner (for AIC; assumes same across learners)
  nF <- uniqueN(sim_sub$fact_id)
  
  # --- Helper: fit all learners with given free_params, return fits + AIC -----
  fit_and_aic <- function(free_params) {
    # Final estimates (always needed for AIC)
    results <- furrr::future_map(learner_list, function(d) {
      fit_one_learner(d, free_params = free_params, keep_history = FALSE)
    }, .options = furrr::furrr_options(seed = TRUE))
    
    results <- results[!vapply(results, is.null, logical(1))]
    if (length(results) == 0) return(list(total_ll = NA_real_, total_aic = NA_real_,
                                          fits = data.table(), histories = data.table()))
    
    fits <- rbindlist(results, fill = TRUE)
    
    total_ll <- sum(fits$ll, na.rm = TRUE)
    k_per_learner <- length(free_params) + (nF - 1)
    total_k  <- k_per_learner * nrow(fits)
    total_aic <- -2 * total_ll + 2 * total_k
    
    # Optionally collect full iteration histories
    histories <- data.table()
    if (keep_history) {
      hist_results <- furrr::future_map(learner_list, function(d) {
        fit_one_learner(d, free_params = free_params, keep_history = TRUE)
      }, .options = furrr::furrr_options(seed = TRUE))
      hist_results <- hist_results[!vapply(hist_results, is.null, logical(1))]
      if (length(hist_results) > 0) histories <- rbindlist(hist_results, fill = TRUE)
    }
    
    list(total_ll = total_ll, total_aic = total_aic, fits = fits, histories = histories)
  }
  
  # --- Greedy forward loop ----------------------------------------------------
  selected   <- character(0)
  remaining  <- all_params
  log_rows   <- list()
  all_fits   <- list()  # per-learner fits for every candidate
  all_histories <- list()  # iteration histories (if keep_history = TRUE)
  
  for (step in seq_along(all_params)) {
    cat(sprintf("\n=== Step %d: testing %d candidate(s) ===\n",
                step, length(remaining)))
    
    step_results <- list()
    
    for (candidate in remaining) {
      trial_params <- c(selected, candidate)
      cat(sprintf("  Testing: {%s} ... ", paste(trial_params, collapse = ", ")))
      
      t0  <- Sys.time()
      res <- fit_and_aic(trial_params)
      dt  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      
      cat(sprintf("AIC = %.0f  (%.0fs)\n", res$total_aic, dt))
      flush.console()
      
      step_results[[candidate]] <- data.table(
        step        = step,
        param_added = candidate,
        free_params = list(trial_params),
        total_ll    = res$total_ll,
        n_free      = length(trial_params),
        n_facts_aic = nF,
        total_aic   = res$total_aic,
        selected    = FALSE
      )
      
      # Store per-learner fits
      key <- paste(step, candidate, sep = "_")
      if (nrow(res$fits) > 0) {
        fit_copy <- copy(res$fits)
        fit_copy[, `:=`(step = step,
                        param_added = candidate,
                        free_params_str = paste(trial_params, collapse = ","))]
        all_fits[[key]] <- fit_copy
      }
      
      # Store iteration histories
      if (keep_history && nrow(res$histories) > 0) {
        hist_copy <- copy(res$histories)
        hist_copy[, `:=`(step = step,
                         param_added = candidate,
                         free_params_str = paste(trial_params, collapse = ","))]
        all_histories[[key]] <- hist_copy
      }
    }
    
    step_dt  <- rbindlist(step_results)
    best_idx <- which.min(step_dt$total_aic)
    best_par <- step_dt$param_added[best_idx]
    step_dt[best_idx, selected := TRUE]
    
    log_rows[[step]] <- step_dt
    
    cat(sprintf("  >>> Selected: %s (AIC = %.0f)\n", best_par,
                step_dt$total_aic[best_idx]))
    flush.console()
    
    selected  <- c(selected, best_par)
    remaining <- setdiff(remaining, best_par)
  }
  
  summary_dt    <- rbindlist(log_rows)
  fits_dt       <- rbindlist(all_fits, fill = TRUE)
  histories_dt  <- if (length(all_histories) > 0) rbindlist(all_histories, fill = TRUE) else data.table()
  
  list(summary = summary_dt, learner_fits = fits_dt, histories = histories_dt)
}