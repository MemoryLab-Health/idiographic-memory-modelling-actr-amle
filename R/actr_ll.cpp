// actr_ll.cpp — C++ implementation of ACT-R activation & likelihood
//
// Replaces the R functions:
//   calculate_activation_memory()
//   .fact_log_likelihood()
//   .session_log_likelihood()
//
// Compile with:  Rcpp::sourceCpp("actr_ll.cpp")
//
// Exported R functions:
//   activation_cpp(t, traces, sof, c = 0.25)
//   fact_ll_cpp(traces, correct, rt, sof, tau, s, lf, ter, start_index = 2, c = 0.25)
//   session_ll_cpp(data_list, sof_vec, tau, s, lf, ter, start_index = 2, c = 0.25)

#include <Rcpp.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

// --- Internal helpers (not exported) ----------------------------------------

static inline double trace_odds(double t, double te, double d, double cap = 10.0) {
  if (t > te) {
    double val = std::pow(t - te, -d);
    return std::min(val, cap);
  }
  return 0.0;
}

static double activation_internal(double t, const double* traces, int n_traces,
                                  double sof, double c_param) {
  // Allocate decay rates on the stack for small n (typical: < 50 traces)
  std::vector<double> dis(n_traces);

  // Phase 1: compute adaptive decay rates
  dis[0] = sof;
  for (int i = 1; i < n_traces; i++) {
    double odds_sum = 0.0;
    for (int j = 0; j < i; j++) {
      odds_sum += trace_odds(traces[i], traces[j], dis[j]);
    }
    dis[i] = c_param * odds_sum + sof;
  }

  // Phase 2: compute total memory odds at time t
  double memory_odds = 0.0;
  for (int i = 0; i < n_traces; i++) {
    memory_odds += trace_odds(t, traces[i], dis[i]);
  }

  return std::log(memory_odds);
}

// --- Exported functions -----------------------------------------------------

// [[Rcpp::export]]
double activation_cpp(double t, NumericVector traces_r, double sof,
                      double c_param = 0.25) {
  // Filter traces < t (matching R behaviour)
  std::vector<double> traces;
  traces.reserve(traces_r.size());
  for (int i = 0; i < traces_r.size(); i++) {
    if (traces_r[i] < t) traces.push_back(traces_r[i]);
  }
  if (traces.empty()) return R_NegInf;
  return activation_internal(t, traces.data(), traces.size(), sof, c_param);
}

// [[Rcpp::export]]
double fact_ll_cpp(NumericVector traces, LogicalVector correct,
                   NumericVector rt, double sof,
                   double tau, double s, double lf, double ter,
                   int start_index = 2, double c_param = 0.25) {
  int n = traces.size();
  if (n < start_index) return 0.0;

  double ll = 0.0;
  double beta = 1.0 / s;
  double log_floor = std::log(1e-20);

  for (int i = start_index - 1; i < n; i++) {  // R is 1-indexed, C++ is 0-indexed
    // Compute activation from traces[0..i-1]
    double a = activation_internal(traces[i], &traces[0], i, sof, c_param);

    if (correct[i]) {
      // P(correct)
      double p_corr = 1.0 / (1.0 + std::exp(-(a - tau) / s));
      ll += std::log(std::max(p_corr, 1e-20));

      // RT density (log-logistic with activation a)
      double alpha_rt = lf * std::exp(-a);
      double t_rt = rt[i] - ter;
      if (t_rt < 0.0) t_rt = 0.0;
      double ratio = t_rt / alpha_rt;
      double density = (beta / alpha_rt) * std::pow(ratio, beta - 1.0) /
                       std::pow(1.0 + std::pow(ratio, beta), 2.0);
      ll += std::log(std::max(density, 1e-20));
    } else {
      // P(incorrect)
      double p_corr = 1.0 / (1.0 + std::exp(-(a - tau) / s));
      ll += std::log(std::max(1.0 - p_corr, 1e-20));

      // RT density (log-logistic with threshold tau as activation)
      double alpha_rt = lf * std::exp(-tau);
      double t_rt = rt[i] - ter;
      if (t_rt < 0.0) t_rt = 0.0;
      double ratio = t_rt / alpha_rt;
      double density = (beta / alpha_rt) * std::pow(ratio, beta - 1.0) /
                       std::pow(1.0 + std::pow(ratio, beta), 2.0);
      ll += std::log(std::max(density, 1e-20));
    }
  }

  return ll;
}

// [[Rcpp::export]]
double session_ll_cpp(List data_list, NumericVector sof_vec,
                      double tau, double s, double lf, double ter,
                      int start_index = 2, double c_param = 0.25) {
  int n_facts = data_list.size();
  double ll = 0.0;

  for (int f = 0; f < n_facts; f++) {
    List fact = data_list[f];
    NumericVector traces = fact["traces"];
    LogicalVector correct = fact["correct"];
    NumericVector rt = fact["rt"];
    double sof = sof_vec[f];

    ll += fact_ll_cpp(traces, correct, rt, sof, tau, s, lf, ter,
                      start_index, c_param);
  }

  return ll;
}
