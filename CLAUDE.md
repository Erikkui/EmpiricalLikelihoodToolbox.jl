# CLAUDE.md

Guidance for Claude Code in this repository. Keep it under 150 lines: describe structure,
contracts and pitfalls, not details the code already shows.

## What this is

`EmpiricalLikelihoodToolbox` is a Julia package for Bayesian inference on simulator-based
(intractable-likelihood) models with empirical-likelihood-style summary statistics. The likelihood
approximation is chosen by `MethodsOptions.inference_method`: Gaussian Subset Likelihood (`GSL()`,
Haario et al. 2015, default), Bayesian Synthetic Likelihood (`BSL(n_sim)`, Wood 2010) or ABC-MCMC
(`ABC(...)`, Marjoram et al. 2003), plus ABC rejection and ABC-SMC. The README has the summary
catalog, method tradeoffs, references and a usage example (`examples/main.jl` mirrors it).

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # install deps
julia --project=. -e 'using Pkg; Pkg.test()'           # run test suite
julia --project=. examples/main.jl                     # run the example end-to-end
```

- Run tests only via `Pkg.test()`: `julia --project=. test/runtests.jl` fails, because
  `StableRNGs` is a test-only dependency. New tests go in a `test/test_*.jl` included from
  `test/runtests.jl`. CI (`.github/workflows/CI.yml`): Julia 1.10, 1.12, `pre` on ubuntu x64.
- No formatter. Match the existing style (e.g. spaces inside call parentheses: `f( x, y )`).

## Layout

`src/EmpiricalLikelihoodToolbox.jl` is a flat `include()` list with no submodules, so load order
matters: `core_types.jl` and `utils/utils.jl` first, `mcmc/loss_functions.jl` before `inference/`
(method structs hold a loss), `inference/interface.jl` before the method files.

| Path | Contents |
|---|---|
| `src/core_types.jl` | Abstract types, `MethodsOptions`, `BufferContainer`, containers |
| `src/models/` | Simulation models, `solvers.jl` |
| `src/summaries/` | Summary statistics, ECDF and Chamfer kernels |
| `src/utils/` | `utils.jl`, `resamplers.jl`, `bin_calculation.jl`, `chi2_test.jl` |
| `src/inference/` | `interface.jl`, `GSL.jl`, `BSL.jl`, `ABC/{method,pilot,mcmc,post_correct}.jl` |
| `src/mcmc/` | `target.jl`, samplers (`AM.jl`, `DRAM.jl`, `runner.jl`) and their parts |
| `src/abc/` | `rejection.jl`, `smc.jl` (included after `mcmc/runner.jl`) |

Three layers: (1) **inference methods** set up the target and evaluate the log-likelihood once;
(2) **samplers** (AM/DRAM) are generic over an `EvaluatedState` and contain no method-specific
code; (3) **ABC population algorithms** reuse the ABC pilot and distance code.

## Pipeline

1. **Model** (`src/models/*.jl`): a `Base.@kwdef struct <: AbstractSimulationModel` with its
   parameters and `all_parameters`/`active_parameters` tuples of `Symbol`s (inactive ones stay
   fixed). `solve_model(model, t_end; rng)` simulates. `get_active_model_params` and
   `update_model_parameters` (`utils/utils.jl`) read and write parameters through these tuples.
2. **Summary statistics** (`src/summaries/*.jl`): structs subtyping `AbstractSummaryStatistic`
   (hierarchy in `core_types.jl`). `XDiff` variants use numerical derivatives of order
   `diff_order`, with step `model.dt_obs` (summaries hold no time step).
   `JointSummaryStatistics(s1, s2, ...)` concatenates outputs.
   Informal interface (copy an existing file): two `calculate_summary_statistic!` methods (training:
   one `DataContainer`; MCMC: observed + simulated), `allocate_buffer`, `get_summary_length`,
   `required_diff_order`, `generate_stat_name`, `finalize_summary`. ECDF summaries also implement
   `get_bin_quantity`; `create_bins` (`utils/bin_calculation.jl`) picks edges within the 0.5–99.5 %
   quantiles, spaced by `axis_uniform` (`:xax`, `:yax` equal-mass, `:log`). Bins given to a
   constructor are kept (`_has_existing_bins`).
3. **Resamplers** (`utils/resamplers.jl`): callable structs, `resampler(data_container, options,
   index_cache) -> (x_inds, y_inds)`. `LengthChangingSampler`: `RademacherSplit`,
   `ContiguousBlockSplit`. `LengthPreservingSampler`: `StandardBootstrap`, `MovingBlockBootstrap`,
   `NoResampling` (identity, all data as both sets). Used for training and per MCMC step.
4. **`MethodsOptions`** (`core_types.jl`): method, resampler, ECDF type, bins, resampling counts,
   `n_summaries`/`n_loss_evals` (averaging), `standardize`, embedding. `TargetData` fills in `N_obs`
   (raw length = simulation length) and `effective_N_obs` (after embedding and trimming).
5. **`TargetData(data, joint_stats, options, model; priors, rng)`** (`mcmc/target.jl`): entry point,
   `data` is the raw `Matrix{Float64}`. Order: `check_method_options`, `N_obs`, prior warnings,
   embedding, `DataContainer`, derivatives, all buffers, bins, `train_target`,
   `finalize_target_covariance`, store model, `finalize_inference_method`. Returns
   `(target, training_summaries)`. The 5-arg form with a `loss` argument is a deprecated shim.
6. **MCMC**: `mcmcrun(target, mcmc_options) -> (results::ResultsBuffer, state::MCMCState)`.
   Catches exceptions mid-chain and returns the partial chain.

## Inference method interface (`src/inference/interface.jl`)

Required, dispatched on the method:
- `train_target(method, statistics, data_container, buffers, options)` →
  `(mean, cov or nothing, training_summaries)`
- `finalize_target_covariance(method, cov_mat, summary_length)`
- `allocate_method_buffer(method, summary_length)` → `BufferContainer.method_buffer` (scratch)
- `check_method_options(method, options)` → throws `ArgumentError`
- `log_likelihood!(method_data, method, inference_state, target, model, mcmc_options, rng)`:
  log-likelihood without prior, at the parameters already set in `model`

Optional, with fallbacks: `finalize_inference_method`, `allocate_method_data`/
`invalidate_method_data!` (per-state data, e.g. ABC distances), `max_log_likelihood` (`Inf`) and
`early_rejection_allowed`, `allocate_inference_state`/`adapt_inference_state!` (mutable per-chain
run state, AM only), `allocate_results`/`record_chain_step!`/`finalize_results`. A new method is a
new file with the required functions, included after `inference/interface.jl`.

Each method has a `Base.show` (`GSL()`, `BSL(500)`, loss shown only when not the default, e.g.
`BSL(500, LogLikelihood(2.0))`). Downstream code puts `string(inference_method)` in file names:
keep this output stable.

- **GSL**: `train_target` resamples the observed data `training_resamplings` times for a fixed mean
  and covariance (`:cov` or `:donsker`), regularized and factorized once. Per step: simulate once
  (or `n_loss_evals` times, averaged), summarize via `calculate_simulated_statistics` (`loss.jl`),
  evaluate `method.loss(sim, obs_mean, cov_factorization)`.
- **BSL**: training gives only the mean (one pass under `NoResampling`). Per step: `n_sim`
  simulations into `method_buffer` `(summary_length, n_sim)`, mean/covariance from their spread,
  same loss minus `0.5*logdet`. The target covariance is an unused identity placeholder.
- Losses (`loss_functions.jl`) return raw values; `scale_loss` applies standardization (GSL only)
  and `inverse_scaling`. `regularized_cholesky` (`utils/utils.jl`) is shared by GSL, BSL and ABC.
- Option rules: BSL rejects `standardize`; GSL rejects it with `NoResampling`; ABC rejects
  `standardize`, `MCMCOptions.loss_function` and the deprecated `TargetData`. A non-`nothing`
  `MCMCOptions.loss_function` must equal the method's loss, else `mcmcrun` throws.

## ABC (`src/inference/ABC/`, `src/abc/`)

- `ABC(; epsilon = NaN, epsilon_quantile, n_pilot, distance = :scaled, kernel, n_sim = 1,
  adapt_epsilon)`. No loss. `log_likelihood!` runs `n_sim` × `simulate_distance` (`Inf` on a failed
  simulation), stores distances in the state's `method_data`, returns the log of the mean kernel
  value. Kernels are unnormalized, so `max_log_likelihood = 0`.
- `finalize_inference_method` resolves the distance (`:scaled` → `WeightedEuclidean`, `:mahalanobis`
  → `Mahalanobis`) and a `NaN` ε from a pilot quantile, then writes the resolved method into
  **both** `target.options` and `target.data.options` (`_replace_inference_method`).
- The pilot (`pilot_simulations`) samples priors: every active parameter needs a prior.
- In MCMC, ε lives in the mutable run state `ABCRunState`, not in the method. `EpsilonAdaptation`
  (Vihola & Franks 2020, AM only) updates ε in the first `n_adapt` steps and recomputes the current
  log-target from stored distances. The target is never rebuilt.
- Results add `distance_chain` (`n_sim × nsteps`), `epsilon_chain` and final `epsilon` (NaN for
  GSL/BSL). `abc_post_correct(results, target, delta; burnin)` gives post-correction weights.
- `abc_rejection(target, n_draws; epsilon | n_keep, rng)`; `abc_smc(target, ABCSMCOptions(...);
  rng)`: Drovandi & Pettitt (2011) replenishment SMC, uniform kernel, `n_sim = 1` only.

## Samplers (`src/mcmc/`)

- `runner.jl`: builds an immutable `RunContext` (`chain_states`, `proposal`, `policy`,
  `inference_state`), evaluates the initial state, calls `mcmc_options.mcmc_algorithm` (callable
  `AM`/`DRAM` struct), then `finalize_results`. `adaptive_proposal.jl` is shared by AM and DRAM.
- `evaluated_state.jl`: mutable `EvaluatedState` (`params`, `log_prior`, `log_likelihood`,
  `log_target`, `method_data`, `simulated`). `evaluate_log_target!`: prior (zero → no simulation),
  set parameters, `Xoshiro(rng_seed)`, `log_likelihood!`. It skips the simulation if
  `log_prior + max_log_likelihood < log_threshold` (early rejection).
- `chain_states[1]` is the current state, the rest are proposals (one per DRAM stage). Acceptance
  swaps entries, no copies, so `chain_states[1]` is valid even after a crash.
- `noisy_likelihood_policy.jl`: `update_interval` (`<= 0` disables kicks), `discard_noisy_updates`,
  `reevaluate_current_loss`; used by AM and DRAM. BSL and ABC are exact pseudo-marginal only with
  `update_interval <= 0` and `reevaluate_current_loss = false`.
- Early rejection (`MCMCOptions.early_rejection`, AM only): `log u` is drawn before the evaluation.
  Exact; saves simulations only for ABC with informative priors. Off with `likelihood_noise_scale >
  0` and during ε adaptation. Chains match on/off only if evaluation draws no global random numbers.
- The hot loop must not allocate: put scratch space in `BufferContainer`, allocated in `TargetData`.
- Priors (`prior_eval.jl`): `NamedTuple` by parameter name; `Distributions.jl` or `nothing` (flat).
