# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`EmpiricalLikelihoodToolbox` is a Julia package for Bayesian inference on intractable-likelihood
(simulator-based) models using empirical-likelihood-style summary statistics. Three likelihood
approximations are supported, selected via `MethodsOptions.inference_method`: Gaussian Subset
Likelihood (`GSL()`, Haario et al. 2015, the default), Bayesian Synthetic Likelihood (`BSL(n_sim)`,
Wood 2010) and ABC-MCMC (`ABC(...)`, Marjoram et al. 2003). See the README for the full
summary-statistic catalog, resamplers, the GSL/BSL/ABC tradeoffs,
references, and a runnable usage example (`examples/main.jl` mirrors it).

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # install deps
julia --project=. -e 'using Pkg; Pkg.test()'           # run test suite
julia --project=. examples/main.jl                     # run the example end-to-end
```

There is no linter/formatter configured. Run the tests with `Pkg.test()`: a direct
`julia --project=. test/runtests.jl` fails, because `StableRNGs` is a test-only dependency. CI
(`.github/workflows/CI.yml`) runs the suite on Julia 1.10, 1.12, and `pre` on ubuntu-latest/x64.
When adding tests, add them to a `test/test_*.jl` file included from `test/runtests.jl`.

## Architecture

The package wires together five kinds of pluggable pieces — **models**, **summary statistics**,
**resamplers**, **inference methods** (`src/inference/`), and **MCMC algorithms/loss functions** —
through a small set of container structs defined in `src/core_types.jl` (which only holds
`abstract type AbstractInferenceMethod`, not the method structs). `src/EmpiricalLikelihoodToolbox.jl`
is a flat `include()` list; there's no submodule structure, so load order there matters (e.g.
`core_types.jl` and `utils/utils.jl` first, `mcmc/loss_functions.jl` before `inference/*.jl`, since
the method structs hold a loss).

Three layers: (1) **inference methods** (`src/inference/`): target setup and one evaluation of the
log-likelihood; (2) **samplers** (`src/mcmc/`): AM/DRAM, generic over an `EvaluatedState`, with no
method-specific code; (3) **ABC population algorithms** (`src/abc/`).

### Inference methods (`src/inference/`: `interface.jl`, `GSL.jl`, `BSL.jl`, `ABC/`)

`interface.jl` documents the method interface and holds the fallbacks for `AbstractInferenceMethod`.
One file per method holds the struct (`GSL(; loss)`, `BSL(n_sim; loss)`, the loss defaults to
`LogLikelihood()`), its `Base.show` (`GSL()`, `BSL(500)`, loss appended only when non-default, eg.
`BSL(500, LogLikelihood(2.0))`; downstream code uses `string(inference_method)` in file names, keep it
stable) and the method interface, each dispatched on the method:
- `train_target(method, statistics, data_container, buffers, options)` -> `(mean, cov or nothing, training_summaries)`
- `finalize_target_covariance(method, cov_mat, summary_length)`
- `allocate_method_buffer(method, summary_length)` -> shared scratch, stored as `BufferContainer.method_buffer`
- `check_method_options(method, options)` -> throws `ArgumentError` (called first in `TargetData`)
- `log_likelihood!(method_data, method, inference_state, target, model, mcmc_options, rng)` ->
  log-likelihood without prior, at the parameters already set in `model`

Optional (fallbacks in `interface.jl`): `finalize_inference_method(method, target, training_summaries,
rng) -> target` (end of `TargetData`), `allocate_method_data(method)` (per-state data an evaluated state
keeps, `nothing` by default) + `invalidate_method_data!`, `max_log_likelihood(method,
inference_state)` (`Inf`; bound for early rejection) + `early_rejection_allowed(method, ii)` (`true`;
ABC: `false` during ε adaptation), `allocate_inference_state(method)` (mutable
run state of one chain, `nothing`), `adapt_inference_state!(inference_state, method, current, acceptance_probability,
ii) -> Bool` (AM only, after each step), `allocate_results(method, model, chain_length, n_stages)`,
`record_chain_step!(results, method, inference_state, current, ii)`, `finalize_results(results, method,
inference_state, last_step)` (final ε, partial-chain slicing). A new method adds a file with the
required functions and an `include` after `inference/interface.jl`.

**ABC** (`ABC/method.jl`: kernels, struct, interface, distance/ε resolution; `ABC/pilot.jl`;
`ABC/mcmc.jl`: `EpsilonAdaptation`, run state, results; `ABC/post_correct.jl`): `ABC(; epsilon = NaN,
epsilon_quantile = 0.05, n_pilot = 1000, distance = :scaled, kernel = UniformKernel(), n_sim = 1,
adapt_epsilon = nothing)`, `ABC(epsilon; kwargs...)`. No loss. `log_likelihood!` runs `n_sim` ×
`simulate_distance(target, model, rng)` (simulate → summarize → distance to `target.obs_mean`; `Inf`
on a failed simulation), writes the distances into the evaluated state's `method_data` (length `n_sim`;
`method_buffer` is unused), and returns `_log_kernel_average` = `logsumexp(log_kernel.(kernel, ρ, ε)) -
log(n_sim)` (kernels `UniformKernel`/`GaussianKernel`/`EpanechnikovKernel`, unnormalized, so
`max_log_likelihood = 0`). ε is read from the run state `ABCRunState` (mutable, built from the
resolved method's ε by `mcmcrun`), the kernel from the method. `train_target` shares
`_train_reference_summary` (in `BSL.jl`) with BSL but keeps the training covariance;
`finalize_target_covariance` stores its regularized factorization (identity placeholder under
`NoResampling`). `finalize_inference_method(::ABC, ...)` resolves `:scaled` →
`WeightedEuclidean(1 ./ scale.^2)` (training sd with a resampler, pilot MAD under `NoResampling`;
zero scales → smallest positive scale) and `:mahalanobis` → `Mahalanobis(inv(regularized cov))`,
sets a `NaN` ε to a pilot-distance quantile, and `@set`s the resolved method into both
`target.options` and `target.data.options` (`_replace_inference_method`). The pilot
(`pilot_simulations(target, n_pilot; rng)` → `(params, summaries, distances)`; internally
`_pilot_summaries` then `_pilot_distances`, so scales are set before distances without re-simulating)
samples the priors and throws if any active parameter has no prior. ABC rejects `standardize`,
`MCMCOptions.loss_function` and the deprecated 5-arg `TargetData`. ABC-MCMC: `record_chain_step!` stores each
state's distances in `ResultsBuffer.distance_chain` (`n_sim × nsteps`; `0 × nsteps` otherwise) and
its ε in `epsilon_chain`; `finalize_results` sets the final `epsilon` (NaN for GSL/BSL).
`EpsilonAdaptation` (Vihola & Franks 2020, AM only; `mcmcrun` rejects it with DRAM) runs in
`adapt_inference_state!`: a Robbins-Monro update of `ABCRunState.epsilon` in the first `n_adapt` steps, then the
current log-target is recomputed from its stored distances (`_refresh_log_target!`, no simulation);
the target is not rebuilt. `abc_post_correct(results, target, delta; burnin)` gives post-correction
weights.

**ABC population samplers** (`src/abc/rejection.jl`, `src/abc/smc.jl`, included after `mcmc/runner.jl`): `abc_rejection(target, n_draws; epsilon | n_keep, rng)` wraps `pilot_simulations`; `abc_smc(target, ABCSMCOptions(...); rng)` is Drovandi & Pettitt (2011) replenishment ABC-SMC (uniform kernel, `n_sim = 1` only; gen 0 via `_pilot_summaries`/`_pilot_distances`, MH moves via `simulate_distance` with a per-simulation `Xoshiro(rand(rng, UInt64))`).

### The pipeline (see `examples/main.jl` / README for the full call sequence)

1. **Model** (`src/models/*.jl`) — a `Base.@kwdef struct <: AbstractSimulationModel` holding
   parameters plus `all_parameters`/`active_parameters` tuples of `Symbol`s (which params exist vs.
   which are being inferred; others are held fixed). `solve_model(model, t_end; rng)` simulates data.
   `get_active_model_params`/`update_model_parameters` (in `utils/utils.jl`) read/write params
   generically off these tuples via `getfield`/`Accessors.setproperties`.
2. **Summary statistics** (`src/summaries/*.jl`) — each is a struct subtyping
   `AbstractSummaryStatistic` (see the type hierarchy in `core_types.jl`:
   `AbstractECDFSummary`/`StandardECDFSummary`/`TwoSetSummary` → `CILSummary`/`IDSummary`/
   `ChamferECDFSummary`, plus `AbstractChamferSummary`/`AbstractCumulativeSumSummary`). Several
   summaries come in `X` and `XDiff` variants (e.g. `StandardECDF`/`StandardECDFDiff`,
   `CIL`/`CILDiff`, `ID`/`IDDiff`, `ChamferDistance`/`ChamferDistanceDiff`) — the `Diff` variant
   computes the same statistic on numerical derivatives of the data (order `diff_order`) instead of
   the raw series. The summaries carry no time step: the derivatives' step is `model.dt_obs`, for
   the observed data in `TargetData` and for the simulated data in `create_simulated_data`, both
   from the model stored in the target. Multiple summaries are combined via
   `JointSummaryStatistics(stat1, stat2, ...)`, which concatenates their outputs into one vector
   using each summary's `summary_length`.
   Each summary struct implements a small informal interface (grep any `summaries/*.jl` file for the
   pattern): two `calculate_summary_statistic!` methods (one training-time signature taking a single
   `DataContainer`, one MCMC-time signature taking observed + simulated `DataContainer`s), plus
   `allocate_buffer`, `get_summary_length`, `required_diff_order`, `generate_stat_name`, and
   `finalize_summary`. ECDF-based summaries additionally need `get_bin_quantity` and go through
   `create_bins`/`_bin_select` (`src/utils/bin_calculation.jl`) which auto-selects bin edges from
   the quantity each summary evaluates its ECDF on (raw rows, `differences[order]` for the Diff
   summaries, resampled distances/ratios for two-set summaries), within the 0.5-99.5 % quantile range
   (`axis_uniform` option controls whether bins are uniform in `:xax`, uniform in CDF-space via `:yax`
   at equal-mass midpoint levels with ties snapped onto distinct data values, or `:log`-spaced) — `create_bins` skips
   this and returns the summary unchanged when it already carries bins (`_has_existing_bins`), so
   bins passed to a summary's constructor survive a full `TargetData` pipeline run.
3. **Resamplers** (`src/utils/resamplers.jl`) — callable structs (`RademacherSplit` 50/50 split,
   `ContiguosBlockSplit` contiguous block for time series, `StandardBootstrap`, `NoResampling` a
   deterministic identity split that returns all data unchanged as both sets) subtyping either
   `LengthPreservingSampler` or `LengthChangingSampler`. Called as `resampler(data_container, options,
   index_cache)` → `(x_inds, y_inds)`, used both to build the training distribution of summaries and,
   per MCMC step, to resample simulated data before comparing to the observed target.
4. **`MethodsOptions`** (`core_types.jl`) bundles cross-cutting config: which inference method (and
   through it the loss), which resampler/ECDF
   calculation to use, bin count, number of training resamplings, `n_summaries`/`n_loss_evals`
   (noise-reduction knobs — average multiple resampled summaries / multiple loss evaluations per
   MCMC step), `standardize` (z-score the loss), and `embedding_dim`/`embedding_type` (embedding via
   `embedding()` in `utils/utils.jl`, applied to the observed data in `TargetData` and to each
   simulation in `create_simulated_data`). `N_obs` (raw series length, the length of every
   simulation) and `effective_N_obs` (post-embedding, post-trim length) are filled in by `TargetData`.
5. **`TargetData(data, summary_statistics, methods_options, model; priors)`** (`src/mcmc/target.jl`)
   is the main entry point that ties everything together: takes the *raw* observed series, sets
   `N_obs` from its length, warns about active parameters without a prior, embeds the series, builds a
   `DataContainer`, precomputes numerical
   derivatives (step `model.dt_obs`) if any summary needs them, allocates all buffers up front (`BufferContainer` — the
   package is written to avoid allocations in the MCMC hot loop, including a method-specific
   `method_buffer` from `allocate_method_buffer`: `(summary_length, n_sim)` under BSL, empty under GSL and ABC), finalizes bin edges per summary, then computes the
   fixed reference summary via `train_target` (see the inference methods section): `GSL()`
   resamples the observed data `training_resamplings` times to estimate both the target mean and its
   covariance (`:cov` or `:donsker`); `BSL()` only needs the mean (its covariance instead comes from
   simulations at every MCMC step, see below) — a single deterministic pass over all the data when
   `resampling_type isa NoResampling`, otherwise GSL's same resampling loop with the covariance
   discarded. `finalize_target_covariance` regularizes+inverts GSL's covariance once, or allocates an
   identity placeholder for BSL (never read: BSL factorizes its own covariance each step). Calls `check_method_options` first (BSL rejects `standardize = true`, GSL rejects
   `standardize` with `NoResampling`). The old `TargetData(data, stats, options, loss, model; priors)`
   is a deprecated shim that stores `loss` in the inference method and warns. Stores the model in
   `target.model`, then calls `finalize_inference_method` (ABC: distance/ε resolution, pilot; takes the
   `rng` kwarg of `TargetData`). Returns `(target::TargetData, training_summaries)`.
6. **MCMC** (`src/mcmc/`) — `mcmcrun(target, mcmc_options)` (`runner.jl`) runs `target.model`. It
   builds a `RunContext` (immutable: `chain_states`, `proposal`, `policy`, `inference_state`),
   evaluates the initial state, calls the pluggable `mcmc_algorithm` (`AM.jl` = adaptive Metropolis,
   `DRAM.jl` = delayed rejection AM), a callable struct `(target, model, run_context, mcmc_options,
   results) -> results`, then `finalize_results`. It returns `(results::ResultsBuffer,
   state::MCMCState)`; it catches exceptions mid-chain and returns the partial chain.
   - `evaluated_state.jl`: mutable `EvaluatedState{A}` (`params`, `log_prior`, `log_likelihood`,
     `log_target`, `method_data`, `simulated`). `evaluate_log_target!(evaluated_state, target, model, params,
     mcmc_options, inference_state; rng_seed, log_threshold = -Inf)`: prior (a zero prior returns
     before simulating, `simulated = false`), `update_model_parameters`, `Xoshiro(rng_seed)`, the
     method's `log_likelihood!`, plus prior and `likelihood_noise_scale` noise. Early rejection: if
     `log_prior + max_log_likelihood < log_threshold` it skips the simulation (`simulated = false`,
     log-target `-Inf`, no noise draw). `evaluate_log_target(params, target, model, mcmc_options;
     rng_seed)` is a thin public wrapper returning the log-target.
   - The samplers own the evaluated states: `chain_states[1]` is the current state, `[2:end]` the proposals
     (one per DRAM stage); on acceptance the entries swap (no copies), so `chain_states[1]` is the
     current state even after a crash.
   - `adaptive_proposal.jl`: `AdaptiveProposal` (Welford running mean/cov via `recursive_welford!`,
     every `adaptation_interval` steps after step 100 `cov = running_cov*2.38²/d + 1e-6 I`):
     `propose_parameters!(params_out, proposal, current_params; scale)` (DRAM stages use `scale = γ_k`),
     `update_proposal_covariance!(proposal, params, ii) -> Bool`, `backup_proposal_adaptation!`/`restore_proposal_adaptation!`.
   - `noisy_likelihood_policy.jl`: `NoisyLikelihoodPolicy` from `update_interval`,
     `discard_noisy_updates`, `reevaluate_current_loss`, honoured by AM and DRAM. The sampler
     re-evaluates the current state with the proposal's step seed when `reevaluate_current_loss`;
     `check_stuck_chain!(policy, accepted)` returns `:none`/`:kick` (re-evaluate current, more than
     `update_interval` consecutive rejections)/`:rewind` (with `discard_noisy_updates`: restore
     `backup_params` + running mean/cov, re-evaluate, repeat the step without storing it);
     `apply_stuck_chain_action!` carries it out and counts `results.stuck_kicks`. `update_interval <= 0`
     disables kicks. BSL and ABC are exact pseudo-marginal only with `update_interval <= 0` and
     `reevaluate_current_loss = false`; `reevaluate_current_loss` is particularly relevant under BSL,
     where the current and proposed states' covariances are independent per-step Monte Carlo estimates.
   - AM step: `propose_parameters!` → step seed → optional re-evaluation → `log_u = log(rand())` → `evaluate_log_target!`
     proposal with `log_threshold = log_u + current.log_target` (early rejection; `-Inf` when
     `MCMCOptions.early_rejection = false`, `likelihood_noise_scale > 0` or `!early_rejection_allowed`)
     → accept iff `log_u < log_ratio` (`backup_state!`, swap) → policy → `adapt_inference_state!` → `record_chain_step!` →
     `update_proposal_covariance!`. Early rejection is exact; it saves simulations only with informative priors, never for
     GSL/BSL (bound `Inf`); chains match bit for bit with it on/off only when the evaluation draws no
     global random numbers (resampler called: `n_summaries > 1` or length-changing sampler). DRAM:
     same, with delayed-rejection stages (`compute_log_path` on the stage states), no early
     rejection and no `adapt_inference_state!`.
   - Under `GSL()` `log_likelihood!` simulates once (or `n_loss_evals` times, averaged),
     resamples/summarizes via `calculate_simulated_statistics` (`loss.jl`, honoring `n_summaries`), and
     evaluates `method.loss(sim, obs_mean, cov_factorization)` against the *fixed* target from
     `TargetData`; under `BSL()` it runs `n_sim` fresh simulations (each resampled/summarized the same
     way, so any resampler works unchanged; summaries in `method_buffer`), estimates the
     mean/covariance *from the spread across those simulations*, and evaluates the same loss callable
     with that simulated factorization minus `0.5*logdet`. The loss callables (`loss_functions.jl`)
     return the raw value; `scale_loss(loss, raw, target)` applies standardization (GSL only) and
     `inverse_scaling`. `MCMCOptions.loss_function` defaults to `nothing`; if set it must equal the
     method's loss or `mcmcrun` throws. `regularized_cholesky` (`utils/utils.jl`) is the shared
     ridge-regularization helper used by GSL and BSL. Priors (`prior_eval.jl`) are `NamedTuple`s keyed
     by parameter name, valued as `Distributions.jl` distributions or `nothing` (flat/uninformative).

