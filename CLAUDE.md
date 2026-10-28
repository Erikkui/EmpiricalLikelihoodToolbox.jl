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
julia --project=. test/runtests.jl                     # run tests directly
julia --project=. examples/main.jl                     # run the example end-to-end
```

There is no linter/formatter configured. `test/runtests.jl` is currently a stub (`@testset` with no
assertions) — CI (`.github/workflows/CI.yml`) runs it on Julia 1.10, 1.12, and `pre` on
ubuntu-latest/x64. When adding tests, add them inside that `@testset` block or new files included
from it.

## Architecture

The package wires together five kinds of pluggable pieces — **models**, **summary statistics**,
**resamplers**, **inference methods** (`src/inference/`), and **MCMC algorithms/loss functions** —
through a small set of container structs defined in `src/core_types.jl` (which only holds
`abstract type AbstractInferenceMethod`, not the method structs). `src/EmpiricalLikelihoodToolbox.jl`
is a flat `include()` list; there's no submodule structure, so load order there matters (e.g.
`core_types.jl` and `utils/utils.jl` first, `mcmc/loss_functions.jl` before `inference/*.jl`, since
the method structs hold a loss).

### Inference methods (`src/inference/GSL.jl`, `BSL.jl`, `ABC.jl`)

One file per method holds the struct (`GSL(; loss)`, `BSL(n_sim; loss)`, the loss defaults to
`LogLikelihood()`), its `Base.show` (`GSL()`, `BSL(500)`, loss appended only when non-default, eg.
`BSL(500, LogLikelihood(2.0))`; downstream code uses `string(inference_method)` in file names, keep it
stable) and the full method interface, each dispatched on the method:
- `train_target(method, statistics, data_container, buffers, options)` -> `(mean, cov or nothing, training_summaries)`
- `finalize_target_covariance(method, cov_mat, summary_length)`
- `allocate_method_buffer(method, summary_length)` -> stored as `BufferContainer.method_buffer`
- `check_method_options(method, options)` -> throws `ArgumentError` (called first in `TargetData`)
- `calculate_loss(method, target, model, mcmc_options, rng)` -> log-likelihood without prior

A new method adds a file here with these five functions and an `include` after `loss_functions.jl`.
Optionally it adds `finalize_inference_method(method, target, training_summaries, rng) -> target`
(default: target unchanged), called at the end of `TargetData` to resolve settings that need the
finished target.

**ABC** (`ABC.jl`): `ABC(; epsilon = NaN, epsilon_quantile = 0.05, n_pilot = 1000, distance = :scaled,
kernel = UniformKernel(), n_sim = 1, adapt_epsilon = nothing)`, `ABC(epsilon; kwargs...)`. No loss.
`calculate_loss` runs `n_sim` × `simulate_distance(target, model, rng)` (simulate → summarize →
distance to `target.obs_mean`; `Inf` on a failed simulation), stores the distances in
`target.buffers.method_buffer` (length `n_sim`), and returns `logsumexp(log_kernel.(kernel, ρ, ε)) -
log(n_sim)` (kernels `UniformKernel`/`GaussianKernel`/`EpanechnikovKernel`, unnormalized). ε and the
kernel are read from `target.options.inference_method`. `train_target` shares
`_train_reference_summary` (in `BSL.jl`) with BSL but keeps the training covariance;
`finalize_target_covariance` stores its regularized factorization (identity placeholder under
`NoResampling`). `finalize_inference_method(::ABC, ...)` resolves `:scaled` →
`WeightedEuclidean(1 ./ scale.^2)` (training sd with a resampler, pilot MAD under `NoResampling`;
zero scales → smallest positive scale) and `:mahalanobis` → `Mahalanobis(inv(regularized cov))`,
sets a `NaN` ε to a pilot-distance quantile, and `@set`s the resolved method into both
`target.options` and `target.data.options`. The pilot (`pilot_simulations(target, n_pilot; rng)` →
`(params, summaries, distances)`; internally `_pilot_summaries` then `_pilot_distances`, so scales are
set before distances without re-simulating) samples the priors and throws if any active parameter has
no prior. ABC rejects `standardize`, `MCMCOptions.loss_function` and the deprecated 5-arg `TargetData`.
ABC-MCMC extras (bottom of `ABC.jl`): AM/DRAM keep the current state's distances in
`MCMCState.current_distances` via dispatch hooks (`_commit_current_distances!` after every evaluation
that becomes the current state, `_store_method_state!`; no-ops for GSL/BSL) and store them in
`ResultsBuffer.distance_chain` (`n_sim × nsteps`; `0 × nsteps` otherwise), plus `epsilon_chain` and
the final `epsilon` (NaN for GSL/BSL). `EpsilonAdaptation` (Vihola & Franks 2020, AM only; `mcmcrun`
rejects it with DRAM) updates ε in the first `n_adapt` steps via `_adapt_epsilon`, rebuilding the
local target with `_replace_inference_method` and recomputing the current log-target from the stored
distances. `abc_post_correct(results, target, delta; burnin)` gives post-correction weights.

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
   `method_buffer` from `allocate_method_buffer`: `(summary_length, n_sim)` under BSL, `n_sim` distances under ABC, empty under GSL), finalizes bin edges per summary, then computes the
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
6. **MCMC** (`src/mcmc/`) — `mcmcrun(target, mcmc_options)` (`runner.jl`) runs `target.model` and drives the chain via
   a pluggable `mcmc_algorithm` (`AM.jl` = adaptive Metropolis, `DRAM.jl` = delayed rejection AM),
   each implemented as a callable struct `(target, model, state, mcmc_options, results_buffer) ->
   (results, state)`. Per step, the param-level `calculate_loss` (`loss.jl`: prior, model update, noise)
   calls `calculate_loss(target.options.inference_method, ...)` from `src/inference/`: under `GSL()` it
   simulates once (or `n_loss_evals` times, averaged), resamples/summarizes via
   `calculate_simulated_statistics` (honoring `n_summaries`), and evaluates `method.loss(sim, obs_mean,
   cov_factorization)` against the *fixed* target from `TargetData`; under `BSL()` it runs `n_sim` fresh
   simulations (each resampled/summarized the same way, so any resampler works unchanged), estimates the
   mean/covariance *from the spread across those simulations*, and evaluates the same loss callable with
   that simulated factorization (no copy of `target`) minus `0.5*logdet`. The loss callables
   (`loss_functions.jl`) return the raw value; `scale_loss(loss, raw, target)` applies standardization
   (GSL only) and `inverse_scaling`. `MCMCOptions.loss_function` defaults to `nothing`; if set it must
   equal the method's loss or `mcmcrun` throws. `regularized_inverse`
   (`utils/utils.jl`) is the shared ridge-regularization-then-invert helper used by both engines.
   Priors (`prior_eval.jl`) are `NamedTuple`s keyed by parameter name, valued as `Distributions.jl`
   distributions or `nothing` (flat/uninformative). `mcmcrun` catches exceptions mid-chain and returns
   the partial chain rather than losing progress. `AM` supports a "noisy likelihood" heuristic
   (`update_interval`, `discard_noisy_updates`) to detect and kick the chain out of a run of
   suspiciously good proposals (`update_interval <= 0` disables it; DRAM has none; BSL and ABC are exact
   pseudo-marginal only with it off and `reevaluate_current_loss = false`); `reevaluate_current_loss` is particularly relevant under BSL, where
   the current and proposed states' covariances are independent per-step Monte Carlo estimates.

