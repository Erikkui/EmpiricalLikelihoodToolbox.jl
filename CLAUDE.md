# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`EmpiricalLikelihoodToolbox` is a Julia package for Bayesian inference on intractable-likelihood
(simulator-based) models using empirical-likelihood-style summary statistics. Inference currently
uses Gaussian Subset Likelihood (GSL, Haario et al. 2015); Bayesian Synthetic Likelihood (BSL) is
planned but not implemented. See the README for the full summary-statistic catalog, references, and
a runnable usage example (`examples/main.jl` mirrors it).

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

The package wires together four kinds of pluggable pieces — **models**, **summary statistics**,
**resamplers**, and **MCMC algorithms/loss functions** — through a small set of container structs
defined in `src/core_types.jl`. `src/EmpiricalLikelihoodToolbox.jl` is a flat `include()` list; there's
no submodule structure, so load order there matters (e.g. `core_types.jl` and `utils/utils.jl` first).

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
   computes the same statistic on numerical derivatives of the data (order `diff_order`, step
   `dt_obs`) instead of the raw series. Multiple summaries are combined via
   `JointSummaryStatistics(stat1, stat2, ...)`, which concatenates their outputs into one vector
   using each summary's `summary_length`.
   Each summary struct implements a small informal interface (grep any `summaries/*.jl` file for the
   pattern): two `calculate_summary_statistic!` methods (one training-time signature taking a single
   `DataContainer`, one MCMC-time signature taking observed + simulated `DataContainer`s), plus
   `allocate_buffer`, `get_summary_length`, `required_diff_order`, `generate_stat_name`, and
   `finalize_summary`. ECDF-based summaries additionally need `get_bin_quantity` and go through
   `initialize_bins`/`bin_select` (`src/utils/bin_calculation.jl`) which auto-selects bin edges from
   resampled training data using IQR-based robust bounds (`axis_uniform` option controls whether bins
   are uniform in `:xax`, uniform in CDF-space via `:yax`, or `:log`-spaced).
3. **Resamplers** (`src/utils/resamplers.jl`) — callable structs (`RademacherSplit` 50/50 split,
   `ContiguosBlockSplit` contiguous block for time series, `StandardBootstrap`) subtyping either
   `LengthPreservingSampler` or `LengthChangingSampler`. Called as `resampler(data_container, options,
   index_cache)` → `(x_inds, y_inds)`, used both to build the training distribution of summaries and,
   per MCMC step, to resample simulated data before comparing to the observed target.
4. **`MethodsOptions`** (`core_types.jl`) bundles cross-cutting config: which resampler/ECDF
   calculation to use, bin count, number of training resamplings, `n_summaries`/`n_loss_evals`
   (noise-reduction knobs — average multiple resampled summaries / multiple loss evaluations per
   MCMC step), `standardize` (z-score the loss), and `embedding_dim` (time-delay embedding via
   `embedding()` in `utils/utils.jl`, applied to simulated data before summarization).
5. **`TargetData(data, summary_statistics, methods_options)`** (`src/mcmc/target.jl`) is the main
   entry point that ties everything together: builds a `DataContainer`, precomputes numerical
   derivatives if any summary needs them, allocates all buffers up front (`BufferContainer` — the
   package is written to avoid allocations in the MCMC hot loop), finalizes bin edges per summary,
   then repeatedly resamples the observed data (`training_resamplings` times) to estimate the mean
   summary vector and its covariance (`:cov` or `:donsker`, regularized before inversion) that
   becomes the GSL target. Returns `(target::TargetData, training_summaries)`.
6. **MCMC** (`src/mcmc/`) — `mcmcrun(target, model, mcmc_options)` (`runner.jl`) drives the chain via
   a pluggable `mcmc_algorithm` (`AM.jl` = adaptive Metropolis, `DRAM.jl` = delayed rejection AM),
   each implemented as a callable struct `(target, model, state, mcmc_options, results_buffer) ->
   (results, state)`. Per step, `calculate_loss` (`loss.jl`) simulates data from the proposed
   parameters, resamples/summarizes it (mirroring the training-time resampling), and compares to the
   target mean/covariance via a `loss_function` (`loss.jl` defines the struct-call interface;
   `loss_functions.jl` implements `LogLikelihood` and `RobustChamfer`). Priors (`prior_eval.jl`) are
   `NamedTuple`s keyed by parameter name, valued as `Distributions.jl` distributions or `nothing`
   (flat/uninformative). `mcmcrun` catches exceptions mid-chain and returns the partial chain rather
   than losing progress. `AM` supports a "noisy likelihood" heuristic (`update_interval`,
   `discard_noisy_updates`) to detect and kick the chain out of a run of suspiciously good proposals.

