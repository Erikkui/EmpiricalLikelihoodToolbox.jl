# EmpiricalLikelihoodToolbox
Toolbox for performing Bayesian inference using empirical likelihoods on simulator-based (intractable-likelihood) models. Inference is based on user-selectable summary statistics and one of two likelihood-approximation engines: Gaussian Subset Likelihood (GSL) (Haario et al., 2015) or Bayesian Synthetic Likelihood (BSL) (Wood, 2010). See also Kuitunen (2026) for definitions of the majority of the summary statistics.

## Usage
### Installation
Enter Julia package manager (`]`), and copy the following: `add https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl.git`

In your script, import the package: `using EmpiricalLikelihoodToolbox`

### Code example
```julia
using EmpiricalLikelihoodToolbox

function main()
    axis_unif = :yax                          # Which axis to use as uniform reference in bin creation 
    nrep_training = 5000                      # Number of training samples used in target mean and covariance creation
    n_summaries = 1                           # Number of summary vectors to generate for mean calculation in MCMC phase
    chain_length = 50000                      # MCMC chain length

    Ndata = 1000                              # Number of synthetic observations
    dt_obs = 1.0                              # Time step between observations

    model = Lorenz63Model( dt_obs = dt_obs )                # Lorenz 63 model
    data = solve_model( model, Ndata*dt_obs )               # Solver for Lorenz model uses DifferentialEquations.jl

    resampler = RademacherSplit()                      # 50-50 resampling
    objective_fun = LogLikelihood()                       # Standard log-likelihood function as an objective function

    # Prior distributions, uninformative priors as nothing. Supports distributions from Distributions.jl.
    # Priors are defined as named tuples, where the keys are the parameter names
    param_names, param_values = get_active_model_params( model )    # Get "active" parameters, ie. those that are being inferred
    prior_distributions = tuple(                                            
      Uniform(0.0, 30.0), 
      Uniform(0.0, 50.0), 
      nothing 
      )
    priors = NamedTuple{ param_names }( prior_distributions )

    npar = length( param_values )
    initial_params = param_values .+ 0.01 .* randn( npar )    # Initial parameters as true parameter values with small added perturbations

    summary_statistics = JointSummaryStatistics(                      # Wrapping summaries into single joint summary
        CIL( 10 ),
        CILDiff( 10, 1, dt_obs ),
        IDDiff( 10, 1, 1, dt_obs ),
        )

    methods_options = MethodsOptions(                        # Collecting defined options into single struct
        N_obs = Ndata,
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        inference_method = GSL(),                 # GSL() is the default; shown here for clarity
        )

    mcmc_options = MCMCOptions(                               # Defining MCMC options
        nsteps = chain_length,
        mcmc_algorithm = AM( 0.01, 50 ),      # Adaptive metropolis with initial proposal width 0.01 and adaptation interval of 50 samples
        update_interval = 50,                 # Update interval for noisy MCMC heuristic; if set larger than chain length, essentially disables heuristic
        loss_function = objective_fun,
        initial_params = initial_params
        )

    # Create target data struct "target". Also ouputs training summaries for intermediary plotting, checks etc.
    target, training_summaries = TargetData( data, summary_statistics, methods_options )    

    # MCMC
    results, state = mcmcrun( target, model, mcmc_options )      

end
```

### Bayesian Synthetic Likelihood (BSL)

By default `MethodsOptions` uses `GSL()`: the target mean and covariance are estimated once from the *observed* data (by resampling it `training_resamplings` times), and are then compared against a single simulated summary at every MCMC step.

Passing `inference_method = BSL(n_sim = k)` instead switches to Bayesian Synthetic Likelihood: the observed summary becomes a single fixed reference value, and the mean/covariance used in the likelihood are instead re-estimated at *every* MCMC step from `k` fresh simulations at the current parameter proposal. This trades a large amount of extra simulation cost (`k` model simulations per MCMC step, instead of one) for a likelihood approximation that adapts to how the summary statistic's variability changes across parameter space, which GSL's fixed target covariance cannot capture.

```julia
methods_options = MethodsOptions(
    N_obs = Ndata,
    resampling_type = RademacherSplit(),      # any resampler works, see below
    inference_method = BSL( n_sim = 10 ),     # 10 fresh simulations per MCMC step
    )
```

Everything else (summary statistics, resamplers, priors, MCMC algorithms, loss functions) is used identically under BSL and GSL — only `MethodsOptions.inference_method` changes. A few things to keep in mind:

* **Any resampler works**, including length-changing ones like `RademacherSplit`. Each of the `n_sim` simulated datasets is resampled/summarized exactly as GSL does at MCMC time (honoring `n_summaries` as well, if set above 1), so two-set summaries such as `CIL`/`ID`/`ChamferECDF` keep the same meaning under BSL as under GSL.
* **`NoResampling()`** (see [Resamplers](#resamplers)) is a useful `resampling_type` choice when your summary statistics don't need a train/test split at all (eg. plain `StandardECDF`): it skips resampling entirely and uses all available data, both for the one-off observed reference statistic and for each simulated dataset.
* **`standardize = true` is not supported with `NoResampling()`** and raises an error, under both GSL and BSL — `NoResampling()` is deterministic, so every training draw is identical and there is no spread to standardize against. Use any other resampler (eg. `RademacherSplit`, `StandardBootstrap`) if you need standardization; under BSL this still works because `train_target` reuses GSL's resampling loop to compute the fixed reference statistic whenever `resampling_type` is not `NoResampling()` (see above).
* **`covariance_type`** is ignored under BSL for the same reason: the covariance always comes from the spread of the `n_sim` simulated summaries, never from resampling the observed data.
* Because BSL's covariance at the current and proposed parameters are each independent Monte Carlo estimates, consider setting `MCMCOptions( reevaluate_current_loss = true, ... )` so the "current" state's log-likelihood is refreshed at every step rather than compared against a stale estimate from a previous proposal — this reduces acceptance-ratio bias that is otherwise a known issue with synthetic-likelihood MCMC.
* Runtime scales directly with `n_sim` (and, for two-set summaries, with `N_obs^2` per simulation on top of that) — tune `n_sim` with the cost of one model simulation in mind.

### Summary Statistics

Currently, ten summary statistics are available:
1. Standard ECDF
   - Initialization: `StandardECDF(nbin)`
   - Calculates an empirical cumulative distribution function from data
   - Defined by number of bins at which the eCDF is evaluated. Number of bins (`nbin`) are required as an argument.
   - Supports multidi-dimensional data via call `StandardECDF(nbin, ndim)`
2. Standard ECDF using numerical derivatives
   - Initialization: `StandardECDFDiff(nbin, diff_order, dt_obs)`
   - Same as above, but calculates the eCDF from the numerical derivative of the data
   - In addition to `nbin`, requires the order of differences (`diff_order`) and time step (`dt_obs`) between observations as arguments
3. Correlation Integral Likelihood (CIL)
   - Initialization: `CIL(nbin)`
4. Correlation Integral Likelihood using numerical derivatives 
   - Initialization: `CILDiff(nbin, diff_order, dt_obs)`
5. ECDF from Intrinsic Dimension nearest-neighbor ratios (ID)
   - Initialization: `ID(nbin, neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple ID ratio
6. ECDF from Intrinsic Dimension nearest-neighbor ratios using numerical derivatives
   - Initialization: `IDDiff(nbin, neighbors, diff_order, dt_obs)`
7. Chamfer Distance
   - Initialization: `ChamferDistance(neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple Chamfer distances
8. Chamfer Distance using numerical derivatives
    - Initialization: `ChamferDistanceDiff(neighbors, diff_order, dt_obs)`
9. ECDF from Chamfer Distances 
    - Initialization: `ChamferECDF(nbin, neighbors)`
    - Support multiple neighbors simlar to basic Chamfer Distance
10. Cumulative Sum
    - Initialization: `CumulativeSum(contracting_window)`
    - Calculates the normalized cumulative sum of data, condensed into non-overlapping windows
    - Takes the running cumulative sum of the a set, sums it within consecutive windows, and divides the result by the final value
    - CURRENTLY DOES NOT SUPPORT MULTI-DIMENSIONAL DATA!

Summary statistics must be wrapped into a `JointSummaryStatistics()` struct, eg. `JointSummaryStatistics(CIL(10), CILDiff(10, 1, 1.0))`, as seen in the code example above.

#### Reusing bin edges

By default, a summary statistic's bin edges (`nbin` of them) are computed automatically from the data the first time `TargetData(...)` is called, and stored on the summary object (`stat.bins`). Every eCDF-based summary statistic above also accepts those bin edges directly instead of `nbin`, which is useful for reusing bins already computed from a previous run (eg. `some_stat.bins[1]`) instead of recomputing them:

* `StandardECDF(bins)`, `StandardECDFDiff(bins, diff_order, dt_obs)`, `CIL(bins)`, `CILDiff(bins, diff_order, dt_obs)` — `bins` is a plain vector of bin edges (`AbstractVector{<:Real}`).
* `ID(bins, neighbors)`, `IDDiff(bins, neighbors, diff_order, dt_obs)`, `ChamferECDF(bins, neighbors)` — since these support multiple `neighbors`, `bins` is either a plain vector (reused for the single given `neighbors::Int`), or a vector of one bins vector per neighbor (`bins[ii]` for `neighbors[ii]`) when `neighbors` is itself a vector; in the latter case `length(bins)` must equal `length(neighbors)`, and every per-neighbor bins vector must have the same length.

```julia
# Compute bins once, reuse them for a second (eg. faster) TargetData construction
target, _ = TargetData( data, JointSummaryStatistics( CIL(10) ), methods_options )
cached_bins = target.summary_statistics.statistics[1].bins[1]

summary_statistics_2 = JointSummaryStatistics( CIL( cached_bins ) )
```

### Resamplers

Resamplers decide how a dataset is split into an "x" and a "y" set for two-set summaries (`CIL`, `ID`, `ChamferECDF`), and are also used to build GSL's target sampling distribution and, under BSL, each simulation's resampled summary. Five resamplers are currently available:

* `RademacherSplit()` — splits the available indices into two equal halves at random. Length-changing (the x/y sets are half the size of the original data). This is the default `resampling_type`.
* `ContiguosBlockSplit(timeseries_block_size=100)` — samples one random contiguous block of the given size as the "x" set and everything else as "y". Length-changing; intended for time series data where random reshuffling (as in `RademacherSplit`) would destroy temporal structure. Has not been tested as extensively as `RademacherSplit` and may contain bugs.
* `StandardBootstrap()` — draws two independent bootstrap samples (with replacement), each the same size as the original data. Length-preserving.
* `NoResampling()` — returns all available data, unchanged, as both the "x" and "y" set. Length-preserving and deterministic (no randomness at all). Mainly useful for BSL (see above) or for any deterministic, non-resampled evaluation of a summary statistic.
* `MovingBlockBootstrap(block_size=100)` — a moving block bootstrap : builds each of the x/y sets by concatenating random contiguous blocks of `block_size` observations (sampled with replacement) until reaching the original data length, truncating the last block to fit exactly. Length-preserving — unlike `ContiguosBlockSplit`, both x and y always have the same length as the original data. Preserves short-range (within-block) temporal correlation, unlike `RademacherSplit`/`StandardBootstrap` which resample individual points and destroy it entirely. `block_size=1` degenerates to a per-point bootstrap (equivalent to `StandardBootstrap`); `block_size=N_obs` degenerates to a single deterministic block in the data's original order.

### Additional settings

#### Inferring a subset of parameters

User can choose to infer only a subset of parameters, in which case the the remaining parameters are left constant during MCMC. This is done by providing a tuple of parameters for the model constructor:

```julia
    active_parameters = ( :sigma, :rho )
    model = Lorenz63Model( dt_obs = dt_obs, active_parameters = active_parameters )
```

#### `MethodsOptions` settings

The following keyword settings are input to the `MethodsOptions` struct:
* `N_obs::Int`: Number of observations in the (simulated) dataset. Required, no default.
* `resampling_type`: Which resampler to use, see [Resamplers](#resamplers). Default: `RademacherSplit()`.
* `axis_uniform::Symbol`: How ECDF-style summaries pick their bin edges — `:xax` (bins uniform in the data's own units), `:yax` (bins chosen so the eCDF is approximately uniform, via inverse-CDF), or `:log` (log-spaced bins). Default: `:xax`.
* `bins_resamplings::Int`: Number of resamplings used to determine bin edges for two-set ECDF-style summaries (`CILDiff`, `ID`, `ChamferECDF`, ...). Default: `40`.
* `training_resamplings::Int`: Number of times to resample the observed data when estimating GSL's target mean/covariance (or BSL's fixed reference statistic, when `resampling_type` is not `NoResampling()`). Default: `1000`.
* `covariance_type::Symbol`: How GSL estimates the target covariance from the `training_resamplings` draws — `:cov` (sample covariance) or `:donsker` (Donsker's-theorem covariance derived from the mean eCDF). Ignored under BSL. Default: `:cov`.
* `n_summaries::Int`: Number of summaries to generate for each proposal parameter at MCMC time. When `n_summaries` > 1, their mean is calculated (this happens per simulated dataset under BSL as well). Default: `1`.
* `n_loss_evals::Int`: Number of loss evaluations to calculate for a proposal parameter under GSL. Each loss evaluation involves calculating `n_summaries` summaries from a fresh simulation, after which loss is calculated; the mean of the evaluated losses is then used in MCMC acceptance. Not used under BSL, which always aggregates over `n_sim` simulations instead. Default: `1`.
* `standardize::Bool`: Whether to perform z-score standardization for the summaries. Not supported with `NoResampling()` (no training-draw variability to standardize against), under either GSL or BSL. Default: `false`.
* `ecdf_calculation_type::Symbol`: `:default` for the raw empirical CDF, or `:kernel_smoothed` for a Gaussian-kernel-smoothed variant. Default: `:default`.
* `embedding_dim`: Embedding lag(s) applied to simulated data before summarization (via `embedding()`); `0` disables embedding. For `embedding_type = :delay`, a single lag or a collection of lags, each contributing one extra dimension. For `embedding_type = :diff`, a single lag `tau`. Default: `0`.
* `embedding_type::Symbol`: `:delay` for time-delay embedding (the series plus one or more lagged copies), or `:diff` for the series paired with its lag-`tau` difference (`y_t`, `y_t - y_{t-tau}`). Default: `:delay`.
* `inference_method`: `GSL()` or `BSL(n_sim=...)`, see [Bayesian Synthetic Likelihood (BSL)](#bayesian-synthetic-likelihood-bsl). Default: `GSL()`.
* `verbose::Bool`: Print progress bars/diagnostics during target training and MCMC. Default: `false`.

#### MCMC algorithms, loss functions, and `MCMCOptions` settings

Two MCMC algorithms are available for `mcmc_algorithm`:
* `AM(proposal_width=0.01, adaptation_interval=50)` — adaptive Metropolis, adapting the proposal covariance from the running chain covariance every `adaptation_interval` steps.
* `DRAM(proposal_width=0.01, adaptation_interval=50, n_stages=2, proposal_scale=[1.0, 0.5])` — delayed rejection adaptive Metropolis: on rejection, up to `n_stages` progressively smaller-scaled proposals are attempted before moving on.

Two loss functions are available for `loss_function`:
* `LogLikelihood(scaling_parameter=1.0)` — the standard GSL/BSL Gaussian log-likelihood, `-0.5*(x - x*)ᵀΣ⁻¹(x - x*)`, comparing a summary against the target mean/covariance (scaled by `1/scaling_parameter`).
* `RobustChamfer(scaling_parameter=1.0)` — an alternative loss geared towards Chamfer-distance-based summaries.

The following keyword settings are input to the `MCMCOptions` struct:
* `nsteps::Int`: MCMC chain length. Default: `1000`.
* `mcmc_algorithm`: `AM(...)` or `DRAM(...)`, see above. Required, no default.
* `loss_function`: `LogLikelihood(...)` or `RobustChamfer(...)`, see above. Default: `LogLikelihood()`.
* `initial_params`: Initial parameter values for the chain. Default: `nothing`, in which case the model's active parameter values are perturbed by 1% noise.
* `update_interval::Int`: Number of consecutive rejections after which the "noisy likelihood" heuristic kicks in and recalculates the current state's loss from a fresh simulation. Default: `50`.
* `discard_noisy_updates::Bool`: When the heuristic above triggers, whether to roll the chain back to its last accepted state (`true`) or just recalculate in place (`false`). Default: `false`.
* `reevaluate_current_loss::Bool`: Recalculate the current state's loss from a fresh simulation at every step, before proposing. Recommended for BSL, see [above](#bayesian-synthetic-likelihood-bsl). Default: `false`.
* `likelihood_noise_scale::Float64`: Standard deviation of artificial Gaussian noise added to every loss evaluation, useful for testing MCMC robustness to a noisy likelihood. Default: `NaN`, treated as `0.0`.

## References
Heikki Haario, Leonid Kalachev, Janne Hakkarainen; Generalized correlation integral vectors: A distance concept for chaotic dynamical systems. Chaos 1 June 2015; 25 (6): 063102. https://doi.org/10.1063/1.4921939

Wood, S. Statistical inference for noisy nonlinear ecological dynamic systems. Nature 466, 1102–1104 (2010). https://doi.org/10.1038/nature09319

Kuitunen, E. Empirical likelihoods for intractable likelihood models. LUT Master's Thesis, LUTPub. (2026). https://lutpub.lut.fi/handle/10024/171299 


[![Build Status](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml?query=branch%3Amain)
