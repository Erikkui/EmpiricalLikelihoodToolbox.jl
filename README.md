# EmpiricalLikelihoodToolbox
Toolbox for performing Bayesian inference using empirical likelihoods on simulator-based (intractable-likelihood) models. Inference is based on user-selectable summary statistics and one of three likelihood-approximation engines: Gaussian Subset Likelihood (GSL) (Haario et al., 2015), Bayesian Synthetic Likelihood (BSL) (Wood, 2010), or Approximate Bayesian Computation (ABC-MCMC) (Marjoram et al., 2003). See also Kuitunen (2026) for definitions of the majority of the summary statistics.

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
    objective_fun = LogLikelihood()                       # Standard log-likelihood function as an objective function, stored in the inference method

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
        CILDiff( 10, 1 ),                     # Derivatives use the model's dt_obs
        IDDiff( 10, 1, 1 ),
        )

    methods_options = MethodsOptions(                        # Collecting defined options into single struct
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        inference_method = GSL( loss = objective_fun ),   # GSL() is the default; the loss lives in the inference method
        )

    mcmc_options = MCMCOptions(                               # Defining MCMC options
        nsteps = chain_length,
        mcmc_algorithm = AM( 0.01, 50 ),      # Adaptive metropolis with initial proposal width 0.01 and adaptation interval of 50 samples
        update_interval = 50,                 # Update interval for noisy MCMC heuristic; set to 0 to disable the heuristic
        initial_params = initial_params
        )

    # Create target data struct "target". Also ouputs training summaries for intermediary plotting, checks etc.
    # data is the raw series: TargetData embeds it (embedding_dim/embedding_type) like every simulation,
    # and takes N_obs from its length. The model supplies dt_obs, the time step of the Diff summaries'
    # derivatives, and is stored in the target as the simulator mcmcrun runs.
    target, training_summaries = TargetData( data, summary_statistics, methods_options, model )

    # MCMC
    results, state = mcmcrun( target, mcmc_options )

end
```

### Bayesian Synthetic Likelihood (BSL)

By default `MethodsOptions` uses `GSL()`: the target mean and covariance are estimated once from the *observed* data (by resampling it `training_resamplings` times), and are then compared against a single simulated summary at every MCMC step.

Passing `inference_method = BSL(k)` (or `BSL(n_sim = k)`) instead switches to Bayesian Synthetic Likelihood: the observed summary becomes a single fixed reference value, and the mean/covariance used in the likelihood are instead re-estimated at *every* MCMC step from `k` fresh simulations at the current parameter proposal. This trades a large amount of extra simulation cost (`k` model simulations per MCMC step, instead of one) for a likelihood approximation that adapts to how the summary statistic's variability changes across parameter space, which GSL's fixed target covariance cannot capture.

```julia
methods_options = MethodsOptions(
    resampling_type = RademacherSplit(),      # any resampler works, see below
    inference_method = BSL( n_sim = 10 ),     # 10 fresh simulations per MCMC step
    )
```

Everything else (summary statistics, resamplers, priors, MCMC algorithms) is used identically under BSL and GSL — only `MethodsOptions.inference_method` changes. The loss function is set in the method: `GSL( loss = RobustChamfer() )`, `BSL( 10; loss = LogLikelihood( 2.0 ) )`. A few things to keep in mind:

* **Any resampler works**, including length-changing ones like `RademacherSplit`. Each of the `n_sim` simulated datasets is resampled/summarized exactly as GSL does at MCMC time (honoring `n_summaries` as well, if set above 1), so two-set summaries such as `CIL`/`ID`/`ChamferECDF` keep the same meaning under BSL as under GSL.
* **`NoResampling()`** (see [Resamplers](#resamplers)) is a useful `resampling_type` choice when your summary statistics don't need a train/test split at all (eg. plain `StandardECDF`): it skips resampling entirely and uses all available data, both for the one-off observed reference statistic and for each simulated dataset.
* **`standardize = true` is not supported under BSL** and raises an error: the standardization is calibrated against a placeholder covariance, which is inconsistent with the covariance BSL estimates from simulations. Under GSL it is also not supported with `NoResampling()`, which is deterministic, so every training draw is identical and there is no spread to standardize against.
* **`RobustChamfer` is not supported under BSL** (it ignores the covariance BSL estimates); use `LogLikelihood`.
* **`covariance_type`** is ignored under BSL for the same reason: the covariance always comes from the spread of the `n_sim` simulated summaries, never from resampling the observed data.
* Because BSL's covariance at the current and proposed parameters are each independent Monte Carlo estimates, consider setting `MCMCOptions( reevaluate_current_loss = true, ... )` so the "current" state's log-likelihood is refreshed at every step rather than compared against a stale estimate from a previous proposal — this reduces acceptance-ratio bias that is otherwise a known issue with synthetic-likelihood MCMC. BSL is an exact pseudo-marginal method only with the noisy-likelihood heuristic off (`update_interval <= 0`) and `reevaluate_current_loss = false`.
* Runtime scales directly with `n_sim` (and, for two-set summaries, with `N_obs^2` per simulation on top of that) — tune `n_sim` with the cost of one model simulation in mind.

### Approximate Bayesian Computation (ABC)

Passing `inference_method = ABC(...)` runs ABC-MCMC (Marjoram et al., 2003) inside the same `mcmcrun`. At every MCMC step, `n_sim` datasets are simulated at the proposed parameters, each is resampled/summarized exactly as under GSL, and its distance `ρ` to the observed summary is computed. The log-likelihood is `log( mean( K_ε(ρ_j) ) )`: the kernel average is an unbiased estimate of the ABC likelihood, so the chain is a pseudo-marginal sampler of the ABC posterior.

```julia
priors = ( mu = Uniform( -5.0, 5.0 ), sigma = Uniform( 0.1, 5.0 ) )   # proper priors: the pilot samples them
methods_options = MethodsOptions(
    resampling_type = NoResampling(),
    inference_method = ABC( epsilon_quantile = 0.05 ),   # ε = 5 % quantile of prior-predictive pilot distances
    )
target, _ = TargetData( data, summary_statistics, methods_options, model; priors = priors )
println( target.options.inference_method )               # eg. ABC(ε=0.9126), the resolved method
```

`ABC` keyword settings:
* `epsilon::Float64`: The tolerance ε. `NaN` sets it from the pilot quantile. `ABC( 0.1 )` is short for `ABC( epsilon = 0.1 )`. Default: `NaN`.
* `epsilon_quantile::Float64`: The quantile of the finite pilot distances used as ε when `epsilon` is `NaN`. Default: `0.05`.
* `n_pilot::Int`: The number of prior-predictive pilot simulations. Default: `1000`.
* `distance`: `:scaled`, `:mahalanobis`, or any callable `(simulated, observed) -> Float64` (eg. a Distances.jl metric such as `Euclidean()`). Default: `:scaled`.
* `kernel`: `UniformKernel()` (`ρ ≤ ε`), `GaussianKernel()` (`exp(-½(ρ/ε)²)`) or `EpanechnikovKernel()` (`1 - (ρ/ε)²` for `ρ < ε`). Default: `UniformKernel()`.
* `n_sim::Int`: The number of simulations per MCMC step. Default: `1`.
* `adapt_epsilon`: `nothing` (fixed ε) or `EpsilonAdaptation( n_adapt = ..., target_acceptance = 0.1, step_exponent = 2/3 )`, which adapts ε during the first `n_adapt` AM iterations towards the target acceptance rate (Vihola & Franks, 2020): `log ε ← log ε + k^(-2/3) (α* - α_k)`, then freezes it. AM only (DRAM raises an error). Default: `nothing`.

`TargetData` resolves the distance and ε, and stores the resolved method in `target.options.inference_method`:
* `:scaled` becomes a weighted Euclidean distance with weights `1/scale²`. With a resampler, `scale` is the per-component sd of the training summaries. Under `NoResampling()`, `scale` is the normalized MAD of the pilot summaries. A component with zero scale (constant, eg. an ECDF bin at 0 or 1) gets the smallest positive scale.
* `:mahalanobis` uses the inverse of the regularized training covariance. It needs a resampler, because the training covariance is zero under `NoResampling()`.
* The pilot (`pilot_simulations( target, n_pilot; rng )`, exported) runs only when ε is `NaN` or when `:scaled` needs MAD scales. It draws the parameters from the priors, so every active parameter needs a proper prior (a `Distribution`, not `nothing`). Pass `rng` to `TargetData` to make the pilot reproducible.

A few things to keep in mind:
* ABC-MCMC is an exact pseudo-marginal method only with the noisy-likelihood heuristic off (`update_interval <= 0`) and `reevaluate_current_loss = false`. With ε adaptation, only the chain after the `n_adapt` adaptation steps is a valid ABC-MCMC chain, at the final ε (`results.epsilon`).
* ABC results also hold `results.distance_chain` (`n_sim × nsteps`, the distances of each stored state), `results.epsilon_chain` (ε at each step) and `results.epsilon` (the final ε). `abc_post_correct( results, target, delta; burnin = n_adapt )` returns `(weights, ess, samples)`: importance weights `mean K_δ(ρ) / mean K_ε(ρ)` (normalized) that correct the chain to a smaller tolerance `delta ≤ ε` (Vihola & Franks, 2020). With the uniform kernel and `n_sim = 1`, this keeps the samples with `ρ ≤ delta`.
* ABC has no loss function. `standardize = true`, `MCMCOptions( loss_function = ... )` and the deprecated `TargetData( data, stats, options, loss, model )` raise an error.
* The distance compares each simulated summary with `target.obs_mean`. For two-set summaries (`CIL`, `ID`, `ChamferECDF`, ...) this is the observed-vs-observed reference, as under GSL.
* Runtime scales with `n_sim`. A larger `n_sim` lowers the variance of the likelihood estimate, which raises the acceptance rate at a fixed ε.

#### ABC rejection and ABC-SMC

Two population samplers use the same ABC target (built with `inference_method = ABC(...)` and proper priors) outside `mcmcrun`:

```julia
result = abc_rejection( target, 10_000; n_keep = 500, rng = Xoshiro( 1 ) )   # or epsilon = 0.5; default: the target's ε
result.params, result.distances, result.epsilon, result.acceptance_rate

smc_options = ABCSMCOptions( n_particles = 1000, drop_fraction = 0.5, c = 0.01, p_acc_min = 0.01 )
result = abc_smc( target, smc_options; rng = Xoshiro( 1 ) )
result.particles, result.distances, result.epsilons, result.acceptance_rates, result.mcmc_repeats, result.n_simulations
```

* `abc_rejection` draws `n_draws` parameters from the priors and keeps those with `ρ ≤ ε`. With `n_keep`, ε is the `n_keep`-th smallest distance.
* `abc_smc` is the replenishment ABC-SMC of Drovandi & Pettitt (2011). Generation 0 draws `n_particles` from the prior. Each generation drops the fraction `drop_fraction` of particles with the largest distances, sets ε_t to the largest kept distance, resamples the dropped particles from the survivors, and moves each copy with `R_t = ⌈log(c) / log(1 - p_acc)⌉` random-walk MH moves (proposal covariance `proposal_scale ×` the survivors' sample covariance). It stops when the MCMC acceptance rate falls below `p_acc_min`, when ε reaches `epsilon_target`, or after `max_generations`. Other settings: `max_mcmc_repeats` (cap on R_t, default 500) and `verbose`.
* `abc_smc` needs the uniform kernel and `n_sim = 1`. It sets its own ε sequence, so the target's ε is not used: give a fixed ε (eg. `ABC( 1.0 )`) to skip the ε pilot in `TargetData`.
* The final particle set contains duplicates. This is expected with replenishment ABC-SMC.
* The results are reproducible for a given `rng` with a deterministic resampler such as `NoResampling()`.

### Summary Statistics

Currently, ten summary statistics are available:
1. Standard ECDF
   - Initialization: `StandardECDF(nbin)`
   - Calculates an empirical cumulative distribution function from data
   - Defined by number of bins at which the eCDF is evaluated. Number of bins (`nbin`) are required as an argument.
   - Supports multidi-dimensional data via call `StandardECDF(nbin, ndim)`
2. Standard ECDF using numerical derivatives
   - Initialization: `StandardECDFDiff(nbin, diff_order)`
   - Same as above, but calculates the eCDF from the numerical derivative of the data
   - In addition to `nbin`, requires the order of differences (`diff_order`) as an argument. The time step of the derivative is the `dt_obs` of the model passed to `TargetData`, so it is set only once, on the model
3. Correlation Integral Likelihood (CIL)
   - Initialization: `CIL(nbin)`
4. Correlation Integral Likelihood using numerical derivatives 
   - Initialization: `CILDiff(nbin, diff_order)`
5. ECDF from Intrinsic Dimension nearest-neighbor ratios (ID)
   - Initialization: `ID(nbin, neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple ID ratio
6. ECDF from Intrinsic Dimension nearest-neighbor ratios using numerical derivatives
   - Initialization: `IDDiff(nbin, neighbors, diff_order)`
7. Chamfer Distance
   - Initialization: `ChamferDistance(neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple Chamfer distances
8. Chamfer Distance using numerical derivatives
    - Initialization: `ChamferDistanceDiff(neighbors, diff_order)`
9. ECDF from Chamfer Distances 
    - Initialization: `ChamferECDF(nbin, neighbors)`
    - Support multiple neighbors simlar to basic Chamfer Distance
10. Cumulative Sum
    - Initialization: `CumulativeSum(contracting_window; transform = identity, scale = nothing)`
    - Calculates the normalized cumulative sum of data, condensed into non-overlapping windows
    - Takes the running cumulative sum of the x set, sums it within consecutive windows, and divides the result by a per-row normalization factor. The final window is dropped.
    - Supports multi-dimensional data: each row is accumulated, contracted and normalized separately, and the outputs are concatenated row after row. All rows share the same resampled indices.
    - `transform` is applied elementwise before accumulating, to observed and simulated data alike, eg. `transform = abs` accumulates absolute increments
    - By default each row is divided by its final contracted value of the observed data. `scale = [s1, s2, ...]` (one entry per row) replaces this with fixed constants
    - Pairs naturally with `MovingBlockBootstrap`: blocks of observations are resampled and then accumulated, which is equivalent to splicing blocks of the cumulative sum

Summary statistics must be wrapped into a `JointSummaryStatistics()` struct, eg. `JointSummaryStatistics(CIL(10), CILDiff(10, 1, 1.0))`, as seen in the code example above.

#### How bin edges are chosen

Bin edges are placed on the series each summary actually evaluates its eCDF on: the raw data for `StandardECDF`, the differences for `StandardECDFDiff`, and the resampled distances or ratios for the two-set summaries (`CIL`, `ID`, `ChamferECDF`, ...). `axis_uniform` selects the spacing:

* `:yax` — the dense eCDF of the data is inverted at the equal-mass midpoint levels `(i - 0.5)/nbin` of its range, so the edges are densest where the eCDF is steepest and no edge sits on the ends of the data (an edge below the minimum or at the maximum would read 0 or 1 in every resample, making the training covariance singular). For discrete data (eg. counts), where several levels fall inside one jump of the eCDF, every edge is snapped onto a distinct data value below the maximum, so each edge reads a different eCDF value.
* `:xax` — uniformly spaced edges between the `0.5/nbin` and `1 - 0.5/nbin` quantiles.
* `:log` — log-spaced edges over the same quantile range, for positive data only.

#### Reusing bin edges

By default, a summary statistic's bin edges (`nbin` of them) are computed automatically from the data the first time `TargetData(...)` is called, and stored on the summary object (`stat.bins`). Every eCDF-based summary statistic above also accepts those bin edges directly instead of `nbin`, which is useful for reusing bins already computed from a previous run (eg. `some_stat.bins[1]`) instead of recomputing them:

* `StandardECDF(bins)`, `StandardECDFDiff(bins, diff_order)`, `CIL(bins)`, `CILDiff(bins, diff_order)` — `bins` is a plain vector of bin edges (`AbstractVector{<:Real}`).
* `ID(bins, neighbors)`, `IDDiff(bins, neighbors, diff_order)`, `ChamferECDF(bins, neighbors)` — since these support multiple `neighbors`, `bins` is either a plain vector (reused for the single given `neighbors::Int`), or a vector of one bins vector per neighbor (`bins[ii]` for `neighbors[ii]`) when `neighbors` is itself a vector; in the latter case `length(bins)` must equal `length(neighbors)`, and every per-neighbor bins vector must have the same length.

```julia
# Compute bins once, reuse them for a second (eg. faster) TargetData construction
target, _ = TargetData( data, JointSummaryStatistics( CIL(10) ), methods_options, model )
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
* `N_obs::Int`: Number of observations in the raw (pre-embedding) series, which is also the length of every simulated dataset. Leave it unset: `TargetData` takes it from the data, and throws if a value passed here disagrees. Default: `0` (unset).
* `resampling_type`: Which resampler to use, see [Resamplers](#resamplers). Default: `RademacherSplit()`.
* `axis_uniform::Symbol`: How ECDF-style summaries pick their bin edges — `:xax` (bins uniform in the data's own units), `:yax` (bins chosen so the eCDF is approximately uniform, via inverse-CDF), or `:log` (log-spaced bins). Default: `:xax`.
* `bins_resamplings::Int`: Number of resamplings used to determine bin edges for two-set ECDF-style summaries (`CILDiff`, `ID`, `ChamferECDF`, ...). Default: `40`.
* `training_resamplings::Int`: Number of times to resample the observed data when estimating GSL's target mean/covariance (or BSL's fixed reference statistic, when `resampling_type` is not `NoResampling()`). Default: `1000`.
* `covariance_type::Symbol`: How GSL estimates the target covariance from the `training_resamplings` draws — `:cov` (sample covariance) or `:donsker` (Donsker's-theorem covariance derived from the mean eCDF). Ignored under BSL. Default: `:cov`.
* `n_summaries::Int`: Number of summaries to generate for each proposal parameter at MCMC time. When `n_summaries` > 1, their mean is calculated (this happens per simulated dataset under BSL as well). Default: `1`.
* `n_loss_evals::Int`: Number of loss evaluations to calculate for a proposal parameter under GSL. Each loss evaluation involves calculating `n_summaries` summaries from a fresh simulation, after which loss is calculated; the mean of the evaluated losses is then used in MCMC acceptance. Not used under BSL, which always aggregates over `n_sim` simulations instead. Default: `1`.
* `standardize::Bool`: Whether to perform z-score standardization of the loss. Not supported under BSL or ABC, nor under GSL with `NoResampling()` (no training-draw variability to standardize against). Default: `false`.
* `ecdf_calculation_type::Symbol`: `:default` for the raw empirical CDF, or `:kernel_smoothed` for a Gaussian-kernel-smoothed variant. Default: `:default`.
* `embedding_dim`: Embedding lag(s) applied before summarization (via `embedding()`), to the observed data by `TargetData` and to every simulated dataset by the same call, so pass `TargetData` the raw series; `0` disables embedding. For `embedding_type = :delay`, a single lag or a collection of lags, each contributing one extra dimension. For `embedding_type = :diff`, a single lag `tau`. Default: `0`.
* `embedding_type::Symbol`: `:delay` for time-delay embedding (the series plus one or more lagged copies), or `:diff` for the series paired with its lag-`tau` difference (`y_t`, `y_t - y_{t-tau}`). Default: `:delay`.
* `inference_method`: `GSL(; loss)`, `BSL(n_sim; loss)` or `ABC(...)`, see [Bayesian Synthetic Likelihood (BSL)](#bayesian-synthetic-likelihood-bsl) and [Approximate Bayesian Computation (ABC)](#approximate-bayesian-computation-abc). Default: `GSL()`.
* `verbose::Bool`: Print progress bars/diagnostics during target training and MCMC. Default: `false`.

#### MCMC algorithms, loss functions, and `MCMCOptions` settings

Two MCMC algorithms are available for `mcmc_algorithm`:
* `AM(proposal_width=0.01, adaptation_interval=50)` — adaptive Metropolis, adapting the proposal covariance from the running chain covariance every `adaptation_interval` steps.
* `DRAM(proposal_width=0.01, adaptation_interval=50, n_stages=2, proposal_scale=[1.0, 0.5])` — delayed rejection adaptive Metropolis: on rejection, up to `n_stages` progressively smaller-scaled proposals are attempted before moving on.

Two loss functions are available, set in the inference method (`GSL( loss = ... )`, `BSL( n_sim; loss = ... )`, default `LogLikelihood()`):
* `LogLikelihood(scaling_parameter=1.0)` — the standard GSL/BSL Gaussian log-likelihood, `-0.5*(x - x*)ᵀΣ⁻¹(x - x*)`, comparing a summary against the target mean/covariance (scaled by `1/scaling_parameter`).
* `RobustChamfer(scaling_parameter=1.0)` — an alternative loss geared towards Chamfer-distance-based summaries.

The following keyword settings are input to the `MCMCOptions` struct:
* `nsteps::Int`: MCMC chain length. Default: `1000`.
* `mcmc_algorithm`: `AM(...)` or `DRAM(...)`, see above. Required, no default.
* `loss_function`: Optional. `nothing` (default) uses the loss of the inference method; if given, it must equal that loss, otherwise `mcmcrun` throws. Under ABC, which has no loss, it must be left unset. The old `TargetData( data, stats, options, loss, model )` signature still works but is deprecated.
* `initial_params`: Initial parameter values for the chain. Default: `nothing`, in which case the model's active parameter values are perturbed by 1% noise.
* `update_interval::Int`: Number of consecutive rejections after which the "noisy likelihood" heuristic kicks in and recalculates the current state's loss from a fresh simulation. `update_interval <= 0` disables the heuristic (AM only; DRAM has no such heuristic). Default: `50`.
* `discard_noisy_updates::Bool`: When the heuristic above triggers, whether to roll the chain back to its last accepted state (`true`) or just recalculate in place (`false`). Default: `false`.
* `reevaluate_current_loss::Bool`: Recalculate the current state's loss from a fresh simulation at every step, before proposing. Recommended for BSL, see [above](#bayesian-synthetic-likelihood-bsl). Default: `false`.
* `likelihood_noise_scale::Float64`: Standard deviation of artificial Gaussian noise added to every loss evaluation, useful for testing MCMC robustness to a noisy likelihood. Default: `NaN`, treated as `0.0`.

## References
Heikki Haario, Leonid Kalachev, Janne Hakkarainen; Generalized correlation integral vectors: A distance concept for chaotic dynamical systems. Chaos 1 June 2015; 25 (6): 063102. https://doi.org/10.1063/1.4921939

Marjoram, P., Molitor, J., Plagnol, V., Tavaré, S. Markov chain Monte Carlo without likelihoods. PNAS 100 (26), 15324–15328 (2003). https://doi.org/10.1073/pnas.0306899100

Drovandi, C. C., Pettitt, A. N. Estimation of parameters for macroparasite population evolution using approximate Bayesian computation. Biometrics 67 (1), 225–233 (2011). https://doi.org/10.1111/j.1541-0420.2010.01410.x

Wood, S. Statistical inference for noisy nonlinear ecological dynamic systems. Nature 466, 1102–1104 (2010). https://doi.org/10.1038/nature09319

Kuitunen, E. Empirical likelihoods for intractable likelihood models. LUT Master's Thesis, LUTPub. (2026). https://lutpub.lut.fi/handle/10024/171299 


[![Build Status](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml?query=branch%3Amain)
