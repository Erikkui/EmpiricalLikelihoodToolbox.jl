#------------Approximate Bayesian Computation (ABC-MCMC, Marjoram et al. 2003). The likelihood is
# replaced by a kernel K_ε(ρ) of the distance ρ between a simulated summary and the observed
# summary (target.obs_mean). With n_sim simulations per step, the average of the kernel values is
# an unbiased estimate of the ABC likelihood, so AM/DRAM run as a pseudo-marginal sampler. The
# sampler is exact (for the ABC posterior) only with the AM noisy-likelihood heuristic off
# (`update_interval <= 0`), `reevaluate_current_loss = false` and a fixed ε: with ε adaptation
# (`EpsilonAdaptation`, AM only), only the chain after the adaptation phase is a valid ABC-MCMC
# chain, at the final ε. The distances of every stored state are kept (`results.distance_chain`), so
# the chain can be post-corrected to a smaller tolerance δ ≤ ε (`abc_post_correct`, Vihola and
# Franks 2020).

#------------Kernels
abstract type AbstractABCKernel end

"""
    UniformKernel()

Uniform ABC kernel (the default): `K_ε(ρ) = 1` if `ρ ≤ ε`, else `0`. With this kernel, ABC-MCMC
accepts a proposal only when its simulated summary is within distance `ε` of the observed summary.

See also [`GaussianKernel`](@ref), [`EpanechnikovKernel`](@ref), [`ABC`](@ref).
"""
struct UniformKernel <: AbstractABCKernel end

"""
    GaussianKernel()

Gaussian ABC kernel: `K_ε(ρ) = exp(-½(ρ/ε)²)`. Never exactly zero, so every proposal has a positive
(but possibly tiny) acceptance probability.

See also [`UniformKernel`](@ref), [`EpanechnikovKernel`](@ref), [`ABC`](@ref).
"""
struct GaussianKernel <: AbstractABCKernel end

"""
    EpanechnikovKernel()

Epanechnikov ABC kernel: `K_ε(ρ) = 1 - (ρ/ε)²` if `ρ < ε`, else `0`.

See also [`UniformKernel`](@ref), [`GaussianKernel`](@ref), [`ABC`](@ref).
"""
struct EpanechnikovKernel <: AbstractABCKernel end

# Log of the *unnormalized* kernel at distance `distance_value` and tolerance `epsilon`. The
# normalizing constants depend on ε only, so they cancel in the MH ratio at a fixed ε, and in
# self-normalized importance weights K_δ/K_ε. A failed simulation has distance Inf, which gives -Inf
# for every kernel.
log_kernel( ::UniformKernel, distance_value::Real, epsilon::Real ) = distance_value <= epsilon ? 0.0 : -Inf

function log_kernel( ::GaussianKernel, distance_value::Real, epsilon::Real )
    scaled_distance = distance_value / epsilon
    return -0.5 * scaled_distance^2
end

function log_kernel( ::EpanechnikovKernel, distance_value::Real, epsilon::Real )
    distance_value < epsilon || return -Inf
    scaled_distance = distance_value / epsilon
    return log1p( -scaled_distance^2 )
end

_kernel_string( kernel ) = string( nameof( typeof( kernel ) ) )


#------------Tolerance adaptation (Vihola and Franks 2020)
"""
    EpsilonAdaptation( ; target_acceptance = 0.1, n_adapt, step_exponent = 2/3 )

Adapt the ABC tolerance ε during the first `n_adapt` AM iterations towards a target acceptance
rate (Vihola and Franks 2020, Biometrika, "On the use of approximate Bayesian computation Markov
chain Monte Carlo with inflated tolerance and post-correction", Algorithm 3). Set it in
`ABC( adapt_epsilon = EpsilonAdaptation( n_adapt = ... ) )`. AM only: DRAM throws.

After the accept/reject step of iteration `k ≤ n_adapt`:

    log ε ← log ε + γ_k ( α* - α_k ),    γ_k = k^(-step_exponent)

with `α*` the `target_acceptance` and `α_k = min( 1, exp( log_ratio ) )` the acceptance probability
of the step (`0` for a `NaN` or `-Inf` log ratio). A low acceptance raises ε, a high one lowers it.
After `n_adapt` iterations ε is frozen, and the final value is returned in `results.epsilon`.

When ε changes, the current state's log-target is recomputed for the new ε from its stored
distances, without simulating again. The `likelihood_noise_scale` noise of `MCMCOptions` is then
not added again.

The chain is a valid ABC-MCMC chain (at the final ε) only after the adaptation phase: discard at
least `n_adapt` samples as burn-in (eg. `abc_post_correct( results, target, delta; burnin = n_adapt )`).

# Fields
- `target_acceptance::Float64`: The target acceptance rate α* (default: 0.1, the value Vihola and Franks recommend in Section 4 and use in their experiments).
- `n_adapt::Int`: The number of adapted iterations (required).
- `step_exponent::Float64`: The exponent r of the step sizes γ_k = k^(-r) (default: 2/3, used by Vihola and Franks). Their convergence result (Assumption 9) needs r in (1/2, 1].

See also [`ABC`](@ref), [`abc_post_correct`](@ref).
"""
struct EpsilonAdaptation
    target_acceptance::Float64
    n_adapt::Int
    step_exponent::Float64

    function EpsilonAdaptation( target_acceptance::Real, n_adapt::Integer, step_exponent::Real )
        if !( 0 < target_acceptance < 1 )
            throw( ArgumentError( "EpsilonAdaptation target_acceptance must be in (0, 1), got $(target_acceptance)." ) )
        end
        if n_adapt < 1
            throw( ArgumentError( "EpsilonAdaptation n_adapt must be at least 1, got $(n_adapt)." ) )
        end
        if !( 0.5 < step_exponent <= 1 )
            throw( ArgumentError( "EpsilonAdaptation step_exponent must be in (1/2, 1], got $(step_exponent)." ) )
        end
        return new( Float64( target_acceptance ), Int( n_adapt ), Float64( step_exponent ) )
    end
end
EpsilonAdaptation( ; target_acceptance = 0.1, n_adapt, step_exponent = 2/3 ) =
    EpsilonAdaptation( target_acceptance, n_adapt, step_exponent )


#------------Method struct
"""
    ABC{D, K, A}

Approximate Bayesian Computation as an inference method (`MethodsOptions.inference_method`). At each
MCMC step, `n_sim` datasets are simulated at the proposed parameters, each is summarized exactly as
under GSL (`calculate_simulated_statistics`), and its distance `ρ` to the observed summary
(`target.obs_mean`) is computed. The log-likelihood is `log( mean( K_ε(ρ_j) ) )`, the log of the
unbiased kernel-average estimate of the ABC likelihood.

`TargetData` resolves the symbolic `distance` and a `NaN` `epsilon`, and stores the resolved method
in `target.options.inference_method` (see `finalize_inference_method`). Resolution rules:
- `:scaled` → `WeightedEuclidean( 1 ./ scale.^2 )`. `scale` is the per-component sd of the training
  summaries when a resampler is used, or the normalized MAD of prior-predictive pilot summaries
  (`pilot_simulations`) under `NoResampling()`. A zero scale (a component that is constant in the
  training or pilot summaries, eg. an ECDF bin at 0 or 1) is replaced by the smallest positive scale.
- `:mahalanobis` → `Mahalanobis( Q )`, `Q` the inverse of the regularized training covariance.
  Needs a resampler: under `NoResampling()` the training covariance is zero.
- any other value is used as is, as a callable `(simulated, observed) -> Float64`, eg. Distances.jl
  `Euclidean()`.
- `epsilon = NaN` → the `epsilon_quantile` quantile of the finite pilot distances.
The pilot (`n_pilot` prior-predictive simulations) needs proper priors for every active parameter.

ABC has no loss function, and `standardize = true` is not supported.

# Fields
- `epsilon::Float64`: The tolerance ε. `NaN` (default) sets it from the pilot quantile.
- `epsilon_quantile::Float64`: The quantile of the pilot distances used as ε when `epsilon` is `NaN` (default: 0.05).
- `n_pilot::Int`: The number of pilot simulations (default: 1000).
- `distance::D`: `:scaled` (default), `:mahalanobis`, or a callable `(simulated, observed) -> Float64`.
- `kernel::K`: `UniformKernel()` (default), `GaussianKernel()` or `EpanechnikovKernel()`.
- `n_sim::Int`: The number of simulations per MCMC step (default: 1).
- `adapt_epsilon::A`: `nothing` (default, fixed ε) or an `EpsilonAdaptation`, which adapts ε during the first iterations of an AM run.

The MCMC results of an ABC run hold the distances of every stored state (`results.distance_chain`,
`n_sim × nsteps`), the ε of every stored state (`results.epsilon_chain`) and the final ε
(`results.epsilon`). The ABC-MCMC chain is exact (for the ABC posterior at ε) only with
`update_interval <= 0`, `reevaluate_current_loss = false` and a fixed ε; with adaptation, only the
samples after the adaptation phase.

# Examples
```julia
method = ABC()                                          # ε from the 5 % pilot quantile, scaled Euclidean distance
method = ABC( 0.1 )                                     # fixed ε = 0.1
method = ABC( epsilon_quantile = 0.01, kernel = GaussianKernel(), n_sim = 5 )
method = ABC( epsilon = 0.5, distance = Euclidean() )   # any Distances.jl metric
method = ABC( adapt_epsilon = EpsilonAdaptation( n_adapt = 2000 ) )
```

See also [`EpsilonAdaptation`](@ref), [`abc_post_correct`](@ref), [`pilot_simulations`](@ref),
[`simulate_distance`](@ref), [`GSL`](@ref), [`BSL`](@ref).
"""
Base.@kwdef struct ABC{D, K, A} <: AbstractInferenceMethod
    epsilon::Float64 = NaN              # NaN → set from the pilot quantile in TargetData
    epsilon_quantile::Float64 = 0.05    # used only when epsilon is NaN
    n_pilot::Int = 1000
    distance::D = :scaled               # :scaled, :mahalanobis, or any (sim, obs) -> Float64 callable
    kernel::K = UniformKernel()
    n_sim::Int = 1
    adapt_epsilon::A = nothing
end
ABC( epsilon::Real; kwargs... ) = ABC(; epsilon = Float64( epsilon ), kwargs... )

# `ABC(ε=0.1234)` once ε is set, `ABC(ε=pilot q=0.05)` before. The kernel is added when it is not
# uniform, n_sim when it is above 1. Distance weights are left out. Stable: used in file names.
function Base.show( io::IO, method::ABC )
    if isnan( method.epsilon )
        print( io, "ABC(ε=pilot q=", method.epsilon_quantile )
    else
        print( io, "ABC(ε=", round( method.epsilon, sigdigits=4 ) )
    end
    if !( method.kernel isa UniformKernel )
        print( io, ", ", _kernel_string( method.kernel ) )
    end
    if method.n_sim > 1
        print( io, ", n_sim=", method.n_sim )
    end
    print( io, ")" )
end


#------------Inference method interface (see GSL.jl)

# Same reference summary as BSL (a single pass under NoResampling, else GSL's training loop), but
# the training covariance is kept: :scaled and :mahalanobis distances are resolved from it.
train_target( ::ABC, statistics, data_container, buffer_container, options ) =
    _train_reference_summary( statistics, data_container, buffer_container, options )

# ABC's loss never reads the target's factorization. When a training covariance exists (a resampler
# is used), its regularized factorization is stored anyway, since :mahalanobis is resolved from it;
# under NoResampling there is none, and an identity placeholder fixes the field's type and size.
finalize_target_covariance( ::ABC, cov_mat, summary_length ) = regularized_cholesky( cov_mat )
finalize_target_covariance( ::ABC, ::Nothing, summary_length ) = cholesky( Matrix( 1.0I, summary_length, summary_length ) )

# Holds the n_sim distances of the latest calculate_loss call (the current proposal's), in
# simulation order. Read target.buffers.method_buffer right after calculate_loss to get them.
allocate_method_buffer( abc::ABC, summary_length ) = zeros( abc.n_sim )

function check_method_options( abc::ABC, options )
    if options.standardize
        throw( ArgumentError( "standardize=true is not supported with ABC (ABC has no loss function to standardize)." ) )
    end
    if abc.n_sim < 1
        throw( ArgumentError( "ABC n_sim must be at least 1, got $(abc.n_sim)." ) )
    end
    if !( isnan( abc.epsilon ) || abc.epsilon > 0 )
        throw( ArgumentError( "ABC epsilon must be positive (or NaN to set it from the pilot quantile), got $(abc.epsilon)." ) )
    end
    if !( 0 < abc.epsilon_quantile < 1 )
        throw( ArgumentError( "ABC epsilon_quantile must be in (0, 1), got $(abc.epsilon_quantile)." ) )
    end
    if abc.n_pilot < 1
        throw( ArgumentError( "ABC n_pilot must be at least 1, got $(abc.n_pilot)." ) )
    end
    if abc.distance isa Symbol && !( abc.distance in ( :scaled, :mahalanobis ) )
        throw( ArgumentError( "Unknown ABC distance :$(abc.distance). Must be :scaled, :mahalanobis or a callable (simulated, observed) -> Float64." ) )
    end
    if abc.distance === :mahalanobis && options.resampling_type isa NoResampling
        throw( ArgumentError( "ABC distance :mahalanobis needs a resampler: under NoResampling the training covariance is zero. Use :scaled or a callable distance." ) )
    end
    if !( isnothing( abc.adapt_epsilon ) || abc.adapt_epsilon isa EpsilonAdaptation )
        throw( ArgumentError( "ABC adapt_epsilon must be nothing or an EpsilonAdaptation, got $(abc.adapt_epsilon)." ) )
    end
    return nothing
end

# Simulate n_sim datasets, store their distances in the method buffer and return the log of the
# kernel average. A failed simulation has distance Inf, ie. kernel value 0: it stays in the average,
# which keeps the estimate unbiased. ε and the kernel are read from `abc`, which the param-level
# calculate_loss takes from target.options.inference_method.
function calculate_loss( abc::ABC, target, model, mcmc_options, rng )
    distances = target.buffers.method_buffer
    kernel = abc.kernel
    epsilon = abc.epsilon

    for jj in 1:abc.n_sim
        distances[jj] = simulate_distance( target, model, rng )
    end

    return _log_kernel_average( kernel, distances, epsilon )
end

# log( mean_j K_ε(ρ_j) ), the log of the kernel-average ABC likelihood estimate.
function _log_kernel_average( kernel, distances, epsilon )
    log_kernels = ( log_kernel( kernel, distance_value, epsilon ) for distance_value in distances )
    n_distances = length( distances )
    return logsumexp( log_kernels ) - log( n_distances )
end


#------------Simulation and distance
_apply_distance( distance, sim_statistic, obs_statistic ) = distance( sim_statistic, obs_statistic )
function _apply_distance( distance::Symbol, sim_statistic, obs_statistic )
    throw( ArgumentError( "ABC distance :$(distance) is not resolved. Build the target with TargetData, which resolves it." ) )
end

"""
    simulate_distance( target, model, rng ) -> Float64

Simulate one dataset from `model`, summarize it as at MCMC time (`calculate_simulated_statistics`)
and return its distance to the observed summary `target.obs_mean`, using the resolved distance in
`target.options.inference_method.distance`. Returns `Inf` when the simulation fails (non-finite
values). Writes only to `target.buffers`, so it does not allocate beyond the simulation itself.

Shared by ABC-MCMC (`calculate_loss`) and the pilot (`pilot_simulations`).
"""
function simulate_distance( target, model, rng )
    options = target.options
    buffers = target.buffers
    distance = options.inference_method.distance

    Rsim_container = create_simulated_data( model, target, buffers, options, rng )
    if isinf( Rsim_container.observations[1] )
        return Inf
    end

    sim_statistic = calculate_simulated_statistics( target, Rsim_container, target.summary_statistics, buffers, options )
    return Float64( _apply_distance( distance, sim_statistic, target.obs_mean ) )
end


#------------Pilot (prior-predictive) simulations

# Draw one value per active parameter, in the order of get_active_model_params. A flat prior
# (`nothing`) cannot be sampled, so every active parameter needs a Distribution.
function _check_samplable_priors( priors, model )
    active_param_names, _ = get_active_model_params( model )
    if isnothing( priors )
        throw( ArgumentError( "Pilot simulations draw parameters from the priors, but priors = nothing. Give a proper prior for every active parameter $(active_param_names)." ) )
    end
    missing_names = Tuple( name for name in active_param_names if isnothing( get( priors, name, nothing ) ) )
    if !isempty( missing_names )
        throw( ArgumentError( "Pilot simulations draw parameters from the priors, but parameter(s) $(missing_names) have no prior (a flat prior cannot be sampled)." ) )
    end
    return active_param_names
end

# Prior-predictive simulations: params (npar × n_pilot) drawn from the priors and the simulated
# summaries (summary_length × n_pilot). A failed simulation leaves a NaN column. Needs no resolved
# distance, so TargetData can compute the :scaled MAD scales from it before any distance exists.
function _pilot_summaries( target, n_pilot::Integer, rng )
    model = target.model
    options = target.options
    buffers = target.buffers
    priors = target.priors

    active_param_names = _check_samplable_priors( priors, model )
    npar = length( active_param_names )

    params = zeros( npar, n_pilot )
    summaries = fill( NaN, target.summary_length, n_pilot )
    param_draw = zeros( npar )

    for jj in 1:n_pilot
        for (ii, name) in enumerate( active_param_names )
            param_draw[ii] = rand( rng, priors[name] )
        end
        params[:, jj] .= param_draw

        pilot_model = update_model_parameters( model, param_draw )
        Rsim_container = create_simulated_data( pilot_model, target, buffers, options, rng )
        if isinf( Rsim_container.observations[1] )
            continue    # failed simulation: the NaN column is kept, its distance becomes Inf
        end

        sim_statistic = calculate_simulated_statistics( target, Rsim_container, target.summary_statistics, buffers, options )
        summaries[:, jj] .= sim_statistic
    end

    return params, summaries
end

# Distance of every pilot summary to the observed summary. Failed simulations (non-finite columns)
# get distance Inf.
function _pilot_distances( distance, obs_statistic, summaries )
    n_pilot = size( summaries, 2 )
    distances = zeros( n_pilot )
    for jj in 1:n_pilot
        summary_column = @view summaries[:, jj]
        if all( isfinite, summary_column )
            distances[jj] = _apply_distance( distance, summary_column, obs_statistic )
        else
            distances[jj] = Inf
        end
    end
    return distances
end

"""
    pilot_simulations( target, n_pilot; rng = Random.default_rng() ) -> (params, summaries, distances)

Run `n_pilot` prior-predictive simulations: draw the active parameters from `target.priors` (in the
order of `get_active_model_params( target.model )`), simulate and summarize as at MCMC time, and
compute each summary's distance to the observed summary.

Returns a `NamedTuple`:
- `params`: `npar × n_pilot` matrix of the parameter draws.
- `summaries`: `summary_length × n_pilot` matrix of the simulated summaries. A failed simulation
  (non-finite values) gives a `NaN` column.
- `distances`: length-`n_pilot` vector of distances, `Inf` for a failed simulation. Under ABC the
  resolved `target.options.inference_method.distance` is used; under other methods there is no
  distance and every entry is `NaN`.

Throws an `ArgumentError` if `target.priors` is `nothing` or any active parameter has no prior
(`nothing`): a flat prior cannot be sampled.

`TargetData` uses this to set ABC's `:scaled` scales (MAD of the summaries) and a `NaN` `epsilon`
(a quantile of the finite distances). It is also the basis of plain ABC rejection sampling.
"""
function pilot_simulations( target, n_pilot::Integer; rng = Random.default_rng() )
    params, summaries = _pilot_summaries( target, n_pilot, rng )
    distances = _method_pilot_distances( target.options.inference_method, target, summaries )
    return ( params = params, summaries = summaries, distances = distances )
end

_method_pilot_distances( ::AbstractInferenceMethod, target, summaries ) = fill( NaN, size( summaries, 2 ) )
_method_pilot_distances( abc::ABC, target, summaries ) = _pilot_distances( abc.distance, target.obs_mean, summaries )


#------------Distance and ε resolution (called by TargetData once the target exists)

# Constant components have zero scale and would get an infinite weight. They are kept, with the
# smallest positive scale: such a component (eg. an ECDF bin at 0 or 1 in the observed data) can
# still separate simulations from the observation, so dropping it (weight 0) would lose
# information, while the smallest positive scale bounds its weight by the most precise component's.
function _guard_zero_scales( scale )
    positive_scales = filter( value -> isfinite( value ) && value > 0, scale )
    if isempty( positive_scales )
        throw( ArgumentError( "Every summary component is constant in the summaries used for the :scaled ABC distance, so no scale can be set. Use more pilot simulations, a different summary, or a callable distance." ) )
    end
    smallest_scale = minimum( positive_scales )
    guarded_scale = map( value -> ( isfinite( value ) && value > 0 ) ? value : smallest_scale, scale )
    return guarded_scale
end

# Per-component normalized MAD of the successful (all-finite) pilot summaries.
function _mad_scales( pilot_summaries )
    finite_columns = [ all( isfinite, column ) for column in eachcol( pilot_summaries ) ]
    finite_summaries = pilot_summaries[:, finite_columns]
    if size( finite_summaries, 2 ) == 0
        throw( ArgumentError( "Every pilot simulation failed, so the :scaled ABC distance cannot be set." ) )
    end
    scale = [ mad( row; normalize=true ) for row in eachrow( finite_summaries ) ]
    return scale
end

_resolve_distance( distance, target, training_summaries, pilot_summaries ) = distance

function _resolve_distance( distance::Symbol, target, training_summaries, pilot_summaries )
    if distance === :scaled
        if target.options.resampling_type isa NoResampling
            raw_scale = _mad_scales( pilot_summaries )
        else
            raw_scale = vec( std( training_summaries, dims=2 ) )
        end
        scale = _guard_zero_scales( raw_scale )
        weights = 1 ./ scale.^2
        return WeightedEuclidean( weights )
    elseif distance === :mahalanobis
        # The target stores the regularized training covariance's factorization (see finalize_target_covariance)
        precision_matrix = inv( target.cov_factorization )
        symmetric_precision = Matrix( Symmetric( precision_matrix ) )
        return Mahalanobis( symmetric_precision )
    end
    throw( ArgumentError( "Unknown ABC distance :$(distance). Must be :scaled, :mahalanobis or a callable." ) )
end

# Replace the inference method in both target.options and target.data.options, keeping them in sync.
function _replace_inference_method( target, method )
    options = target.options
    options = @set options.inference_method = method
    target = @set target.options = options
    target = @set target.data.options = options
    return target
end

# GSL and BSL need nothing resolved.
finalize_inference_method( method::AbstractInferenceMethod, target, training_summaries, rng ) = target

# Resolve ABC's distance and, if NaN, ε. The pilot runs only when needed (ε is NaN, or :scaled
# needs MAD scales under NoResampling), and once: its summaries give the scales, then the distances
# are computed from the same summaries with the resolved distance, without re-simulating.
function finalize_inference_method( abc::ABC, target, training_summaries, rng )
    needs_mad_scales = abc.distance === :scaled && target.options.resampling_type isa NoResampling
    needs_pilot = isnan( abc.epsilon ) || needs_mad_scales

    pilot_summaries = nothing
    if needs_pilot
        if target.options.verbose
            println( "Running $(abc.n_pilot) ABC pilot simulations from the prior" )
        end
        _, pilot_summaries = _pilot_summaries( target, abc.n_pilot, rng )
    end

    distance = _resolve_distance( abc.distance, target, training_summaries, pilot_summaries )
    resolved_method = @set abc.distance = distance

    if isnan( abc.epsilon )
        pilot_distances = _pilot_distances( distance, target.obs_mean, pilot_summaries )
        finite_distances = filter( isfinite, pilot_distances )
        if isempty( finite_distances )
            throw( ArgumentError( "Every ABC pilot simulation failed, so epsilon cannot be set from the pilot quantile." ) )
        end
        epsilon = quantile( finite_distances, abc.epsilon_quantile )
        if !( epsilon > 0 )
            throw( ArgumentError( "The ABC pilot quantile gives epsilon = $(epsilon), but epsilon must be positive. Raise epsilon_quantile or set epsilon directly." ) )
        end
        resolved_method = @set resolved_method.epsilon = epsilon
    end

    return _replace_inference_method( target, resolved_method )
end


#------------ABC-MCMC hooks (called by AM, DRAM and mcmcrun; no-ops for GSL and BSL)

# The distances of the current MCMC state (n_sim values). GSL and BSL keep none.
_allocate_current_distances( ::AbstractInferenceMethod ) = Float64[]
_allocate_current_distances( abc::ABC ) = fill( NaN, abc.n_sim )

# Distances of every stored state: n_sim × chain_length under ABC, 0 × chain_length otherwise.
_allocate_distance_chain( ::AbstractInferenceMethod, chain_length ) = zeros( 0, chain_length )
_allocate_distance_chain( abc::ABC, chain_length ) = fill( NaN, abc.n_sim, chain_length )

# Make the distances of the latest calculate_loss call (in target.buffers.method_buffer) the current
# state's. Call it right after the evaluation of `params` that becomes the current state (an
# accepted proposal, or a re-evaluation of the current state). When `params` has zero prior
# density, calculate_loss returned before simulating and the buffer is stale: the current state
# then has no distances (NaN).
_commit_current_distances!( ::AbstractInferenceMethod, current_distances, target, params ) = nothing

function _commit_current_distances!( ::ABC, current_distances, target, params )
    logprior = evaluate_log_prior( params, target.priors, target.model.active_parameters )
    if isinf( logprior )
        fill!( current_distances, NaN )
    else
        copyto!( current_distances, target.buffers.method_buffer )
    end
    return nothing
end

# Store the current state's distances and ε in column/entry `ii` of the results.
_store_method_state!( ::AbstractInferenceMethod, results, current_distances, ii ) = nothing

function _store_method_state!( abc::ABC, results, current_distances, ii )
    results.distance_chain[:, ii] .= current_distances
    results.epsilon_chain[ii] = abc.epsilon
    return nothing
end

# The final ε of a run that stored `last_step` states. NaN for GSL and BSL.
_final_epsilon( ::AbstractInferenceMethod, epsilon_chain, last_step ) = NaN
_final_epsilon( abc::ABC, epsilon_chain, last_step ) = last_step >= 1 ? epsilon_chain[last_step] : abc.epsilon

_epsilon_adaptation( ::AbstractInferenceMethod ) = nothing
_epsilon_adaptation( abc::ABC ) = abc.adapt_epsilon

_check_epsilon_adaptation( ::Nothing, method ) = nothing

function _check_epsilon_adaptation( ::EpsilonAdaptation, method )
    if isnan( method.epsilon )
        throw( ArgumentError( "ABC epsilon is NaN, so it cannot be adapted. Build the target with TargetData, which resolves it." ) )
    end
    return nothing
end

# The current state's log-target at the ε of `abc`, from its stored distances (no new simulation).
# Mirrors the param-level calculate_loss, without the likelihood_noise_scale noise.
function _current_log_target( abc::ABC, target, state )
    logprior = evaluate_log_prior( state.current_params, target.priors, target.model.active_parameters )
    isinf( logprior ) && return -Inf

    log_likelihood = _log_kernel_average( abc.kernel, state.current_distances, abc.epsilon )
    isfinite( log_likelihood ) || return -Inf

    return log_likelihood + logprior
end

# One Robbins-Monro step of the Vihola-Franks tolerance adaptation at iteration `ii` (see
# EpsilonAdaptation). Returns the target with the new ε and the state with its log-target
# recomputed at the new ε. Without adaptation, or after n_adapt iterations, both are unchanged.
_adapt_epsilon( ::Nothing, target, state, log_ratio, ii ) = target, state

function _adapt_epsilon( adaptation::EpsilonAdaptation, target, state, log_ratio, ii )
    if ii > adaptation.n_adapt
        return target, state
    end
    abc = target.options.inference_method

    if isnan( log_ratio ) || log_ratio == -Inf
        acceptance_probability = 0.0
    else
        acceptance_probability = min( 1.0, exp( log_ratio ) )
    end

    step_size = ii^( -adaptation.step_exponent )
    log_epsilon = log( abc.epsilon ) + step_size * ( adaptation.target_acceptance - acceptance_probability )
    adapted_method = @set abc.epsilon = exp( log_epsilon )
    target = _replace_inference_method( target, adapted_method )

    ss_current = _current_log_target( adapted_method, target, state )
    state = @set state.ss_current = ss_current
    return target, state
end


#------------Post-correction to a smaller tolerance (Vihola and Franks 2020)
"""
    abc_post_correct( results, target, delta; burnin = 0 ) -> (weights, ess, samples)

Importance weights that correct an ABC-MCMC chain run at tolerance ε (`results.epsilon`) to a
smaller tolerance `delta ≤ ε` (Vihola and Franks 2020). A stored state with distances
`ρ_1, …, ρ_n` (`results.distance_chain`, `n = n_sim`) gets the weight

    mean_i K_δ(ρ_i) / mean_i K_ε(ρ_i)

with `K` the kernel of the target's ABC method. With the uniform kernel and `n_sim = 1` this is
`1{ρ ≤ δ}`: the post-corrected sample keeps the states with `ρ ≤ δ`. `delta = ε` gives equal
weights. A state with zero kernel average at ε (only possible before the first acceptance) gets
weight 0.

Samples `1:burnin` are dropped. With ε adaptation, `burnin` must cover the adaptation phase
(eg. `burnin = n_adapt`): every kept sample must have the final ε, else this throws.

Returns a `NamedTuple`:
- `weights`: the normalized weights (sum 1) of the kept samples.
- `ess`: the effective sample size `1 / sum( weights.^2 )`.
- `samples`: the kept samples, `results.chain[:, burnin+1:end]`.

Weighted posterior mean: `samples * weights`.

Throws an `ArgumentError` if the target is not ABC, if `delta` is not in `(0, ε]`, if `burnin`
leaves no sample, or if every weight is zero (no kept state is within `delta`).
"""
function abc_post_correct( results, target, delta::Real; burnin::Integer = 0 )
    method = target.options.inference_method
    return _post_correct( method, results, Float64( delta ), burnin )
end

function _post_correct( method::AbstractInferenceMethod, results, delta, burnin )
    throw( ArgumentError( "abc_post_correct needs a target with the ABC inference method, got $(method)." ) )
end

function _post_correct( abc::ABC, results, delta, burnin )
    epsilon = results.epsilon
    distance_chain = results.distance_chain
    n_samples = size( distance_chain, 2 )

    if isnan( epsilon )
        throw( ArgumentError( "results.epsilon is NaN: these results are not from an ABC run." ) )
    end
    if !( 0 < delta <= epsilon )
        throw( ArgumentError( "delta must be in (0, ε] with ε = results.epsilon = $(epsilon), got $(delta)." ) )
    end
    if !( 0 <= burnin < n_samples )
        throw( ArgumentError( "burnin must be in [0, $(n_samples)), got $(burnin)." ) )
    end

    kept_range = ( burnin + 1 ):n_samples
    kept_epsilons = @view results.epsilon_chain[kept_range]
    if any( !=( epsilon ), kept_epsilons )
        throw( ArgumentError( "The kept samples were run at different tolerances (ε adaptation). Raise burnin to cover the adaptation phase (eg. burnin = n_adapt)." ) )
    end

    log_weights = zeros( length( kept_range ) )
    for (jj, column) in enumerate( kept_range )
        distances = @view distance_chain[:, column]
        log_weights[jj] = _log_correction_weight( abc.kernel, distances, delta, epsilon )
    end

    log_total = logsumexp( log_weights )
    if log_total == -Inf
        throw( ArgumentError( "Every post-correction weight is zero: no kept state has a positive kernel value at delta = $(delta). Use a larger delta." ) )
    end

    weights = exp.( log_weights .- log_total )
    ess = 1 / sum( abs2, weights )
    samples = results.chain[:, kept_range]
    return ( weights = weights, ess = ess, samples = samples )
end

# log( mean K_δ(ρ) / mean K_ε(ρ) ). A state without distances (NaN, zero prior density) or with zero
# kernel average at ε has zero ABC posterior mass, so its weight is 0.
function _log_correction_weight( kernel, distances, delta, epsilon )
    if any( isnan, distances )
        return -Inf
    end
    log_denominator = _log_kernel_average( kernel, distances, epsilon )
    if log_denominator == -Inf
        return -Inf
    end
    log_numerator = _log_kernel_average( kernel, distances, delta )
    return log_numerator - log_denominator
end
