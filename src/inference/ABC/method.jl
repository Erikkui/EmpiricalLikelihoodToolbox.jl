#------------Approximate Bayesian Computation (ABC-MCMC, Marjoram et al. 2003). The likelihood is
# replaced by a kernel K_ε(ρ) of the distance ρ between a simulated summary and the observed
# summary (target.obs_mean). With n_sim simulations per step, the average of the kernel values is
# an unbiased estimate of the ABC likelihood, so AM/DRAM run as a pseudo-marginal sampler. The
# sampler is exact (for the ABC posterior) only with the noisy-likelihood heuristic off
# (`update_interval <= 0`), `reevaluate_current_loss = false` and a fixed ε: with ε adaptation
# (`EpsilonAdaptation`, AM only), only the chain after the adaptation phase is a valid ABC-MCMC
# chain, at the final ε. The distances of every stored state are kept (`results.distance_chain`), so
# the chain can be post-corrected to a smaller tolerance δ ≤ ε (`abc_post_correct`, Vihola and
# Franks 2020).
#
# Files: method.jl (kernels, struct, method interface, distance and ε resolution), pilot.jl
# (prior-predictive pilot), mcmc.jl (run state, ε adaptation, results), post_correct.jl.

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


#------------Inference method interface (see interface.jl)

# Same reference summary as BSL (a single pass under NoResampling, else GSL's training loop), but
# the training covariance is kept: :scaled and :mahalanobis distances are resolved from it.
train_target( ::ABC, statistics, data_container, buffer_container, options ) =
    _train_reference_summary( statistics, data_container, buffer_container, options )

# ABC's loss never reads the target's factorization. When a training covariance exists (a resampler
# is used), its regularized factorization is stored anyway, since :mahalanobis is resolved from it;
# under NoResampling there is none, and an identity placeholder fixes the field's type and size.
finalize_target_covariance( ::ABC, cov_mat, summary_length ) = regularized_cholesky( cov_mat )
finalize_target_covariance( ::ABC, ::Nothing, summary_length ) = cholesky( Matrix( 1.0I, summary_length, summary_length ) )

# The distances live in each evaluated state's method data, so ABC needs no shared scratch buffer.
allocate_method_buffer( ::ABC, summary_length ) = Float64[]

# The n_sim distances of one evaluated state, in simulation order. NaN until simulated.
allocate_method_data( abc::ABC ) = fill( NaN, abc.n_sim )

# ABC has no loss function (see _check_loss_function in runner.jl).
_method_loss( ::ABC ) = nothing

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

# Simulate n_sim datasets, store their distances in `distances` (the evaluated state's method data) and
# return the log of the kernel average. A failed simulation has distance Inf, ie. kernel value 0: it
# stays in the average, which keeps the estimate unbiased. ε is read from the run state (it changes
# during ε adaptation), the kernel from `abc`.
function log_likelihood!( distances, abc::ABC, inference_state, target, model, mcmc_options, rng )
    kernel = abc.kernel
    epsilon = inference_state.epsilon

    for jj in 1:abc.n_sim
        distances[jj] = simulate_distance( target, model, rng )
    end

    return _log_kernel_average( kernel, distances, epsilon )
end

# All three kernels are unnormalized with maximum 1, so log K_ε(ρ) ≤ 0 for every distance.
max_log_likelihood( ::ABC, inference_state ) = 0.0

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

Shared by ABC-MCMC (`log_likelihood!`) and the pilot (`pilot_simulations`).
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
