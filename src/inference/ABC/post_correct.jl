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
