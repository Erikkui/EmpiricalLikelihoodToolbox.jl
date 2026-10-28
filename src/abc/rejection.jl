#------------ABC rejection sampling (Pritchard et al. 1999). Draw parameters from the priors, simulate
# and summarize each draw, and keep the draws whose distance to the observed summary is within ε.
# A thin wrapper over pilot_simulations (src/inference/ABC.jl).

# ABC rejection, ABC-SMC and the pilot all need a resolved ABC distance.
function _check_abc_target( target, caller::AbstractString )
    if !( target.options.inference_method isa ABC )
        throw( ArgumentError( "$(caller) needs a target built with inference_method = ABC(...), got $(target.options.inference_method)." ) )
    end
    return target.options.inference_method
end

"""
    abc_rejection( target, n_draws; epsilon = nothing, n_keep = nothing, rng = Random.default_rng() )

ABC rejection sampling. Draws `n_draws` parameter vectors from `target.priors`, simulates and
summarizes each one as at MCMC time (`pilot_simulations`), and keeps the draws whose distance `ρ` to
the observed summary satisfies `ρ ≤ ε`. The distance is the resolved ABC distance of the target.

Give at most one of `epsilon` and `n_keep`:
- `epsilon`: keep every draw with `ρ ≤ epsilon`.
- `n_keep`: keep the `n_keep` draws with the smallest distances. ε is then the `n_keep`-th smallest
  distance, so exactly `n_keep` draws are returned.
- neither: use the target's ε (`target.options.inference_method.epsilon`).

A failed simulation has distance `Inf` and is kept only when ε is `Inf`. Every active parameter
needs a proper prior. The target must be built with `inference_method = ABC(...)`.

Returns a `NamedTuple`:
- `params`: `npar × n_accepted` matrix of the accepted draws, sorted by increasing distance. The rows
  follow the order of `get_active_model_params( target.model )`.
- `distances`: length-`n_accepted` vector of their distances (increasing).
- `epsilon`: the tolerance used.
- `acceptance_rate`: `n_accepted / n_draws`.

# Examples
```julia
result = abc_rejection( target, 10_000; n_keep = 500, rng = Xoshiro( 1 ) )
result = abc_rejection( target, 10_000; epsilon = 0.5 )
```

See also [`pilot_simulations`](@ref), [`abc_smc`](@ref), [`ABC`](@ref).
"""
function abc_rejection( target, n_draws::Integer; epsilon = nothing, n_keep = nothing, rng = Random.default_rng() )
    abc = _check_abc_target( target, "abc_rejection" )
    if !isnothing( epsilon ) && !isnothing( n_keep )
        throw( ArgumentError( "Give at most one of epsilon and n_keep, not both." ) )
    end
    if n_draws < 1
        throw( ArgumentError( "n_draws must be at least 1, got $(n_draws)." ) )
    end
    if !isnothing( n_keep ) && !( 1 <= n_keep <= n_draws )
        throw( ArgumentError( "n_keep must be in [1, n_draws = $(n_draws)], got $(n_keep)." ) )
    end

    pilot = pilot_simulations( target, n_draws; rng = rng )
    order = sortperm( pilot.distances )
    sorted_distances = pilot.distances[order]

    if !isnothing( n_keep )
        n_accepted = n_keep
        used_epsilon = sorted_distances[n_keep]
    else
        used_epsilon = isnothing( epsilon ) ? abc.epsilon : Float64( epsilon )
        n_accepted = count( distance_value -> distance_value <= used_epsilon, sorted_distances )
    end

    accepted_order = order[1:n_accepted]
    accepted_params = pilot.params[:, accepted_order]
    accepted_distances = sorted_distances[1:n_accepted]
    acceptance_rate = n_accepted / n_draws

    return ( params = accepted_params, distances = accepted_distances, epsilon = used_epsilon, acceptance_rate = acceptance_rate )
end
