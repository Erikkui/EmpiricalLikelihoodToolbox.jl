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

