# If it is a real distribution, calculate the logpdf
evaluate_single_prior( prior::Distribution, val::Real ) = logpdf( prior, val )

# If it is 'nothing' (uninformative flat prior), it contributes to 0.0
evaluate_single_prior( ::Nothing, val::Real) = 0.0

# No priors for any of the parameters specified -- flat prior everywhere.
evaluate_log_prior( params::AbstractVector, priors::Nothing, active_param_names ) = 0.0

# priors is matched to params by name (via active_param_names, the canonical order params follows),
# not by NamedTuple key order -- a name missing from priors falls back to a flat prior.
function evaluate_log_prior( params::AbstractVector, priors::NamedTuple, active_param_names )

    log_prior_val = 0.0
    for (ii, name) in enumerate( active_param_names )
        log_prior_val += evaluate_single_prior( get( priors, name, nothing ), params[ii] )
    end

    return log_prior_val
end
