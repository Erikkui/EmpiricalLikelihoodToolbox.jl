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

# Same, with the names as a type (TargetData.active_names): the loop is unrolled and every prior
# lookup is resolved at compile time. Summed left to right from 0.0, like the loop above.
function evaluate_log_prior( params::AbstractVector, priors::NamedTuple, ::Val{active_param_names} ) where {active_param_names}
    npar = length( active_param_names )
    log_prior_terms = ntuple( Val( npar ) ) do ii
        evaluate_single_prior( get( priors, active_param_names[ii], nothing ), params[ii] )
    end
    return foldl( +, log_prior_terms; init = 0.0 )
end
