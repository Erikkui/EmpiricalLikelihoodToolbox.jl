# If it is a real distribution, calculate the logpdf
evaluate_single_prior( prior::Distribution, val::Real ) = logpdf( prior, val )

# If it is 'nothing' (uninformative flat prior), it contributes to 0.0
evaluate_single_prior( ::Nothing, val::Real) = 0.0

function evaluate_log_prior( params::AbstractVector, priors )
    log_prior_val = 0.0
    for (ii, param) in enumerate( keys(priors) )
        log_prior_val += evaluate_single_prior( priors[param], params[ii] )
    end

    # ntuple forces the unrolling at compile time
    # log_prior = sum( ntuple( Val(N) ) do ii
    #     @inbounds evaluate_single_prior( priors[ii], params[ii] )
    # end)

    return log_prior_val
end
