@kwdef struct LogLikelihood
    scaling_parameter::Float64 = 1.0
end

function LogLikelihood(scaling::Real)
    scaling = 1.0/scaling
    return LogLikelihood( scaling_parameter = Float64(scaling) )
end

function (loss::LogLikelihood)( target::TargetData, sim_mean::AbstractVector )

    standardize = target.options.standardize

    obs_mean = target.obs_mean
    inv_C = target.inverse_cov

    delta = sim_mean - obs_mean

    ss = -0.5*dot( delta, inv_C*delta )

    if standardize
        standardize!( ss, target.standardization_mean, target.standardization_sd )
    end

    ss *= loss.scaling_parameter

    return ss
end

function (loss::LogLikelihood)( x::AbstractVector, x_star::AbstractVector, inv_cov::AbstractMatrix )
    delta = x - x_star
    ss = -0.5*dot( delta, inv_cov*delta )
    return ss
end



@kwdef struct RobustChamfer
    scaling_parameter::Float64 = 1.0
end

function RobustChamfer(scaling::Real)
    return RobustChamfer(scaling_parameter = Float64(scaling))
end

function (loss::RobustChamfer)( target::TargetData, sim_mean::AbstractVector )

    inverse_std = sqrt.( diag(target.inverse_cov) )

    ss = -sum( ( sim_mean .* inverse_std )  )

    if standardize
        standardize!( ss, target.standardization_mean, target.standardization_sd )
    end

    ss *= loss.scaling_parameter

    return ss
end
