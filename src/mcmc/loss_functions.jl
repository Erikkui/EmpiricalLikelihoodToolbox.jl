@kwdef struct LogLikelihood
    scaling_parameter::Float64 = 1.0
end

function LogLikelihood(scaling::Real)
    return LogLikelihood(scaling_parameter = Float64(scaling))
end

function (loss::LogLikelihood)( target::TargetData, sim_mean::AbstractVector )

    obs_mean = target.obs_mean
    inv_C = target.inverse_cov

    delta = sim_mean - obs_mean

    ss = -0.5*dot( delta, inv_C*delta )

    ss *= 1/loss.scaling_parameter

    return ss
end
