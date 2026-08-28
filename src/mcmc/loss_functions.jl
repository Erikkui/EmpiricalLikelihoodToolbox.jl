struct LogLikelihood
    scaling_parameter::Float64
    inverse_scaling_parameter::Float64
end

function LogLikelihood( scaling_parameter::Real )
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/Float64( scaling_parameter )
    return LogLikelihood( scaling, inverse_scaling )
end

function LogLikelihood(; scaling_parameter::Real)
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/Float64( scaling_parameter )
    return LogLikelihood( scaling, inverse_scaling )
end

function (loss::LogLikelihood)( target::TargetData, sim_mean::AbstractVector )

    standardize_loss = target.options.standardize

    obs_mean = target.obs_mean
    inv_C = target.inverse_cov

    delta = sim_mean - obs_mean

    ss = -0.5*dot( delta, inv_C*delta )

    if standardize_loss
        standardize!( ss, target.standardization_mean, target.standardization_sd )
    end

    ss *= loss.inverse_scaling_parameter

    return ss
end

function (loss::LogLikelihood)( x::AbstractVector, x_star::AbstractVector, inv_cov::AbstractMatrix )
    delta = x - x_star
    ss = -0.5*dot( delta, inv_cov*delta )
    return ss
end




@kwdef struct RobustChamfer
    scaling_parameter::Float64 = 1.0
    inverse_scaling::Float64 = 1.0
end

function RobustChamfer( scaling_parameter::Real )
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/scaling
    return RobustChamfer( scaling, inverse_scaling )
end

function RobustChamfer(; scaling_parameter::Real)
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/Float64( scaling_parameter )
    return RobustChamfer( scaling, inverse_scaling )
end

function (loss::RobustChamfer)( target::TargetData, sim_mean::AbstractVector )

    standardize_loss = target.options.standardize

    ss = -sum( sim_mean )

    if standardize_loss
        standardize!( ss, target.standardization_mean, target.standardization_sd )
    end

    ss *= loss.inverse_scaling

    return ss
end

function (loss::RobustChamfer)( x::AbstractVector, x_star::AbstractVector, inv_cov::AbstractMatrix )
    # delta = x - x_star
    # ss = -0.5*dot( delta, inv_cov*delta )
    inverse_std = sqrt.( diag(inv_cov) )
    ss = -sum( ( x .* inverse_std )  )
    return ss
end
