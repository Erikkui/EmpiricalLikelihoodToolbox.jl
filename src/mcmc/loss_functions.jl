# Loss functions: raw callables `loss( sim, obs, cov_factorization )` returning the unscaled,
# unstandardized value. Standardization and `inverse_scaling` are applied by `scale_loss`, which
# the inference methods call (see src/inference/).
@kwdef struct LogLikelihood
    scaling_parameter::Float64 = 1.0
    inverse_scaling::Float64 = 1.0/scaling_parameter
end

function LogLikelihood( scaling_parameter::Real )
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/Float64( scaling_parameter )
    return LogLikelihood( scaling, inverse_scaling )
end

function (loss::LogLikelihood)( x::AbstractVector, x_star::AbstractVector, cov_factorization )
    delta = x - x_star
    ss = -0.5*dot( delta, cov_factorization \ delta )
    return ss
end


@kwdef struct RobustChamfer
    scaling_parameter::Float64 = 1.0
    inverse_scaling::Float64 = 1/scaling_parameter
end

function RobustChamfer( scaling_parameter::Real )
    scaling = Float64( scaling_parameter )
    inverse_scaling = 1.0/scaling
    return RobustChamfer( scaling, inverse_scaling )
end

function (loss::RobustChamfer)( x::AbstractVector, x_star::AbstractVector, cov_factorization )
    ss = -sum( x )
    return ss
end


# Apply the target's standardization (only if requested in the options) and then the loss's
# `inverse_scaling` to a raw loss value.
function scale_loss( loss, raw_value, target )
    scaled_value = raw_value
    if target.options.standardize
        scaled_value = standardize( scaled_value, target.standardization_mean, target.standardization_sd )
    end
    scaled_value *= loss.inverse_scaling
    return scaled_value
end

# Compact, stable text for a loss, used in the inference methods' `show`: `LogLikelihood(2.0)`.
function _loss_string( loss::Union{LogLikelihood, RobustChamfer} )
    loss_name = nameof( typeof( loss ) )
    return "$(loss_name)($(loss.scaling_parameter))"
end
_loss_string( loss ) = string( loss )
