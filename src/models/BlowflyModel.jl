"""
    BlowflyModel{T}

A struct for the Nicholson blowfly model, which is a delay differential equation model of blowfly population dynamics.

# Fields
- `delta::Float64`: (default: 0.16).
- `P::Float64`: (default: 6.5).
- `N0::Float64`: (default: 400).
- `sigma2_p::Float64`: (default: 0.1).
- `tau::Float64`: (default: 14).
- `sigma2_d::Float64`: (default: 0.1).
- `x0::T`: The initial population. Default: 180.
- `dt_obs::Float64`: The observation time step (default: 1.0).

# Examples
model = BlowflyModel(dt_obs = 1.0, x0 = [1.0, 0.0, 0.0])
"""
Base.@kwdef struct BlowflyModel{N} <: AbstractSimulationModel
    delta::Float64 = 0.16
    P::Float64    = 6.5
    N0::Float64   = 400.0
    sigma2_p::Float64 = 0.1
    tau::Float64 = 14.0
    sigma2_d::Float64 = 0.1
    x0::Int = 180
    dt_obs::Float64 = 1.0
    dt_sol::Float64 = 1.0
    dim::Int = length(x0)
    burn_in::Int = 20
    mu::Float64 = 1.0
    all_parameters::Tuple{ Vararg{Symbol} } = (:delta, :P, :N0, :sigma2_p, :tau, :sigma2_d)
    active_parameters::Tuple{ Vararg{Symbol} } = (:delta, :P, :N0, :sigma2_p, :tau, :sigma2_d)
end

function BlowflyModel(theta::AbstractVector{<:Real}; kwargs...)
    N = length( theta )
    param_names = fieldnames( BlowflyModel )[ 1:N ]

    # 2. Create the NamedTuple
    named_params = NamedTuple{ param_names }( Tuple(theta) )

    return BlowflyModel( named_params...; kwargs... )
end
