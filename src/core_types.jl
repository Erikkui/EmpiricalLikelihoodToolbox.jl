# Abstract types
abstract type AbstractSummaryStatistic end

abstract type AbstractECDFSummary <: AbstractSummaryStatistic end
abstract type AbstractChamferSummary <: AbstractSummaryStatistic end

abstract type DifferenceECDFSummary <: AbstractECDFSummary end
abstract type ChamferDifference <: AbstractChamferSummary end

abstract type AbstractSimulationModel end


# Resampling types
struct StandardResampling end

struct TimeseriesResampling end


# Container structs
#------------Main options struct
Base.@kwdef struct MethodsOptions{R, T}
    axis_uniform::Symbol = :xax
    bins_resamplings::Int = 40
    resampling_type::R = StandardResampling()
    training_resamplings::Int = 200
    mcmc_resamplings::Int = 200
    n_summaries::Int = 1
    timeseries_block_size::T = nothing
end

#------------Buffer container for non-allocating in-place computations
struct BufferContainer{S, I}
    summary_buffers::S
    training_buffer::Matrix{Float64}
    mcmc_buffer::Matrix{Float64}
    simulation_obs::Matrix{Float64}
    simulation_diffs::Vector{Matrix{Float64}}
    simulation_statistic::Vector{Float64}
    index_cache::I
end

#------------Container for passing data, differences and options to functions
Base.@kwdef struct DataContainer{M <: AbstractMatrix{Float64}, D <: AbstractVector{Matrix{Float64}}, O, P}
    observations::M
    differences::D = Vector{Matrix{Float64}}(undef, 0)
    difference_orders::O = [0]
    options::P = MethodsOptions()
end

#------------MCMC options struct
Base.@kwdef struct MCMCOptions{A, G, F}
    nsteps::Int = 1000
    update_interval::Int = 50
    mcmc_algorithm::A
    initial_params::G = nothing
    loss_function::F = LogLikelihood()
end

#------------MCMC target
struct TargetData{S, O, B, C}
    data::C
    summary_statistics::S
    options::O
    buffers::B
    obs_mean::Vector{Float64}
    inverse_cov::Matrix{Float64}
    summary_length::Int
end
