# Abstract types
abstract type AbstractSummaryStatistic end

abstract type AbstractECDFSummary <: AbstractSummaryStatistic end
abstract type AbstractChamferSummary <: AbstractSummaryStatistic end

abstract type DifferenceECDFSummary <: AbstractECDFSummary end
abstract type ECDFMultiDimensionalSummary <: AbstractECDFSummary end
abstract type IDSummary <: AbstractECDFSummary end
abstract type ChamferDifference <: AbstractChamferSummary end

abstract type AbstractSimulationModel end


# Container structs
#------------Main options struct
Base.@kwdef struct MethodsOptions{R}
    axis_uniform::Symbol = :xax
    covariance_type::Symbol = :cov
    bins_resamplings::Int = 40
    resampling_type::R = StandardResampling()
    training_resamplings::Int = 1000
    N_obs::Int
    mcmc_resamplings::Int = training_resamplings
    n_summaries::Int = 1
    n_loss_evals::Int = 1
    use_ecdf_sampling::Bool = false
    standardize::Bool = false
    verbose::Bool = false
end

#------------Buffer container for non-allocating in-place computations
struct BufferContainer{S, I, M, SO, SD, ST}
    summary_buffers::S
    training_buffer::M
    mcmc_buffer::M
    simulation_obs::SO
    simulation_diffs::SD
    simulation_statistic::ST
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
    discard_noisy_updates::Bool = false
    reevaluate_current_loss::Bool = false
    mcmc_algorithm::A
    initial_params::G = nothing
    loss_function::F = LogLikelihood()
    likelihood_noise_scale::Float64 = NaN
end

#------------MCMC target
struct TargetData{C, S, P, O, B, T, OM, IC, SL}
    data::C
    summary_statistics::S
    priors::P
    options::O
    buffers::B
    obs_mean::OM
    inverse_cov::IC
    summary_length::SL
    standardization_mean::T
    standardization_sd::T
end
