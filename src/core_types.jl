# Abstract types
abstract type AbstractSummaryStatistic end

abstract type AbstractInferenceMethod end

abstract type AbstractECDFSummary <: AbstractSummaryStatistic end
abstract type AbstractChamferSummary <: AbstractSummaryStatistic end
abstract type AbstractCumulativeSumSummary <: AbstractSummaryStatistic end

abstract type StandardECDFSummary <: AbstractECDFSummary end
abstract type TwoSetSummary <: AbstractECDFSummary end

abstract type CILSummary <: TwoSetSummary end
abstract type IDSummary <: TwoSetSummary end
abstract type ChamferECDFSummary <: TwoSetSummary end

abstract type AbstractSimulationModel end


# Inference methods
#------------Gaussian Subset Likelihood (Haario et al., 2015): the target mean/covariance are
# estimated once from the observed data (via resampling), and compared against a single simulated
# summary at each MCMC step.
struct GSL <: AbstractInferenceMethod end

#------------Bayesian Synthetic Likelihood (Wood, 2010): the observed summary is a fixed quantity,
# and the mean/covariance used in the likelihood are re-estimated at every MCMC step from `n_sim`
# fresh simulations at the current parameter proposal.
Base.@kwdef struct BSL <: AbstractInferenceMethod
    n_sim::Int = 10
end


# Container structs
#------------Main options struct
Base.@kwdef struct MethodsOptions{R, F, E, IM}
    axis_uniform::Symbol = :xax
    covariance_type::Symbol = :cov
    bins_resamplings::Int = 40
    resampling_type::R = RademacherSplit()
    training_resamplings::Int = 1000
    N_obs::Int
    effective_N_obs::Int = N_obs
    mcmc_resamplings::Int = training_resamplings
    n_summaries::Int = 1
    n_loss_evals::Int = 1
    standardize::Bool = false
    verbose::Bool = false
    ecdf_calculation_type::Symbol = :default    # :default, :kernel_smoothed
    ecdf_function::F = resolve_ecdf( ecdf_calculation_type )
    embedding_dim::E = 0
    embedding_type::Symbol = :delay    # :delay, :diff
    inference_method::IM = GSL()
end

#------------Buffer container for non-allocating in-place computations
struct BufferContainer{S, I, M, SD, ST, BB}
    summary_buffers::S
    training_buffer::M
    mcmc_buffer::M
    simulation_diffs::SD
    simulation_statistic::ST
    index_cache::I
    bsl_buffer::BB
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
    cov_factorization::IC
    summary_length::SL
    standardization_mean::T
    standardization_sd::T
end
