module EmpiricalLikelihoodToolbox

using StaticArrays
using Random
using DifferentialEquations
using Distances
using Distributions
using NearestNeighbors
using StatsBase
using Accessors
using LinearAlgebra
using Statistics
using ProgressBars
using SpecialFunctions


# Summaries
export StandardECDF, StandardECDFDiff, CIL, CILDiff, ChamferDistance, ChamferDistanceDiff, ChamferECDF, JointSummaryStatistics

# Models and solvers
export Lorenz63Model, OUModel, NormalModel, solve_model

# Resamplers and containers
export StandardResampling, TimeseriesResampling, MethodsOptions, TargetData

# MCMC functionalities
export MCMCOptions, mcmcrun, AM

include("core_types.jl")

include("summaries/StandardECDF.jl")
include("summaries/StandardECDFDiff.jl")
include("summaries/CIL.jl")
include("summaries/CILDiff.jl")
include("summaries/ChamferDistance.jl")
include("summaries/ChamferDistanceDiff.jl")
include("summaries/ChamferECDF.jl")
include("summaries/JointSummaryStatistics.jl")
include("summaries/empirical_cdf.jl")
include("summaries/chamfer_distance.jl")

include("mcmc/target.jl")
include("mcmc/samplers.jl")
include("mcmc/runner.jl")
include("mcmc/loss_functions.jl")

include("models/Lorenz63Model.jl")
include("models/OUModel.jl")
include("models/NormalModel.jl")
include("models/solvers.jl")

include("utils/bin_calculation.jl")
include("utils/resamplers.jl")
include("utils/utils.jl")

end
