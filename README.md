# EmpiricalLikelihoodToolbox
Toolbox for performing Bayesian inference using empirical likelihoods. The package features several summary statistics, which can be almost freely combined; currently, inference is based on so-called "Gaussian Subset Likelihood" (GSL) (Haario et al., 2015), but support Bayesian Synthetic Likelihood (BSL) (Wood, 2010) approach is planned. See also Kuitunen (2026) for definitions for majority of the summary statistics.

## Usage 
### Installation
Enter Julia package manager (`]`), and copy the following: `add https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl.git`

In your script, import the package: `using EmpiricalLikelihoodToolbox`

### Code example
```julia
using EmpiricalLikelihoodToolbox

function main()
    axis_unif = :yax                          # Which axis to use as uniform reference in bin creation 
    nrep_training = 5000                      # Number of training samples used in target mean and covariance creation
    n_summaries = 1                           # Number of summary vectors to generate for mean calculation in MCMC phase
    chain_length = 50000                      # MCMC chain length

    Ndata = 1000                              # Number of synthetic observations
    dt_obs = 1.0                              # Time step between observations

    model = Lorenz63Model( dt_obs = dt_obs )                # Lorenz 63 model
    data = solve_model( model, Ndata*dt_obs )               # Solver for Lorenz model uses DifferentialEquations.jl

    resampler = StandardResampling()                      # 50-50 resampling
    objective_fun = LogLikelihood()                       # Standard log-likelihood function as an objective function

    # Prior distributions, uninformative priors as nothing. Supports distributions from Distributions.jl.
    # Priors are defined as named tuples, where the keys are the parameter names
    param_names, param_values = get_active_model_params( model )    # Get "active" parameters, ie. those that are being inferred
    prior_distributions = tuple(                                            
      Uniform(0.0, 30.0), 
      Uniform(0.0, 50.0), 
      nothing 
      )
    priors = NamedTuple{ param_names }( prior_distributions )

    npar = length( param_values )
    initial_params = param_values .+ 0.01 .* randn( npar )    # Initial parameters as true parameter values with small added perturbations

    summary_statistics = JointSummaryStatistics(                      # Wrapping summaries into single joint summary
        CIL( 10 ),
        CILDiff( 10, 1, dt_obs ),
        IDDiff( 10, 1, 1, dt_obs ),
        )

    methods_options = MethodsOptions(                        # Collecting defined options into single struct
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        )

    mcmc_options = MCMCOptions(                               # Defining MCMC options
        nsteps = chain_length,
        mcmc_algorithm = AM( 0.01, 50 ),      # Adaptive metropolis with initial proposal width 0.01 and adaptation interval of 50 samples
        update_interval = 50,                 # Update interval for noisy MCMC heuristic; if set larger than chain length, essentially disables heuristic
        loss_function = objective_fun,
        initial_params = initial_params
        )

    # Create target data struct "target". Also ouputs training summaries for intermediary plotting, checks etc.
    target, training_summaries = TargetData( data, summary_statistics, methods_options )    

    # MCMC
    results, state = mcmcrun( target, model, mcmc_options )      

end
```

### Summary Statistics

Currently, nine summary statistics are available:
1. Standard ECDF
   - Initialization: `StandardECDF(nbin)`
   - Calculates an empirical cumulative distribution function from data
   - Defined by number of bins at which the eCDF is evaluated. Number of bins (`nbin`) are required as an argument.
   - CURRENTLY DOES NOT SUPPORT MULTI-DIMENSIONAL DATA!
2. Standard ECDF using numerical derivatives
   - Initialization: `StandardECDFDiff(nbin, diff_order, dt_obs)`
   - Same as above, but calculates the eCDF from the numerical derivative of the data
   - In addition to `nbin`, requires the order of differences (`diff_order`) and time step (`dt_obs`) between observations as arguments
3. Correlation Integral Likelihood (CIL)
   - Initialization: `CIL(nbin)`
4. Correlation Integral Likelihood using numerical derivatives 
   - Initialization: `CILDiff(nbin, diff_order, dt_obs)`
5. ECDF from Intrinsic Dimension nearest-neighbor ratios (ID)
   - Initialization: `ID(nbin, neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple ID ratio
6. ECDF from Intrinsic Dimension nearest-neighbor ratios using numerical derivatives
   - Initialization: `IDDiff(nbin, neighbors, diff_order, dt_obs)`
7. Chamfer Distance
   - Initialization: `ChamferDistance(neighbors)`
   - `neighbors` as and integer, or vector of integers for multiple Chamfer distances
8. Chamfer Distance using numerical derivatives
    - Initialization: `ChamferDistanceDiff(neighbors, diff_order, dt_obs)`
9. ECDF from Chamfer Distances 
    - Initialization: `ChamferECDF(nbin, neighbors)`
    - Support multiple neighbors simlar to basic Chamfer Distance

Summary statistics must be wrapped into a `JointSummaryStatistics()` struct, eg. `JointSummaryStatistics(CIL(10), CILDiff(10, 1, 1.0))`, as seen in the code example above.

### Resamplers
Currently, the package includes two resamplers: `StandardResampling`, which performs random 50-50 division for a data set; and `TimeseriesResampling`, which samples a random contiguous partition from the data. `TimeseriesResampling` requires user to define a `timeseries_block_size` as an input for the resampler type, eg. `TimeseriesResampling( timeseries_block_size = 70 )`. `TimeSeriesResampling` has not been tested as extensively as `StandardResampling`, and may contain bugs. 

### Additional settings

#### Inferring a subset of parameters

User can choose to infer only a subset of parameters, in which case the the remaining parameters are left constant during MCMC. This is done by providing a tuple of parameters for the model constructor:

```julia
    active_parameters = ( :sigma, :rho )
    model = Lorenz63Model( dt_obs = dt_obs, active_parameters = active_parameters )
```

#### Settings for summary generation

The following keyword settings are input to MethodsOptions struct:  
* `n_summaries::Int`: Number of sumaries to generate for each proposal parameter. When `n_summaries` > 1, their mean is calculated. Default: `1`. 
* `n_loss_evals::Int`: Number of loss evaluations to calculate for a proposal parameter. Each loss evaluation involves calculating `n_summaries` summaries, after which loss is calculated. Mean of the the evaluated losses is then used in MCMCM acceptance. Default: `1`.    
* `standardize::Bool`: Whether to perform z-score standardization for the summaries. Default: `false`.

## References
Heikki Haario, Leonid Kalachev, Janne Hakkarainen; Generalized correlation integral vectors: A distance concept for chaotic dynamical systems. Chaos 1 June 2015; 25 (6): 063102. https://doi.org/10.1063/1.4921939

Wood, S. Statistical inference for noisy nonlinear ecological dynamic systems. Nature 466, 1102–1104 (2010). https://doi.org/10.1038/nature09319

Kuitunen, E. Empirical likelihoods for intractable likelihood models. LUT Master's Thesis, LUTPub. (2026). https://lutpub.lut.fi/handle/10024/171299 


[![Build Status](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Erikkui/EmpiricalLikelihoodToolbox.jl/actions/workflows/CI.yml?query=branch%3Amain)
