# In VSCode terminal:
#   0. Strt Julia REPL
#   1. ] activate .
#   2. Press backspace
#   3. using Revise
#   4. include("examples/main.jl")
#   5. run run_test_mcmc() in REPL: res = run_test_mcmc()

using EmpiricalLikelihoodToolbox
using Distributions
using CairoMakie


function run_test_mcmc()
    axis_unif = :yax
    nrep_training = 2000
    nrep_sampling = nrep_training
    chain_length = 10000

    Ndata = 1000
    dt_obs = 1.0

    model = Lorenz63Model( dt_obs = dt_obs )
    data = solve_model( model, Ndata*dt_obs )

    resampler = StandardResampling()
    lossfun = LogLikelihood( scaling_parameter = 1.0 )
    # sampler = AM( proposal_width = 0.01 )
    sampler = AM( proposal_width = 0.01, adaptation_interval = 50 )
    priors = ( Uniform(0.0, 30.0), Uniform(0.0, 50.0), Uniform(0.0, 10.0) )

    summary_statistics = JointSummaryStatistics(
        ChamferDistance(1)
        )

    methods_options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        mcmc_resamplings=nrep_sampling,
        n_loss_evals = 1,
        verbose = true
        )

    mcmc_options = MCMCOptions(
        nsteps = chain_length,
        mcmc_algorithm = sampler,
        update_interval = 50,
        loss_function = lossfun,
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options; priors=priors )

    results, state = mcmcrun( target, model, mcmc_options )

    true_params = EmpiricalLikelihoodToolbox.get_params( model )
    npara = size(results.chain, 1)
    fig = Figure(size=(1200, 200*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results.chain[i, :])
        hlines!(ax, [true_params[i]], color=:red, linestyle=:dash)
    end
    display(fig)

    return results, state
end
