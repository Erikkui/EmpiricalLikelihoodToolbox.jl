# In VSCode terminal:
#   0. Strt Julia REPL
#   1. ] activate .
#   2. Press backspace
#   3. using Revise
#   4. include("examples/main.jl")
#   5. run run_test_mcmc() in REPL: res = run_test_mcmc()

using EmpiricalLikelihoodToolbox
using CairoMakie


function run_test_mcmc()
    axis_unif = :yax
    nrep_training = 1000
    nrep_sampling = nrep_training
    chain_length = 10000

    Ndata = 1000
    dt_obs = 1.0

    model = Lorenz63Model( dt_obs = dt_obs )
    data = solve_model( model, Ndata*dt_obs )

    resampler = StandardResampling()
    lossfun = RobustChamfer( scaling_parameter = 1.0 )

    summary_statistics = JointSummaryStatistics(
        CIL( 10 ),
        CILDiff(10, 1, dt_obs ),
        )

    methods_options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        mcmc_resamplings=nrep_sampling )


    mcmc_options = MCMCOptions(
        nsteps = chain_length,
        mcmc_algorithm = AM( 0.01, 50 ),
        update_interval = 50,
        loss_function = lossfun,
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options )

    results, state = mcmcrun( target, model, mcmc_options )


    npara = size(results[:chain], 1)
    fig = Figure(size=(1200, 200*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results[:chain][i, :])
    end
    display(fig)

    return results, state
end
