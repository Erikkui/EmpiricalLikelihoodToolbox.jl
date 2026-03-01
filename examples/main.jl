# In VSCode terminal:
#   0. Strt Julia REPL
#   1. ] activate .
#   2. Press backspace
#   3. using Revise
#   4. using EmpiricalLikelihoodToolbox
#   5. include("examples/main.jl")
#   6. run run_test_mcmc() in REPL: res = run_test_mcmc()

using EmpiricalLikelihoodToolbox
using CairoMakie


function run_test_mcmc()
    axis_unif = :yax
    nrep_training = 200
    nrep_sampling = nrep_training
    chain_length = 500

    Ndata = 200
    dt = 1.0

    model = Lorenz63Model( dt = dt )
    data = solve_model( model, Ndata*dt )

    resampler = StandardResampling()
    lossfun = LogLikelihood()

    summary_statistics = JointSummaryStatistics(
        ChamferECDF( 10, 1 ),
        )

    methods_options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        mcmc_resamplings=nrep_sampling )


    mcmc_options = MCMCOptions(
        nsteps = chain_length,
        mcmc_algorithm = AM( 0.01, 50 ),
        loss_function = lossfun,
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options )

    results, state = mcmcrun( target, model, mcmc_options )


    npara = size(results[:chain], 1)
    fig = Figure(size=(1200, 300*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results[:chain][i, :])
    end
    display(fig)

    return results, state
end
