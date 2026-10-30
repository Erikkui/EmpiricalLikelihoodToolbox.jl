# Allocations of one MCMC step (evaluate_log_target!). The bounds are the counts measured after the
# allocation fixes plus a small margin, so a regression (a type instability, a boxed variable, a
# per-call scratch array) fails here. What remains per step is mainly the simulation output, the
# Xoshiro seeding and the loss. RickerModel is used because its solver allocates only its output;
# the generic solver (eg. NormalModel) allocates per time point, which would hide the rest.

using EmpiricalLikelihoodToolbox: EvaluatedState, allocate_inference_state, evaluate_log_target!

function step_allocation_count( target, mcmc_options )
    params = copy( mcmc_options.initial_params )
    method = target.options.inference_method
    evaluated_state = EvaluatedState( method, length( params ) )
    inference_state = allocate_inference_state( method )
    evaluate_step() = evaluate_log_target!( evaluated_state, target, target.model, params, mcmc_options,
                                            inference_state; rng_seed = UInt64( 7 ) )
    evaluate_step()
    return @allocations evaluate_step()
end

@testset "allocations per MCMC step" begin
    data = wiggly( 100 )
    resampler = ContiguousBlockSplit( timeseries_block_size = 24 )
    model = RickerModel()
    mcmc_options = MCMCOptions( nsteps = 10, mcmc_algorithm = AM(), initial_params = [ 3.8, -1.2, 2.3 ] )

    @testset "GSL, StandardECDF" begin
        target, _ = make_target( data, StandardECDF( 10 ); resampling_type = resampler, model = model )
        # 17 measured on Julia 1.13, was 55 before the allocation fixes
        @test step_allocation_count( target, mcmc_options ) <= 25
    end

    @testset "GSL, ChamferECDF" begin
        target, _ = make_target( data, ChamferECDF( 5, 1 ); resampling_type = resampler, model = model,
                                 training_resamplings = 5 )
        # One small allocation per k-d tree rebuild (dists_for_ecdf = 1000): 1018 measured on
        # Julia 1.13, was ~451k before the allocation fixes
        @test step_allocation_count( target, mcmc_options ) <= 1100
    end
end
