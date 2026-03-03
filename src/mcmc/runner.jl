function allocate_results_buffer( npar, chain_length )
    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    prop_cov = Matrix{Float64}( I, npar, npar )
    results_buffer = (
        chain=chain_buffer,
        sschain=ss_buffer,
        prop_cov=prop_cov,
        acceptance=0.0, )
    return results_buffer

end


function mcmcrun( target::TargetData, model::AbstractSimulationModel, mcmc_options::MCMCOptions )

    # Setup
    MCMCRun = mcmc_options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    chain_length = mcmc_options.nsteps

    model_params = get_params( model )
    npar = length( model_params )

    if isnothing( mcmc_options.initial_params )
        current_params = model_params .+ 0.1 .* randn( npar )
    else
        current_params = mcmc_options.initial_params
    end

    proposal_cov = Matrix{Float64}( I, npar, npar )*proposal_width
    ss_current = calculate_loss( current_params, target, model, mcmc_options.loss_function )

    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    results_buffers = ( chain=chain_buffer, ss=ss_buffer )

    state = MCMCState( current_params, ss_current, proposal_cov, 0.0, 0 )

    try
        results, state = MCMCRun( target, model, state, mcmc_options, results_buffers )
    catch e
        println( "Error during MCMC run: ", e )
        results = ( chain=chain_buffer, ss=ss_buffer )
    end

    return results, state
end


function calculate_loss( params, target, model, loss_function )

    if any( params .< 0.0 )
        return -Inf
    end

    options = target.options
    R0_all = target.data
    summaries = target.summary_statistics
    diff_orders = target.data.difference_orders
    buffers = target.buffers
    dt_obs = model.dt_obs

    resampler = options.resampling_type
    n_summaries = options.n_summaries

    model = reconstruct( model, params )

    ndata = size( R0_all.observations, 2 )
    dt_obs = model.dt_obs

    Rsim = solve_model( model, ndata*dt_obs )::Matrix{Float64}
    copyto!( buffers.simulation_obs, Rsim )
    Rsim_diff = Vector{Matrix{Float64}}(undef, 0)
    if maximum(diff_orders) > 0
        Rsim_diff = calculate_diffs( Rsim, diff_orders, dt_obs )::Vector{Matrix{Float64}}
        copyto!.( buffers.simulation_diffs, Rsim_diff )
    end
    Rsim_container = DataContainer( observations=Rsim, differences=Rsim_diff, difference_orders=diff_orders, options=options )

    # Resample data and calculate summary statistics for current parameters
    resample_buffer = buffers.mcmc_buffer
    index_cache = buffers.index_cache
    mean_buffer = buffers.simulation_mean
    for ii in 1:n_summaries
        view_in = @view resample_buffer[ :, ii ]
        x_inds, _ = resampler( R0_all, options, index_cache )
        summaries( view_in, x_inds, R0_all, Rsim_container, buffers )
    end

    # mean_summary = mean( resample_buffer, dims=2 ) |> vec
    mean!( mean_buffer, resample_buffer)

    loss = loss_function( target, mean_buffer )

    # bins = target.summary_statistics.statistics[1].bins[1]
    # fig = Figure()
    # ax = Axis( fig[1, 1] )
    # for col in eachcol( resample_buffer )
    #     scatter!( ax, bins, col )
    # end
    # lines!( ax, bins, target.obs_mean, color=:red, linewidth=2, label="Mean Summary" )
    # lines!( ax, bins, mean_buffer, color=:blue, linewidth=2, label="Mean Summary" )
    # display(fig)
    # println("loss: ", loss)

    # sleep(0.1)
    # push!( debug[:means], deepcopy(mean_buffer) )
    # push!( debug[:resamples], deepcopy(Rsim) )

    return loss
end
