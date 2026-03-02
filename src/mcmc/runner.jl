function mcmcrun( target::TargetData, model::AbstractSimulationModel, options::MCMCOptions )

    # Setup
    MCMCRun = options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    model_params = get_params( model )
    npar = length( model_params )

    if isnothing( options.initial_params )
        current_params = model_params .+ 0.1 .* randn( npar )
    else
        current_params = options.initial_params
    end

    proposal_cov = Matrix{Float64}( I, npar, npar )*proposal_width
    ss_current = calculate_loss( current_params, target, model, options.loss_function )

    state = MCMCState( current_params, ss_current, proposal_cov, 0.0, 0 )

    options = @set options.state = state

    results, state = MCMCRun( target, model, options )

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
    index_cache = buffers.indices_buffer
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
