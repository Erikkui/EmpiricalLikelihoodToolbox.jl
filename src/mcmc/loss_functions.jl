function log_likelihood( target::TargetData, sim_mean::AbstractVector )
    obs_mean = target.obs_mean
    inv_C = target.inverse_cov

    delta = sim_mean - obs_mean

    ss = -0.5*dot( delta, inv_C*delta )

    return ss
end
