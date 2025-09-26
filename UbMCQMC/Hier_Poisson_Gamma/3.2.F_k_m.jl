##################################### Unbiased estimator of E[f(x)] #############################################

function UbFkm_HPG(m, seq, mdl::H_Poisson_Gamma, k, f)
    n = mdl.n
    x_t = Matrix{Float64}(undef, n+1, m+1)    # First chain X_t
    y_t = Matrix{Float64}(undef, n+1, m)      # Second chain Y_t
    x_t[:,1] = init_par(mdl)                  # First chain initialisation
    y_t[:,1] = init_par(mdl)                  # Second chain initialisation

    # yt is 1-period delayed
    x_t[:,2] = proposal_sample(mdl, x_t[:,1], seq[:,1])

    t = 2
    tau = Inf   # meeting time
    while t <= max(tau, m)
        # Source of randomness
        v = t <= m ? seq[:,t] : rand(n+1)

        # Before meeting
        if t < tau
            x_t[:,t+1], y_t[:,t] = coupling_sample(mdl, x_t[:,t], y_t[:,t-1], v)
            if x_t[:,t+1] == y_t[:,t]
                tau = t
            end
        # After meeting
        else
            newstate = proposal_sample(mdl, x_t[:,t], v)
            x_t[:,t+1] = newstate
            y_t[:,t]   = newstate
        end

        t += 1
    end

    m = size(x_t, 2)
    M = m - k
    f_X_t = Matrix{Float64}(undef, length(f(x_t[:,1])), M)

    for ell in 1:M
        tt = k - 1 + ell
        if tt >= tau
            # After meeting: only first chain matters
            f_X_t[:,ell] = f(x_t[:,tt+1])
        else
            # Before meeting: correction term
            f_X_t[:,ell] = sum(f(x_t[:,tt+1:tau])) - sum(f(y_t[:,tt+1:tau-1]))
        end
    end

    return mean(f_X_t; dims=2)
end