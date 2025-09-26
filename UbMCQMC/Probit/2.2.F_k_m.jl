include("2.1.UnMCMQMC_Probit.jl")

##################################### Biased estimator of E[f(x)]  #############################################

function Fkm_probit(mdl::UnMCQMC_probit, seq::Matrix{Float64}, n_iter, k, f::Function)
    p = mdl.p                                          # number of parameters (interpect included)
    n = mdl.d                                         
    xt = Matrix{Float64}(undef, p+n, n_iter+1)         # store chain states
    xt[:,1] = init_par_probit(mdl)                     # initialisation

    for t in 1:n_iter
        xt[:, t+1] = proposal_sample_probit(mdl, xt[:,t], seq[:,t])
    end

    betas = xt[1:p,:]
    vals = hcat([f(betas[:,t]) for t in (k+1):(n_iter)]...)  # matrix 
    return mean(vals, dims=2)                       
end


##################################### Unbiased estimator of E[f(x)]  #############################################

### Time_averages estimator_F_k_m, eq. 7 (He, and Du 2024)
# m: length of the simulation 
# d: 
# seq: (W)CUD sequence
# mdl: model structure
# k: burn-in period
# f: function to be estimated (i.e the mean)

function UbFkm_probit(mdl::UnMCQMC_probit, m::Int, seq::Matrix{Float64}, k::Int, f::Function)
    p = mdl.p                                           # Number of parameters (p-1 regressor + 1 incercept)
    n = mdl.d
    x_t = Matrix{Float64}(undef, p+n, m+1)         # First chain X_t
    y_t = Matrix{Float64}(undef, p+n, m+1)         # Second chain Y_t
    x_t[:,1] = init_par_probit(mdl)                     # First chain initialisation
    y_t[:,1] = init_par_probit(mdl)                     # Second chain initialisation

    # yt is 1-period delayed
    x_t[:,2] = proposal_sample_probit(mdl, x_t[:,1], seq[:,1])

    t=2; tau = Inf                               # Iterator
    while t <= max(tau, m)
        # Source of randomness; if the meeting time is greater than the m-dim (W)CUD sequence then append IIDs
        if t <= m
            v = seq[:,t]
        else
            v = rand(p+n)
        end 
        # Line 13 Alg 4, before chains meet up
        if t < tau                                      
            x_t[:,t+1], y_t[:,t] = coupling_sample_probit(mdl, x_t[:,t], y_t[:,t-1], v)
                if x_t[:,t+1] == y_t[:,t]
                    tau = t;
                end
        # Line 20 Alg 4, after chains meet up 
        else                                            
            newstate = proposal_sample_probit(mdl, x_t[:,t], v)
            x_t[:, t+1] = newstate
            y_t[:, t]   = newstate
        end

        t += 1 
    end
 
    m = size(x_t)[2]
    M = m-k                                                           # Number chain states to consider
    f_X_t = Matrix{Float64}(undef, length(f(x_t[1:p,1])), M)            # As many rows as the number of values returned by f(x) -- betas + variance

    for ell = 1:M
        tt = k-1+ell
        if tt >= tau
        # The meeting time is overcome and the second term of the btw eq. 6 and eq.7 is zero 
            f_X_t[:,ell] = f(x_t[1:p,tt+1])
        else
            f_X_t[:,ell] = sum(f(x_t[1:p,tt+1:tau])) - sum(f(y_t[1:p,tt+1:tau-1])) 
        end
    end
        return  mean(f_X_t; dims=2)                                    # Mean value along the columns

end



println("Check Probit: Imported F_k_m estimator")


