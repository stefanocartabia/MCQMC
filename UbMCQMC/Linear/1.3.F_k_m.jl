include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/UbMCQMC/Linear/1.1.UnMCQMC_linear.jl")

##################################### Biased estimator of E[f(x)]  #############################################

function Fkm(n_iter, seq, mdl, k, f)
    p = mdl.p                                        # number of parameters
    xt = Matrix{Float64}(undef, p+1, n_iter+1)           # store chain states
    xt[:,1] = init_par(mdl)                          # initialisation

    for t in 1:n_iter
        xt[:, t+1] = proposal_sample(mdl, xt[:,t], seq[:,t])
    end

    vals = hcat([f(xt[:,t]) for t in (k+1):(n_iter)]...)  # matrix 
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

function UbFkm(m,seq,mdl,k,f)
    p = mdl.p                                    # Number of parameters (p-1 regressor + 1 incercept)
    x_t = Matrix{Float64}(undef,p+1,m+1)         # First chain X_t
    y_t = Matrix{Float64}(undef,p+1,m)           # Second chain Y_t
    x_t[:,1] = init_par(mdl)                     # First chain initialisation
    y_t[:,1] = init_par(mdl)                     # Second chain initialisation

    # yt is 1-period delayed
    x_t[:,2] = proposal_sample(mdl, x_t[:,1], seq[:,1])

    # Alg.4, Line 7  
    t=2; tau = Inf                               # Iterator
    while t <= max(tau, m)
        # Source of randomness; if the meeting time is greater than m; append IIDs to (W)CUD sequence 
        # 
        if t <= m
            v = seq[:,t]
        else
            v = rand(p)
        end 
        # # Alg.4, Line 13, before chains meet up
        if t < tau                                      
            x_t[:,t+1], y_t[:,t] = coupling_sample(mdl, x_t[:,t], y_t[:,t-1], v)
                if x_t[:,t+1] == y_t[:,t]
                    meeting_time = tau
                    tau = t;
                end
        # # Alg.4, Line 20, after chains meet up 
        else                                            
            newstate = proposal_sample(mdl, x_t[:,t], v)
            x_t[:, t+1] = newstate
            y_t[:, t]   = newstate
        end

        t += 1 
    end
 
    m = size(x_t)[2]
    M = m-k                                                           # Number chain states to consider
    f_X_t = Matrix{Float64}(undef, length(f(x_t[:,1])), M)            # As many rows as the number of values returned by f(x) -- betas + variance

    for ell = 1:M
        tt = k-1+ell
        if tt >= tau
        # The meeting time is overcome and the second term of the btw eq. 6 and eq.7 is zero 
            f_X_t[:,ell] = f(x_t[:,tt+1])
        else
            f_X_t[:,ell] = sum(f(x_t[:,tt+1:tau])) - sum(f(y_t[:,tt+1:tau-1])) 
        end
    end
        return  mean(f_X_t; dims=2)                                    # Mean value along the columns

end


println("Check: Imported F_k_m estimator")