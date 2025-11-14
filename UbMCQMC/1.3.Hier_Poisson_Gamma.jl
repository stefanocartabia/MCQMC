include("../Data_Libraries/1.Libraries.jl")
struct H_Poisson_Gamma
    n::Int                  # number obs
    Y::Vector{Int}          
    X::Vector{Float64}      
    alpha::Float64          # parametro del prior per λ_n
    gamma::Float64          # parametro shape per β
    delta::Float64          # parametro rate per β
end

function H_Poisson_Gamma(Y::Vector{Int}, X::Vector{Float64};
                         alpha::Float64=1.802, gamma::Float64=0.1, delta::Float64=1.0)
    n = length(Y)
    return H_Poisson_Gamma(n, Y, X, alpha, gamma, delta)
end

function HPG_Gibbs_sampler(model::H_Poisson_Gamma, seq::Array{Float64}, n_iter::Int, burn_in::Int, R::Int)
    n, Y, X, alpha, gamma, delta = model.n, model.Y, model.X, model.alpha, model.gamma, model.delta

    X_t = Array{Float64}(undef, n+1, n_iter, R)     # Lambdas + Beta

    # Loop over replications
    for r in 1:R
        # Initialization
        lambda = ones(n)
        beta = 1.0

        # Iterations
        for t in 1:n_iter
            # Step 1: Update λ_j
            for j in 1:n
                lambda[j] = quantile(Gamma(alpha + Y[j], 1/(beta + X[j])), seq[j, t, r])       # scale = 1/rate
            end
            # Step 2: update β
            beta = quantile(Gamma(gamma + n*alpha, 1/(delta + sum(lambda))), seq[end, t, r])

            # Store samples
            X_t[1:end-1, t, r] = lambda
            X_t[end, t, r] = beta
        end
    end

    # Discard burn-in
    Lambdas_t = X_t[1:end-1, burn_in+1:end, :]
    Beta_t = X_t[end, burn_in+1:end, :]

    # Posterior means across iterations and replications
    mean_lambdas_per_chain = dropdims(mean(Lambdas_t; dims=2), dims=2)   # (n, R)
    mean_beta_per_chain    = dropdims(mean(Beta_t; dims=2), dims=2)      # (R,)
    mean_lambdas_post_dist = mean(mean_lambdas_per_chain; dims=2)        # Posterior Mean Lambda;  (n,)
    mean_beta_post_dist    = mean(mean_beta_per_chain)                   # Posterior Mean Beta; scalar
    # Variances Computation                      
    var_lambdas = vec(sum((mean_lambdas_per_chain .- mean_lambdas_post_dist).^2; dims=2) ./ (R*(R-1)))
    var_beta    = sum((mean_beta_per_chain .- mean_beta_post_dist).^2) / (R*(R-1))

    return (
        lambda_chains = Lambdas_t,
        beta_chains = Beta_t,
        lambda_post_mean = mean_lambdas_post_dist,
        beta_post_mean = mean_beta_post_dist,
        var_lambdas = var_lambdas,
        var_beta = var_beta
    )
end


println("Status Hierarchical Poisson-Gamma: Correct Import")