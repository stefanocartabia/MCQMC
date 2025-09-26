include("../../Data_Libraries/1.Libraries.jl")

## Definition of the Hierarchical Poisson–Gamma model structure
mutable struct H_Poisson_Gamma
    n::Int                  # number of observations
    Y::Vector{Int}          # response counts
    X::Vector{Float64}      # covariates (one per observation)
    alpha::Float64          # prior parameter for λ_j
    gamma::Float64          # shape parameter for β
    delta::Float64          # rate parameter for β
end

function H_Poisson_Gamma(Y::Vector{Int}, X::Vector{Float64};
                         alpha::Float64=1.802, gamma::Float64=0.1, delta::Float64=1.0)
    n = length(Y)
    return H_Poisson_Gamma(n, Y, X, alpha, gamma, delta)
end

## Parameter initialisation 
function init_par(model::H_Poisson_Gamma)
    lambda_0 = [rand(Gamma(model.alpha, 1 / (model.delta + model.X[j]))) for j in 1:model.n]
    beta_0 = rand(Gamma(model.gamma, 1 / model.delta))
    return vcat(lambda_0, beta_0)
end

## Proposal sampler 
## seq length is n+1: n lambdas + beta

function proposal_sample(model::H_Poisson_Gamma, X, seq)
    n, Y, X, alpha, gamma, delta = model.n, model.Y, model.X, model.alpha, model.gamma, model.delta
    X_t = Vector{Float64}(undef, n+1)

    # Step 1: update lambdas
    beta = X[end]
    for j in 1:n
        shape = alpha + Y[j]
        rate  = beta + X[j]
        lambda_j = quantile(Gamma(shape, 1/rate), seq[j])
        X_t[j] = lambda_j
    end

    # Step 2: update beta
    beta_shape = gamma + n*alpha
    beta_rate  = delta + sum(X_t[1:n])
    beta  = quantile(Gamma(beta_shape, 1/beta_rate), seq[end])
    X_t[end] = beta

    return X_t
end


