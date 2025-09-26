include("../Data_Libraries/1.Libraries.jl")

##################################### Bayesian Linear Regression  #############################################
######### Structure definition #######################################################################################

mutable struct BLR_Gibbs_WCUD
    p::Int                              # number of regressors/parameters (intercept included)
    d::Int                              # number of observations
    dataY::Vector{Float64}
    dataX::Matrix{Float64}
    dataXTX::Matrix{Float64}
    dataXTY::Vector{Float64}
    b0::Vector{Float64}
    B0::Matrix{Float64}
    B0_inv::Matrix{Float64}
    n0::Float64
    n1::Float64
    s0::Float64
end

function BLR_Gibbs_WCUD(p::Int, datay::Vector{Float64}, dataX::Matrix{Float64}, b0::Vector{Float64}, B0::Matrix{Float64}, n0::Float64, s0::Float64)
    d = length(datay)
    dataXTX = dataX' * dataX
    dataXTY = dataX' * datay
    B0_inv = inv(B0)
    n1 = n0 + d
    return BLR_Gibbs_WCUD(p, d, datay, dataX, dataXTX, dataXTY, b0, B0, B0_inv, n0, n1, s0)
end

######## Sampling Steps #################################################################################################

function init_par(model::BLR_Gibbs_WCUD)
    Beta_0 = rand(MvNormal(model.b0, model.B0))
    sigma_0 = 1 / rand(Gamma(model.n0 / 2, 2 / model.s0))
    return vcat(Beta_0, sigma_0)
end

function proposal_sample(model::BLR_Gibbs_WCUD, X, seq)

    # Betas
    # B(t) = (B0^-1 + σ^-2(t-1) * X'X)^-1 
    # b(t) = B(t) * (B0^-1 * b0 + σ^(-2(t-1)) * X' * y) 
    sigma_t_1 = X[end]
    B_t = Symmetric(inv(model.B0_inv + 1 /sigma_t_1 * model.dataXTX))          
    b_t = B_t*(model.B0_inv * model.b0 + (1 /sigma_t_1) * model.dataXTY)       
    beta = b_t .+ cholesky(B_t).L * quantile.(Normal(0, 1), seq[1:end-1])       

    # Sigma/Variance
    # n_t = n0 + n         -- shape is constant 
    # s_t = s0 + (y - X*β(t))' * (y - X*β(t))   -- scale changes
    RSS = (model.dataY-model.dataX*beta)'*(model.dataY-model.dataX*beta)
    s_t = (model.s0 + RSS)
    sigma = quantile(InverseGamma(model.n1/2, s_t/2), seq[end])

    return vcat(beta, sigma)

end

######## Gibbs sampler   ######################################################################################

function BLR_gibbs_sampler(model::BLR_Gibbs_WCUD, seq::Array{Float64,3}, burn_in::Int, R::Int, alpha_level::Float64=0.95)
    tot_par = model.p+1
    n_iter = size(seq)[2]
    x_t = Array{Float64}(undef, tot_par, n_iter, R)             # Betas + sigma
    
    for r in 1:R 
        x_t[:,1,r] = init_par(model)                            # Chain Initialisation 
        for t in 2:n_iter
            x_t[:,t,r] = proposal_sample(model, x_t[:,t-1,r], seq[:,t,r])
        end
    end 

    chains = x_t[:, burn_in+1:end, :]
    mean_x_t =  reshape(mean(chains; dims=3), tot_par, n_iter-burn_in)

    # Credibility Intervals 
    lower_q = (1 - alpha_level)/2
    upper_q = 1 - lower_q
    CIs = [(quantile(mean_x_t[i, :], lower_q),
            quantile(mean_x_t[i, :], upper_q)) for i in 1:tot_par]

    return  (
                chains = chains,                               # (p+1, n_iter - burn_in, R)
                mean_x_t = mean_x_t,                           # averaging over the chains
                mean_post_dist =  mean(mean_x_t; dims=2),      # averaging across iterations
                CIs = CIs                                      # Credibility Intervals
            )
end

