include("../Data&Libraries/1.Libraries.jl")

## Definition of the UnMCQMC_linear structure and method constructor

mutable struct UnMCQMC_linear
    p::Int                              # number of regressors/parameters
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

function UnMCQMC_linear(p::Int, datay::Vector{Float64}, dataX::Matrix{Float64}, b0::Vector{Float64}, B0::Matrix{Float64}, n0::Float64, s0::Float64)
    d = length(datay)
    dataXTX = dataX' * dataX
    dataXTY = dataX' * datay
    B0_inv = inv(B0)
    n1 = n0 + d
    return UnMCQMC_linear(p, d, datay, dataX, dataXTX, dataXTY, b0, B0, B0_inv, n0, n1, s0)
end

## Parameter initialisation -- pag 14 (He, and Du 2024) --
function init_par(model::UnMCQMC_linear)
    Beta_0 = rand(MvNormal(model.b0, model.B0))
    sigma_0 = 1 / rand(Gamma(model.n0 / 2, 2 / model.s0))
    return vcat(Beta_0, sigma_0)
end

## Proposal sampler
# X : last chain state (a vector)
# seq: p-dim vector of (W)CUD
function proposal_sample(model::UnMCQMC_linear, X, seq)

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

## Maximum coupling sampling 
#  Preliminary function: Normal log-ratio of qi(xi)\pi(xi)
#  -½[ log|Σq| - log|Σp| + (x-μq)'Σq^{-1}(x-μq) - (x-μp)'Σp^{-1}(x-μp)]
function q_p_log_ratio_normal(x_t, p_mu, p_sigma, q_mu, q_sigma)
    x_p = x_t - p_mu
    x_q = x_t - q_mu
    p_sigma_inv = inv(Symmetric(p_sigma))
    q_sigma_inv = inv(Symmetric(q_sigma))

    return -0.5(logdet(q_sigma)-logdet(p_sigma)+ dot(x_q, q_sigma_inv * x_q) - dot(x_p, p_sigma_inv * x_p))
end

#  Preliminary function: InvGamma log_ratio of qi(xi)\pi(xi)
function q_p_log_ratio_inv_gamma(x_t::Float64, shape::Float64, scale_p::Float64, scale_q::Float64)

    return shape*(log(scale_q)-log(scale_p)) - (scale_q - scale_p)/x_t
end


# Sample proposal for Maximum coupling 
function coupling_sample(model::UnMCQMC_linear, X, Y, seq)
    p = model.p
    X_t = Vector{Float64}(undef, p+1)           # p-1 regressors + 1 intercept + variance 
    Y_t = Vector{Float64}(undef, p+1)           # p-1 regressors + 1 intercept + variance
    
    ################################### Betas ###########################################
    # Line 16, Alg 4 
    # Set pi(.) = P(x-i,.)
    sigma_t_1_p = X[end]
    Bt_p = Symmetric(inv(model.B0_inv + 1 /sigma_t_1_p * model.dataXTX))
    bt_p = Bt_p*(model.B0_inv * model.b0 + (1 /sigma_t_1_p) * model.dataXTY)
    # Set qi = P(y-i,.)
    sigma_t_1_q = Y[end]
    Bt_q = Symmetric(inv(model.B0_inv + 1 /sigma_t_1_q * model.dataXTX))
    bt_q = Bt_q*(model.B0_inv * model.b0 + (1 /sigma_t_1_q) * model.dataXTY)
    
    # Line 17, Alg 4
    beta_x = bt_p .+ cholesky(Bt_p).L * quantile.(Normal(0, 1), seq[1:end-1])
    if log(rand()) <= min(0,q_p_log_ratio_normal(beta_x, bt_p, Bt_p, bt_q, Bt_q))                   # Equivalent: min(1,q(.)/p(.))
          beta_y = beta_x
          X_t[1:end-1] = beta_x
          Y_t[1:end-1] = beta_y
    else 
        while true 
            beta_y = rand(MvNormal(bt_q, Bt_q))
            if rand() < -q_p_log_ratio_normal(beta_y, bt_p, Bt_p, bt_q, Bt_q)
                Y_t[1:end-1] = beta_y
                X_t[1:end-1] = beta_x
                break 
            end
        end
    end

    ################################### Sigma ###########################################
    # Line 16, Alg 4
    # Set pi(.) = P(x-i,.)
    s_t_p = (model.s0 + (model.dataY-model.dataX*beta_x)'*(model.dataY-model.dataX*beta_x))         # RSS = (model.dataY-model.dataX*beta_x)'*(model.dataY-model.dataX*beta_x)
    # Set qi = P(y-i,.)
    s_t_q = (model.s0 + (model.dataY-model.dataX*beta_y)'*(model.dataY-model.dataX*beta_y))         
    
    # Line 17, Alg 4 
    sigma_x = quantile(InverseGamma(model.n1/2, s_t_p/2), seq[end])
    if log(rand()) <= min(0, q_p_log_ratio_inv_gamma(sigma_x, model.n1, s_t_p, s_t_q))
          sigma_y = sigma_x
          X_t[end] = sigma_x
          Y_t[end] = sigma_y
    else 
        while true 
            sigma_y = rand(InverseGamma(model.n1/2, s_t_q/2))
            if rand() < -q_p_log_ratio_inv_gamma(sigma_y, model.n1, s_t_p, s_t_q)
                Y_t[end] = sigma_y
                X_t[end] = sigma_x
                break 
            end
        end
    end 


    return X_t, Y_t
end


println("Check: Imported functions")