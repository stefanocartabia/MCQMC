include("../../Data_Libraries/1.Libraries.jl")

## Definition of the UnMCQMC_probit structure and method constructor

mutable struct UnMCQMC_probit
    p::Int                                                   # number of regressors/parameters
    d::Int                                                   # number of observations
    dataY::Vector{Int}
    dataX::Matrix{Float64}
    XTX::Matrix{Float64}
    inv_XTXX::Matrix{Float64}                                # (X'X)^(-1) * X'
    inv_XTX::Matrix{Float64}                                 # (X'X)^(-1)
    cho_inv_XTX::Matrix{Float64}                             # Lower Cholesky of (X'X)^(-1)
    Beta_prior_mean::Union{Nothing, Vector{Float64}}
    Beta_prior_var::Union{Nothing, Matrix{Float64}}
end

function UnMCQMC_probit(dataY::Vector{Int}, dataX::Matrix{Float64};
                        Beta_prior_mean::Union{Nothing, Vector{Float64}}=nothing,
                        Beta_prior_var::Union{Nothing, Matrix{Float64}}=nothing)

    d, p = size(dataX)
    λ = 1e-4
    XTX = Symmetric(dataX' * dataX + λ*I)
    inv_XTX = Symmetric(inv(XTX))
    inv_XTXX   = inv_XTX * dataX'
    cho_inv_XTX = cholesky(inv_XTX).L

    return UnMCQMC_probit(p, d, dataY, dataX, XTX, inv_XTXX, inv_XTX,
                          cho_inv_XTX, Beta_prior_mean, Beta_prior_var)
end


## Parameter initialisation

function init_par_probit(model::UnMCQMC_probit)
    beta0 = model.inv_XTXX * model.dataY                # length p 
    z0 = rand.(Normal.(model.dataX * beta0, 1.0))       # length n (number of latent variables)
    return vcat(beta0, z0)
end


## Proposal sampler
# X : last chain state (a vector)
# seq: p-dim vector of (W)CUD

function proposal_sample_probit(model::UnMCQMC_probit, x_t::Vector{Float64}, seq::Vector{Float64})
    p = model.p                       # number of regressors/parameters
    d = model.d                       # number of observations
    X = model.dataX
    Y = model.dataY
    C = model.inv_XTXX                # (X'X)^(-1) * X'
    L = model.cho_inv_XTX             # Cholesky factor of (X'X)^(-1)

    Betas = x_t[1:p]
    z = x_t[p+1:end]

    # split WCUD sequence
    u_Beta = seq[1:p]                 # first p uniforms → for β
    u_zi   = seq[p+1:end]             # next d uniforms → for z

    #  Step 1: Update z and sample z | β, y
    zi_mu = X * Betas              
    
    f0 = cdf.(Normal(0,1), -zi_mu)
    for i in 1:d
        if Y[i] == 1
            # truncated normal above 0
            z[i] = zi_mu[i] + quantile(Normal(0,1), f0[i] + (1 - f0[i]) * u_zi[i])
        else
            # truncated normal below 0
            z[i] = zi_mu[i] + quantile(Normal(0,1), f0[i] * u_zi[i])
        end
    end

    # Step 2: Sample θ | z
    # β | z ~ N((X'X)^(-1) X' z , (X'X)^(-1))
    M = C * z                                           # conditional mean
    Betas = M .+ L * quantile.(Normal(0,1), u_Beta)

    return vcat(Betas, z)
end


## Maximum coupling sampling 

function coupling_sample_probit(model::UnMCQMC_probit, x::Vector{Float64}, y::Vector{Float64}, seq::Vector{Float64})
    p = model.p                             # number regressors
    d = model.d                             # number observations/latents
    X = model.dataX
    Y = model.dataY

    Isigma = model.XTX
    C      = model.inv_XTXX
    L      = model.cho_inv_XTX
    Sigma  = model.inv_XTX                  # Gram matrix, Conditional Covariance matrix β∣z in Albert & Chib (1993      

    x_t = Vector{Float64}(undef, p + d)
    y_t = Vector{Float64}(undef, p + d)

    # estrai z
    z_p = x[p+1:end]
    z_q = y[p+1:end]

    ####################################################### Betas ###############################################################
    # β | z ~ N((X'X)^(-1) X' z , (X'X)^(-1))
    # Conditional means (the covariance matrix is fixed to the Gram matrix)
    p_mu = C * z_p
    q_mu = C * z_q

    # Line 17, Alg 4
    # i) New state beta states, based on the full conditional
    beta_x = p_mu .+ L * quantile.(Normal(0,1), seq[1:p])

    # ii) u <= q(betas_x)/p(beta_x)      with u ~ IID
    if rand() <= exp(0.5 * (p_mu - q_mu)' * Isigma * (2*beta_x - p_mu - q_mu))
        beta_y = beta_x
    else
        while true
            byy = rand(MvNormal(q_mu, Symmetric(model.inv_XTX)))
            if rand() > exp(0.5 * (p_mu - q_mu)' * Isigma * (2*byy - p_mu - q_mu))
                beta_y = byy
                break
            end
        end
    end

    # Save betas
    x_t[1:p] = beta_x
    y_t[1:p] = beta_y

    ################################################### Z: Latents ##################################################################
    # Update z and sample z | θ, y
    u_zi = seq[p+1:end]                                                     # WCUD vector for Latents
    case_1 = (Y .== 1); case_0 = .!case_1;                                  # index for Y[i] == 1 and for Y[i] == 0

    # Update means full conditionals
    z_mu_x = X * beta_x                                                     # Updated mean truncated normal for first chain x_t
    z_mu_y = X * beta_y                                                     # Updated mean truncated normal for second chain y_t

    z_x = zeros(d); z_y = zeros(d); qdpx = zeros(d)
    
    # Φ(-μ_x) and Φ(-μ_y)
    f0_p = cdf.(Normal(0,1), -z_mu_x); f0_q = cdf.(Normal(0,1), -z_mu_y)
    # Updatez z for the first chain x_t based on the truncated normal with mean z_mu_x
    z_x[case_1] = z_mu_x[case_1] + quantile.(Normal(), f0_p[case_1] + (1 .- f0_p[case_1]) .* u_zi[case_1])    # case Y[i] == 1
    z_x[case_0] = z_mu_x[case_0] + quantile.(Normal(), f0_p[case_0] .* u_zi[case_0])                          # case Y[i] == 0

    # Coupling probability 
    muqmip = z_mu_y - z_mu_x; muqplp = z_mu_y + z_mu_x
    qdpx[case_1] = (1 .- f0_p[case_1]) ./ (1 .- f0_q[case_1]) .* exp.(0.5 .* muqmip[case_1] .* (2 .* z_x[case_1] .- muqplp[case_1]))   # case Y[i] == 1
    qdpx[case_0] = f0_p[case_0] ./ f0_q[case_0] .* exp.(0.5 .* muqmip[case_0] .* (2 .* z_x[case_0] .- muqplp[case_0]))                 # case Y[i] == 0

    for i in 1:d
        if rand() <= qdpx[i]
            # direct coupling
            z_y[i] = z_x[i]
        else
            # need to resample until accepted
            while true
                u = rand()
                if Y[i] == 1
                    z_y[i] = z_mu_y[i] + quantile(Normal(), f0_q[i] + (1 - f0_q[i])*u)
                    pdqyy = (1 - f0_q[i])/(1 - f0_p[i]) * exp(0.5 * (-muqmip[i]) * (2*z_y[i] - muqplp[i]))
                else
                    z_y[i] = z_mu_y[i] + quantile(Normal(), f0_q[i]*u)
                    pdqyy = f0_q[i]/f0_p[i] * exp(0.5 * (-muqmip[i]) * (2*z_y[i] - muqplp[i]))
                end

                if rand() > pdqyy
                    break
                end
            end
        end
    end

    x_t[p+1:end] .= z_x
    y_t[p+1:end] .= z_y

    return x_t, y_t
end





println("Status Probit: UnMCQMC_probit imported correctly")
