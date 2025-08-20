include("../Data&Libraries/1.Libraries.jl")

struct Probit_Gibbs_WCUD
    p::Int                              # number of regressors/parameters (intercept included)
    d::Int                              # number of observations
    dataX::Matrix{Float64}
    dataY::Vector{Int}
    Beta_prior_mean::Union{Nothing, Vector{Float64}}
    Beta_prior_var::Union{Nothing, Matrix{Float64}}
end


function Probit_Gibbs_WCUD(dataX::Matrix{Float64}, dataY::Vector{Int}, 
                           Beta_prior_mean::Union{Nothing, Vector{Float64}} = nothing,
                           Beta_prior_var::Union{Nothing, Matrix{Float64}} = nothing)
    # If the conjucate prior is not passed, it is considered a flat prior 
    d, p = size(dataX)

    return Probit_Gibbs_WCUD(p, d, dataX, dataY, Beta_prior_mean, Beta_prior_var)
end


######## Gibbs sampler ######################################################################################

function Probit_Gibbs_sampler(model::Probit_Gibbs_WCUD, seq::Matrix{Float64}, burn_in::Int, n_iter::Int, R::Int, alpha_level::Float64=0.95)
                
    X = model.dataX
    Y = model.dataY 
    x_t = Array{Float64}(undef, model.p, n_iter, R)
    N, p = size(X) 

    for r in 1:R
        # Parameters initialisation
        Beta_0 =  (X' * X) \ (X' * Y) + rand(MvNormal(zeros(p), 10^2 * I))
        z = rand.(Normal.(X * Beta_0, 1.0))
        Betas = copy(Beta_0)

        # Flat Prior case
        if model.Beta_prior_var === nothing || model.Beta_prior_mean === nothing

            for t in 1:n_iter
                # WCUD sequence 
                u_Beta = seq[1:p, t]     # first p uniforms → for θ
                u_zi = seq[p+1:end, t]    # next N uniforms → for z

                #  Step 1: Update z and sample z | θ, y
                zi_mu = X * Betas
                for i in 1:N
                #     if Y[i] == 1
                #         f0 = cdf(Normal(0,1), -zi_mu[i])
                #         z[i] = zi_mu[i] + quantile(Normal(0,1), f0 + (1 - f0) * u_zi[i])
                #     else
                #         f0 = cdf(Normal(), -zi_mu[i])
                #         z[i] = zi_mu[i] + quantile(Normal(0,1), f0 * u_zi[i])
                #     end
                
                    ε = 1e-12
                    if Y[i] == 1
                        f0 = cdf(Normal(0,1), -zi_mu[i])
                        u_val = clamp(f0 + (1 - f0) * u_zi[i], ε, 1 - ε)
                        z[i] = zi_mu[i] + quantile(Normal(0,1), u_val)
                    else
                        f0 = cdf(Normal(0,1), -zi_mu[i])
                        u_val = clamp(f0 * u_zi[i], ε, 1 - ε)
                        z[i] = zi_mu[i] + quantile(Normal(0,1), u_val)
                    end
                end    
                # Step 2: Sample θ | z
                V = inv(X' * X)
                M = V*(X' * z)
                Betas = M .+ (cholesky(Symmetric(V)).L) * quantile.(Normal(0,1), u_Beta)

                # Store
                x_t[:, t, r] = Betas
                
            end

        # Conjucate Prior case
        else
            # Beta_prior_mean = model.Beta_prior_mean
            # Beta_prior_var = model.Beta_prior_var
            # z = rand.(Normal.(X * Beta_prior_mean, 1.0))
            # Inv_Beta_prior_var = inv(Beta_prior_var)

            # for t in 1:n_iter
            #     u_beta = seq[1:p, t, r]
            #     u_zi    = seq[p+1:end, t, r]

            #     # Step 1: sample z
            #     zi_mu = X * Betas
            #     for i in 1:N
            #         if Y[i] == 1
            #             f0 = cdf(Normal(0,1), -zi_mu[i])
            #             z[i] = zi_mu[i] + quantile(Normal(), f0 + (1 - f0) * u_zi[i])
            #         else
            #             f0 = cdf(Normal(0,1), -zi_mu[i])
            #             z[i] = zi_mu[i] + quantile(Normal(), f0 * u_zi[i])
            #         end
            #     end

            #     # Step 2: sample Betas with prior
            #     V = inv(X' * X + Inv_Beta_prior_var)
            #     M = V * (X' * z + Inv_Beta_prior_var * Beta_prior_mean)
            #     L = cholesky(Symmetric(V)).L
            #     Betas = M .+ L * quantile.(Normal(0,1), u_beta)

            #     chains[:, t, r] = Betas
        end

    end

    chains = x_t[:, burn_in+1:end, :]
    mean_x_t =  reshape(mean(chains; dims=3), p, n_iter-burn_in)      # Mean across R
    mean_Beta = reshape(mean(chains; dims=2), p, R)                   # Mean across iterations
    mean_post_dist =  mean(mean_x_t; dims=2)                          # Posterior Mean Beta 
    Var_Beta = vec(mean((mean_Beta .- mean_post_dist).^2; dims=2)) 

    # Credibility Intervals 
    lower_q = (1 - alpha_level)/2
    upper_q = 1 - lower_q
    CIs = [(quantile(mean_x_t[i, :], lower_q),
            quantile(mean_x_t[i, :], upper_q)) for i in 1:p]

    return  (
                x_t = x_t,
                chains = chains,                               # (p+1, n_iter - burn_in, R)
                mean_x_t = mean_x_t,                           # Inter-replicates-mean Beta_t
                mean_post_dist =  mean_post_dist,              # averaging across iterations
                Var_Beta = Var_Beta,                           # p-vector for Beta variances
                CIs = CIs                                      # Credibility Intervals
            )
end

######## Gibbs sampler ######################################################################################
# In this case we consider a WCUD array, meaning that chain replicates use different sequences

function Probit_Gibbs_sampler_2(model::Probit_Gibbs_WCUD, seq::Array{Float64}, burn_in::Int, n_iter::Int, R::Int, alpha_level::Float64=0.95)
                
    X = model.dataX
    Y = model.dataY 
    x_t = Array{Float64}(undef, model.p, n_iter, R)
    N, p = size(X) 

    for r in 1:R
        # Parameters initialisation
        Beta_0 =  (X' * X) \ (X' * Y) 
        z = rand.(Normal.(X * Beta_0, 1.0))
        Betas = copy(Beta_0)

        # Flat Prior case
        if model.Beta_prior_var === nothing || model.Beta_prior_mean === nothing

            for t in 1:n_iter
                # WCUD sequence 
                u_Beta = seq[1:p, t, r]     # first p uniforms → for θ
                u_zi = seq[p+1:end, t, r]    # next N uniforms → for z

                #  Step 1: Update z and sample z | θ, y
                zi_mu = X * Betas
                for i in 1:N
                    if Y[i] == 1
                        f0 = cdf(Normal(0,1), -zi_mu[i])
                        z[i] = zi_mu[i] + quantile(Normal(0,1), f0 + (1 - f0) * u_zi[i])
                    else
                        f0 = cdf(Normal(), -zi_mu[i])
                        z[i] = zi_mu[i] + quantile(Normal(0,1), f0 * u_zi[i])
                    end
                
                end    
                # Step 2: Sample θ | z
                V = inv(X' * X)
                M = V*(X' * z)
                Betas = M .+ (cholesky(Symmetric(V)).L) * quantile.(Normal(0,1), u_Beta)

                # Store
                x_t[:, t, r] = Betas
                
            end

        # Conjucate Prior case
        else
            # Beta_prior_mean = model.Beta_prior_mean
            # Beta_prior_var = model.Beta_prior_var
            # z = rand.(Normal.(X * Beta_prior_mean, 1.0))
            # Inv_Beta_prior_var = inv(Beta_prior_var)

            # for t in 1:n_iter
            #     u_beta = seq[1:p, t, r]
            #     u_zi    = seq[p+1:end, t, r]

            #     # Step 1: sample z
            #     zi_mu = X * Betas
            #     for i in 1:N
            #         if Y[i] == 1
            #             f0 = cdf(Normal(0,1), -zi_mu[i])
            #             z[i] = zi_mu[i] + quantile(Normal(), f0 + (1 - f0) * u_zi[i])
            #         else
            #             f0 = cdf(Normal(0,1), -zi_mu[i])
            #             z[i] = zi_mu[i] + quantile(Normal(), f0 * u_zi[i])
            #         end
            #     end

            #     # Step 2: sample Betas with prior
            #     V = inv(X' * X + Inv_Beta_prior_var)
            #     M = V * (X' * z + Inv_Beta_prior_var * Beta_prior_mean)
            #     L = cholesky(Symmetric(V)).L
            #     Betas = M .+ L * quantile.(Normal(0,1), u_beta)

            #     chains[:, t, r] = Betas
        end

    end

    chains = x_t[:, burn_in+1:end, :]
    mean_x_t =  reshape(mean(chains; dims=3), p, n_iter-burn_in)      # Mean across R
    mean_Beta = reshape(mean(chains; dims=2), p, R)                   # Mean across iterations
    mean_post_dist =  mean(mean_x_t; dims=2)                          # Posterior Mean Beta 
    Var_Beta = vec(mean((mean_Beta .- mean_post_dist).^2; dims=2)) 

    # Credibility Intervals 
    lower_q = (1 - alpha_level)/2
    upper_q = 1 - lower_q
    CIs = [(quantile(mean_x_t[i, :], lower_q),
            quantile(mean_x_t[i, :], upper_q)) for i in 1:p]

    return  (
                x_t = x_t,
                chains = chains,                               # (p+1, n_iter - burn_in, R)
                mean_x_t = mean_x_t,                           # Inter-replicates-mean Beta_t
                mean_post_dist =  mean_post_dist,              # averaging across iterations
                Var_Beta = Var_Beta,                           # p-vector for Beta variances
                CIs = CIs                                      # Credibility Intervals
            )
end


println("Status: Correct Import")


    