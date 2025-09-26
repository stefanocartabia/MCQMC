include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")

# Numerical simulation with Runge-Kutta method
function simulate_lorenz(f::Function, u0::AbstractVector{<:Real}, tspan::Tuple{<:Real,<:Real}, dt::Real)
    prob = ODEProblem(f, Float64.(u0), (float(tspan[1]), float(tspan[2])))
    sol = solve(prob, Tsit5(); saveat=float(dt))
    return Array(sol)  
end

# Gaussian log-likelihood (obs and sim are 3×T)
@inline function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix{<:Real}, var::Real)
    @assert size(obs) == size(sim)      "obs and sim must have same shape (3×T)"
    sse = 0.0
    @inbounds @simd for i in eachindex(sim)
        d = float(obs[i]) - float(sim[i])
        sse += d*d
    end
    n = length(sim)   # 3*T
    return -(n/2)*log(2π*var) - 0.5*sse/var
end


# function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix{<:Real}, cov_mat::AbstractMatrix{<:Real})
    
#     # Preliminary Check 
#     @assert size(obs) == size(sim)   "obs and sim must have same shape (3×T)"

#     cov_mat  = Matrix{Float64}(cov_mat)
#     d, K = size(sim)
#     sse = 0.0

#     F = cholesky(Symmetric(cov_mat))
#     inv_cov_mat = inv(Symmetric(cov_mat))

#     # L2 error norm
#     @inbounds for t in 1:K
#         err = obs[:, t] .- sim[:, t] 
#         sse += dot(err, inv_cov_mat*err)
#     end

#     # By cholensky decomposition: logdetΣ as 2*sum(log(diag(U)))  
#     return -(K*d/2)*log(2π) - K*sum(log, diag(F.U)) - 0.5*sse
# end



#################################################################################################################################


function mh_lorenz_pure(f::Function, 
                        x_init::AbstractVector{<:Real}, var_init::Real,
                        N_iter::Int,
                        obs_noisy::AbstractMatrix{<:Real},
                        tspan::Tuple{<:Real,<:Real}, dt::Real,
                        logprior_x0::Function, logprior_var::Function;
                        mix_rate_prop_x0::NTuple{3,<:Real}=(0.3,0.3,0.3),
                        mix_rate_prop_var::Real=0.1)

    # PRELIMINARY CHECKS                   
    @assert length(x_init) == 3 
    @assert var_init > 0
    @assert size(obs_noisy,1) == 3

    t0, t1 = float(tspan[1]), float(tspan[2])
    @assert size(obs_noisy,2) == Int(floor((t1 - t0)/float(dt))) + 1       "# noisy obs different from expected"
    
    # propσx = (float(mix_rate_prop_x0[1]), float(mix_rate_prop_x0[2]), float(mix_rate_prop_x0[3]))
    # τ_logσ2 = float(mix_rate_prop_var)

    # Chains definition 
    x_t  = Array{Float64}(undef, N_iter, 3)                             # Chain for initial conditions
    var_t = Array{Float64}(undef, N_iter)                               # Chain for variance 
    log_posterior  = Array{Float64}(undef, N_iter)

    # Current state
    x_cur = (float(x_init[1]), float(x_init[2]), float(x_init[3]))
    log_var_cur = log(float(var_init))
    var_cur = exp(log_var_cur)

    # Initialisation 
    sim_cur = simulate_lorenz(f, [x_cur...], tspan, float(dt))
    loglik_cur  = loglik_gaussian(obs_noisy, sim_cur, var_cur)
    logprior_x_cur = logprior_x0([x_cur[1], x_cur[2], x_cur[3]])
    logprior_var_cur = logprior_var(var_cur)
    post_cur = loglik_cur + logprior_x_cur + logprior_var_cur

    acc = 0      # accepted proposal counter 

    for i in 1:N_iter
        # propose jointly: x0 and log_var
        x_prop = x_cur .+ mix_rate_prop_x0 .* randn(3)
        log_var_prop = log_var_cur + mix_rate_prop_var*randn()
        var_prop = exp(log_var_prop)

        sim_prop = simulate_lorenz(f, [x_prop...], tspan, float(dt))
        loglik_prop  = loglik_gaussian(obs_noisy, sim_prop, var_prop)
        logprior_x_prop = logprior_x0([x_prop[1], x_prop[2], x_prop[3]])
        logpior_var_prop = logprior_var(var_prop)
        post_prop = loglik_prop + logprior_x_prop + logpior_var_prop

        if log(rand()) < (post_prop - post_cur)
            x_cur = x_prop
            log_var_cur = log_var_prop
            var_cur = var_prop
            sim_cur = sim_prop
            loglik_cur  = loglik_prop
            logprior_x_cur = logprior_x_prop
            logprior_var_cur = logpior_var_prop
            post_cur = post_prop
            acc += 1
        end

        x_t[i,1] = x_cur[1]; x_t[i,2] = x_cur[2]; x_t[i,3] = x_cur[3]; var_t[i]  = var_cur
        log_posterior[i] = post_cur
    end

    return (
        chain_theta = x_t,
        chain_var = var_t,
        acc_rate = acc / N_iter,
        logpost = log_posterior
    )
end

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


# autocovarianza via FFT
function acovf(x::AbstractVector{<:Real})
    n = length(x)
    m = mean(x)
    y = x .- m
    m2 = 2^ceil(Int, log2(2n))          # padding a potenza di 2
    fy = rfft(vcat(y, zeros(m2 - n)))
    ac = irfft(abs.(fy).^2, m2)[1:n]
    ac ./= (n:-1:1)                      # correzione per fine serie
    return ac
end

# ESS con initial positive sequence (Geyer)
function ess_ips(x::AbstractVector{<:Real})
    n = length(x)
    ac = acovf(x)
    γ0 = ac[1]
    ρ =  ac[2:end] ./ γ0            # autocorrelazioni ai lag ≥ 1
    s = 0.0
    k = 1
    while k < length(ρ)
        s_pair = ρ[k] + (k+1 <= length(ρ) ? ρ[k+1] : 0.0)
        if s_pair <= 0
            break
        end
        s += s_pair
        k += 2
    end
    τ = 1 + 2s                           # time-series variance factor
    return n / τ
end

# ESS per più catene: concateno dopo burn-in (assumendo stazionarietà/mescolamento simile)
function ess_ips_multichain(chains::Vector{<:AbstractVector{<:Real}})
    x = vcat(chains...)
    return ess_ips(x)
end

