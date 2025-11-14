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



#####################################################################################################################
###########################################           RWMC             ##############################################
#####################################################################################################################


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
########################################          MALA MCMC          ################################################
#####################################################################################################################

# =====================================================
# Finite-difference gradients (unificati)
# =====================================================

# R → R
grad_cdiff(f::Function, v::Real, args...; eps::Real=1e-5) =
    (f(v + eps, args...) - f(v - eps, args...)) / (2eps)

# R^d → R
function grad_cdiff(f::Function, x::AbstractVector{<:Real}, args...; eps::Real=1e-4)
    xloc = Float64.(x)
    g = similar(xloc, Float64)
    @inbounds for j in eachindex(xloc)
        xj = xloc[j]
        xloc[j] = xj + eps; fp = f(xloc, args...)
        xloc[j] = xj - eps; fm = f(xloc, args...)
        g[j] = (fp - fm) / (2eps)
        xloc[j] = xj
    end
    return g
end

# =====================================================
# Log-posterior (unica definizione)
# =====================================================

function logposterior_lorenz(f::Function,
                             x0::AbstractVector{<:Real}, var::Real,
                             obs_noisy::AbstractMatrix{<:Real},
                             tspan::Tuple{<:Real,<:Real}, dt::Real,
                             logprior_x0::Function, logprior_var::Function)
    # Simulazione del sistema
    sim = simulate_lorenz(f, x0, tspan, dt)
    ℓlik = loglik_gaussian(obs_noisy, sim, var)

    n = length(sim)
    sse = -2var * (ℓlik + (n/2)*log(2π*var))

    ℓprior_x = logprior_x0(x0)
    ℓprior_v = logprior_var(var)
    return (ℓlik + ℓprior_x + ℓprior_v, ℓlik, sse)
end

# =====================================================
# Gradiente wrt log(var)
# =====================================================

function grad_logpost_logvar(logvar::Real, sse::Real, n::Int, logprior_var::Function)
    var = exp(logvar)
    dℓlik_dvar    = -(n/(2var)) + 0.5*sse/var^2
    dℓlik_dlogvar = var * dℓlik_dvar
    dℓprior_dvar    = grad_cdiff(logprior_var, var)
    dℓprior_dlogvar = var * dℓprior_dvar
    return dℓlik_dlogvar + dℓprior_dlogvar
end

# =====================================================
# MALA Sampler (versione semplificata)
# =====================================================

function mala_lorenz_pure(f::Function,
                          x_init::AbstractVector{<:Real}, var_init::Real,
                          N_iter::Int,
                          obs_noisy::AbstractMatrix{<:Real},
                          tspan::Tuple{<:Real,<:Real}, dt::Real,
                          logprior_x0::Function, logprior_var::Function;
                          ε::Real = 0.05,
                          fd_eps_x0::Real = 1e-4)

    @assert length(x_init) == 3
    @assert var_init > 0
    @assert size(obs_noisy,1) == 3

    chain_x0  = Array{Float64}(undef, N_iter, 3)
    chain_var = Array{Float64}(undef, N_iter)
    chain_logpost = Array{Float64}(undef, N_iter)

    # Stato iniziale
    x_cur = Float64.(x_init)
    logvar_cur = log(float(var_init))
    var_cur = exp(logvar_cur)

    logpost_cur, loglik_cur, sse_cur = logposterior_lorenz(f, x_cur, var_cur, obs_noisy, tspan, dt, logprior_x0, logprior_var)

    n = length(obs_noisy)  # = 3*T
    acc = 0
    tmp = similar(x_cur)

    for it in 1:N_iter
        # ∇ rispetto a x0 con differenze finite
        g_x = grad_cdiff((x, f, obs, tspan, dt, var, lp_x, lp_v) ->
            logposterior_lorenz(f, x, var, obs, tspan, dt, lp_x, lp_v)[1],
            x_cur, f, obs_noisy, tspan, dt, var_cur, logprior_x0, logprior_var;
            eps = fd_eps_x0)

        # ∇ rispetto a log(var)
        g_l = grad_logpost_logvar(logvar_cur, sse_cur, n, logprior_var)

        # Proposta
        μ_x = @. x_cur + 0.5*ε^2*g_x
        μ_l = logvar_cur + 0.5*ε^2*g_l
        ηx = randn(3); ηl = randn()
        x_prop = @. μ_x + ε*ηx
        logvar_prop = μ_l + ε*ηl
        var_prop = exp(logvar_prop)

        # Target proposto
        logpost_prop, loglik_prop, sse_prop = logposterior_lorenz(
            f, x_prop, var_prop, obs_noisy, tspan, dt, logprior_x0, logprior_var)

        # Gradienti al proposto
        g_x_prop = grad_cdiff((x, f, obs, tspan, dt, var, lp_x, lp_v) ->
            logposterior_lorenz(f, x, var, obs, tspan, dt, lp_x, lp_v)[1],
            x_prop, f, obs_noisy, tspan, dt, var_prop, logprior_x0, logprior_var;
            eps = fd_eps_x0)

        g_l_prop = grad_logpost_logvar(logvar_prop, sse_prop, n, logprior_var)

        # Correzione di Metropolis–Hastings
        @. tmp = x_cur - (x_prop + 0.5*ε^2*g_x_prop)
        dx_back = sum(abs2, tmp)
        @. tmp = x_prop - (x_cur + 0.5*ε^2*g_x)
        dx_fwd  = sum(abs2, tmp)
        dl_back = (logvar_cur)  - (logvar_prop + 0.5*ε^2*g_l_prop)
        dl_fwd  = (logvar_prop) - (logvar_cur  + 0.5*ε^2*g_l)

        log_q_ratio = -0.5/ε^2 * (dx_back + dl_back^2 - dx_fwd - dl_fwd^2)
        logα = (logpost_prop - logpost_cur) + log_q_ratio

        if log(rand()) < logα
            x_cur = x_prop
            logvar_cur = logvar_prop
            var_cur = var_prop
            logpost_cur = logpost_prop
            loglik_cur = loglik_prop
            sse_cur = sse_prop
            acc += 1
        end

        chain_x0[it, :] .= x_cur
        chain_var[it]  = var_cur
        chain_logpost[it] = logpost_cur
    end

    return (;
        chain_theta = chain_x0,
        chain_var   = chain_var,
        acc_rate    = acc / N_iter,
        logpost     = chain_logpost
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

