###############################################################################################################################################
################################################    MP-IS-MCQMC-MALA for Lotka-Volterra  ######################################################
###############################################################################################################################################

include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/1.Libraries_setup.jl")

# ODEs simulation
function simulate_system(f!::Function, u0, par, tspan::Tuple, dt)
    prob = ODEProblem(f!, u0, tspan, par)
    sol = solve(prob, Tsit5() ;saveat=dt, reltol=1e-6, abstol=1e-8, maxiters=10000, save_everystep=false)     
    return Array(sol)
end

# Gaussian log-likelihood with known noise covariance 
function loglik_gaussian(obs::AbstractMatrix{<:Real}, sim::AbstractMatrix{<:Real}, cov_mat::AbstractMatrix{<:Real})

    d, K = size(sim)
    S = Symmetric(Matrix(cov_mat))      
    F = cholesky(S; check = true)        # Σ = U' * U

    sse = 0.0
    @inbounds @views for t in 1:K
        err = obs[:, t] .- sim[:, t]     
        y = F.U \ err                    #  U * y = e_t  
        sse += dot(y, y)                 # e_t' Σ^{-1} e_t = y'y
    end

    return -0.5 * (K*d * log(2π) +  2 * K * sum(log, diag(F.U)) + sse)
end

# Log Posterior on logparameter space 
function build_log_posterior(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function)

    function logpost(theta)
        par = exp.(theta)
        sim = simulate_system(f!, u0, par, tspan, dt)
        return loglik_gaussian(obs, sim, cov_mat) + logprior_par(par) + sum(theta)
    end
    return logpost
end

# Gradient (finite difference) Log-posterior
function grad_fd!(grad::AbstractVector{Float64}, log_post::Function, x::AbstractVector{<:Real}; h::Float64=1e-3)
    @inbounds for i in 1:length(x)
        xp = collect(x); xp[i] += h
        xm = collect(x); xm[i] -= h
        grad[i] = (log_post(tuple(xp...)) - log_post(tuple(xm...))) / (2h)
    end
    return grad
end

# Tensor Metric: Fisher Information in the log-parameter space
function Tensor_Metric(f!, u0, theta::AbstractVector{<:Real}, tspan, dt, cov_mat::AbstractMatrix{<:Real}; λ::Float64 = 1e-6, h::Float64 = 1e-4)

    sim = simulate_system(f!, u0, Tuple(exp.(theta)), tspan, dt)  
    d, K = size(sim)
    D = length(theta)
    chol_cov_mat = cholesky(Symmetric(Matrix(cov_mat)))
    sim_p = Array{Float64}(undef, d, K, D)    
    sim_m = Array{Float64}(undef, d, K, D)   

    for j in 1:D
        theta_p = collect(theta); theta_p[j] += h
        sim_p[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_p)), tspan, dt)
        theta_m = collect(theta); theta_m[j] -= h
        sim_m[:, :, j] = simulate_system(f!, u0, Tuple(exp.(theta_m)), tspan, dt)

    end

    G = zeros(D, D)
    Jt = zeros(d, D)                     
    tmp = similar(sim[:, 1])             
    for t in 1:K
        for j in 1:D
            tmp .= sim_p[:, t, j] .- sim_m[:, t, j]
            Jt[:, j] .= tmp ./ (2*h)
        end
        # G += Jt' * (invΣ * Jt)
        W  = chol_cov_mat \ Jt            # W = Σ^{-1} Jt
        G .+= transpose(Jt) * W
    end

    G = Symmetric(G + λ * I)
    Minv = inv(G)
    L = cholesky(Symmetric(Minv)).L

    return (FisherInfo = G, InvFisherInfo = Minv, L = L)
end

function MP_IS_MALA_LV_2(f!::Function, u0, obs, cov_mat, tspan, dt, logprior_par::Function,
                       init_par::AbstractVector{<:Float64}; seq::AbstractMatrix{<:Float64},
                       N_prop::Integer, N_iter::Integer, step_size::Float64=0.12, h_der::Float64=1e-3)

    D = length(init_par)
    @assert D == 4 "This implementation expects 4 log-parameters."
    @assert size(seq,2) ≥ D+1 "seq needs ≥ D Normal-quantile columns + 1 resampling column."

    # Definition Log-posterior distribution
    f_logpost = build_log_posterior(f!, u0, obs, cov_mat, tspan, dt, logprior_par)

    # State and storage
    xI = init_par                            # Initial point in log-parameter space                          
    I_new = 1

    chain = Array{Float64}(undef, N_iter, D)
    accept_proxy = 0.0

    z = zeros(D); grad = zeros(D); yi = zeros(D)

    proposals = Array{Float64}(undef, N_prop + 1, D, N_iter)
    log_post_i = zeros(N_prop + 1)                              # current log-posterior 
    mu_yi = zeros(D)                                # mean normal Langevin dynamics
    logK_yi_z = zeros(N_prop + 1)
    logK_z_yi = zeros(N_prop + 1)
    w = Array{Float64}(undef, N_iter, N_prop + 1)
    logw = Vector{Float64}(undef, N_prop + 1)

    # Early stopping criterion
    stall_cnt = 0 
    l_effective = 0
    # max_stall  = max(5, cld(N_iter, 20))
    max_stall = cld(N_iter, 10)
    tol_move   = 1e-5

    metric = Tensor_Metric(f!, u0, init_par, tspan, dt, cov_mat)

    # MALA constants 
    alpha = (step_size^2) / 2
    G_inv = metric.InvFisherInfo                    # Fisher Information G^(-1)
    G = metric.FisherInfo                           # Inverse Fisher Information G                 
    L = metric.L
    CovScaling = 0.5
   
    
    row = 1
    for l in 1:N_iter
        wcud = seq[row:row + N_prop, :]
        row += (N_prop + 1)

        # K(xi,xj) = K(xi,z)K(z,xj)
        # Sample N+1 PROPOSALS: current state and N new ones
        proposals[1, :, l] = xI
        # Sample bridge variable z with MALA kernel K(z|xI)
        # Drift sm-MALA with ∇logπ(xI​)
        mu_x = xI + alpha * (G_inv*grad_fd!(grad, f_logpost, xI; h = h_der))
        z = mu_x + (step_size*CovScaling) * (L*quantile.(Normal(), wcud[1, 1:D]))
        # Sample N proposals with MALA kernel K(yi|z)
        # Drift sm-MALA with ∇logπ(z​)
        mu_z = z + alpha * (G_inv*grad_fd!(grad, f_logpost, z; h = h_der))
        
        @inbounds for j in 2:(N_prop + 1) 
            proposals[j, :, l] = mu_z + (step_size*CovScaling) * (L*quantile.(Normal(), wcud[j, 1:D]))
        end

        # Stationary distribution 𝑝(𝐼 = 𝑖 ∣ 𝑦_(1:𝑁+1))
        # Weights and transition kernels
        # In logarithmic scale:   Log 𝑝(𝐼 = 𝑖 ∣ 𝑦_(1:𝑁+1)) = LogPosteriors + LogKs
        #                                                 = LogPosteriors + LogKiz + sum(LogKzi) - LogKzi
        #  K(yᵢ, z) = (2π)^(-d/2) |Σ|^(-1/2) * exp( -1/2 * (z - μ(yᵢ))ᵀ * Σ^(-1) * (z - μ(yᵢ)) )

        Prec   = (1 / (step_size^2 * CovScaling^2)) .* G            # Precision matrix, Σ^-1 = (step*cov_scaling)^2 * G
        R = cholesky(Symmetric(Prec)).U                             # Precision Cholesky upper-triangular, Σ^(-1) = R'R

        for i in 1:(N_prop + 1)
            yi = proposals[i, 1:D, l] 

            # Posterior log-density: log π(yi)
            log_post_i[i] = f_logpost(tuple(yi...))

            # μ(yi): Langevin drift 
            mu_yi   = yi + alpha * (G_inv * grad_fd!(grad, f_logpost, yi; h=h_der))

            # FORWARD TRANSITION  log K(yi -> z): -1/2 (z - μ(yi))' Prec (z - μ(yi))
            diff = z - mu_yi
            half_dot_1 = R * diff                                  # Prec^(1/2) * diff
            logK_yi_z[i] = -0.5 * dot(half_dot_1, half_dot_1)

            # BACKWARD TRANSITION log K(z -> yi): -1/2 (yi - μ(z))' Prec (yi - μ(z))
            diff_2 = yi - mu_z
            half_dot_2 = R * diff_2
            logK_z_yi[i] = -0.5 * dot(half_dot_2, half_dot_2)

        end

        # Importance weights w_i ∝ π(yi) K(yi,y\i)
        logw .= log_post_i .+ logK_yi_z .+ sum(logK_z_yi) .- logK_z_yi
        logw .-= maximum(logw)
        logZ = log(sum(exp.(logw)))
        w[l, 1:(N_prop+1)] .= exp.(logw .- logZ)

        # Resampling step and Update xI
        I_new = findfirst(cumsum(w[l,:]) .>= wcud[end, D + 1])
        accept_proxy += (1.0 - w[l, I_new])
        chain[l, :] = proposals[I_new, :, l]
        xI = chain[l, :]
        
        # Early stopping criterion: norm(xI​−xl−1​)=∑​(xI,j ​− xl−1,j​)^2​
        if l > 1 && norm(xI .- chain[l-1, :]) ≤ tol_move
            stall_cnt += 1
        else
            stall_cnt = 0
        end

        l_effective = l

        if stall_cnt ≥ max_stall
            break
        end
    end

    chain = chain[1:l_effective, :]
    w     = w[1:l_effective, :]
    proposals = proposals[:, :, 1:l_effective]

    return (
            proposals = proposals,
            chain = chain,
            accept_proxy = accept_proxy / l_effective,
            grad = grad,
            weights = w,
            FisherInfo = G,
            Scaling = L)
end


######################################################################################################################################################
######################################################################################################################################################

function lotka_volterra!(du, u, p, t)
    alpha, beta, delta, gamma = p
    x, y = u
    du[1] = alpha*x - beta*x*y
    du[2] = -gamma*y + delta*x*y
end

Theta_true = (1.5, 0.1, 0.075, 1.0); tspan = (0.0, 1.0); dt = 0.05; u0 = [1.0, 1.0]; 
lok_volt = ODEProblem(lotka_volterra!, u0, tspan, Theta_true);
sol = solve(lok_volt, Tsit5(), saveat=dt);   

# Noisy Data 
sigma_eta = .01 * I(2)              
obs_noisy = Array(sol) .+ rand(MvNormal(zeros(2), sigma_eta), size(sol, 2)); 

#----------------------------------------------------------------------------------------------------------------#
p1 = plot(sol[1,:], sol[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, grid = true, gridalpha = 0.3, title = "True Orbit", fontsize =8)
p2 = plot(obs_noisy[1,:], obs_noisy[2,:], linewidth = 1.5, xlabel = "x", ylabel = "y", legend = false, color = "red", grid = true, gridalpha = 0.3, title = "Observed Orbit", fontsize =8)
lv_orbits = plot( p1, p2, layout = (1, 2), size = (700, 350), bottom_margin=5mm, top_margin=5mm)
#----------------------------------------------------------------------------------------------------------------#


priors = (
          Gamma(2, 1),   # alpha
          Gamma(2, 1),   # beta 
          Gamma(2, 1),   # delta
          Gamma(2, 1)    # gamma
)

logprior_par = p -> (logpdf(priors[1], p[1]) + logpdf(priors[2], p[2]) + logpdf(priors[3], p[3]) + logpdf(priors[4], p[4]))

N_iter = 2000; 
a = [ 1.1,  1.1,  1.1,  1.1]
init_par = log.(a)

println("Running sm-MALA MCMC...")
mcqmc_time = @elapsed out = MP_IS_MALA_LV_2(lotka_volterra!, u0, obs_noisy, sigma_eta, tspan, dt, logprior_par, init_par, N_iter=N_iter, step_size= 0.03);
println("Execution time: $(mcqmc_time) sec")


mean(out.chain_par[500:end, :], dims=1)
out.grad_record
out.acc_rate
# ------------------------------------------------------------------------------------------------------------------#

for i in 1:4
    println("ESS $(i): Parameters = ", ess_ips(out.chain_par[:, i]))
end

#------------------------------------------------------------------------------------------------------------------# 

chain = out.chain_par
iters = 1:size(chain, 1)

# Create trace plots
gr()
p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [Theta_true[3]], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [Theta_true[4]], linestyle=:dash, color=:red)

# Combine in a 4×1 layout
plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false)


####--------------------------------------------------------------------------------------------####


chain= out.grad_record
iters = 1:size(chain, 1)

# Create trace plots
gr()
p1 = plot(iters, chain[:, 1], label="α", title="Trace of α", xlabel="Iteration", ylabel="Value")
hline!(p1, [Theta_true[1]], linestyle=:dash, color=:red)
p2 = plot(iters, chain[:, 2], label="β", title="Trace of β", xlabel="Iteration", ylabel="Value")
hline!(p2, [Theta_true[2]], linestyle=:dash, color=:red)
p3 = plot(iters, chain[:, 3], label="δ", title="Trace of δ", xlabel="Iteration", ylabel="Value")
hline!(p3, [Theta_true[3]], linestyle=:dash, color=:red)
p4 = plot(iters, chain[:, 4], label="γ", title="Trace of γ", xlabel="Iteration", ylabel="Value")
hline!(p4, [Theta_true[4]], linestyle=:dash, color=:red)

# Combine in a 4×1 layout
plot(p1, p2, p3, p4, layout=(4,1), size=(900,800), legend=false)

#--------------------------------------------------------------------------------------------####


lok_volt_new = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain_par[end,:])...))
sol_new = solve(lok_volt_new, Tsit5(), saveat = dt)

mid_iter = max(1, div(size(out.chain_par, 1), 2))
lok_volt_t1 = ODEProblem(lotka_volterra!, u0, tspan, tuple(exp.(out.chain_par[mid_iter,:])...))
sol_new_t1 = solve(lok_volt_t1, Tsit5(), saveat = dt)

lok_volt_initial = ODEProblem(lotka_volterra!, u0, tspan, tuple(a...))
sol_initial = solve(lok_volt_initial, Tsit5(), saveat = dt)

p_overlay = plot(sol[1, :], sol[2, :], linewidth = 1.5, color = "blue", label = "True orbit", xlabel = "x", ylabel = "y", title = "Lotka–Volterra Orbits (Overlayed)", grid = true, gridalpha = 0.3, fontsize = 8)
plot!(p_overlay, sol_initial[1, :], sol_initial[2, :], linewidth = 1.5, color = "black", label = "Initial orbit")
plot!(p_overlay, sol_new_t1[1, :], sol_new_t1[2, :], linewidth = 1, color = "black", linestyle = :dashdot, label = "Mid-chain orbit")
plot!(p_overlay, obs_noisy[1, :], obs_noisy[2, :], linewidth = 1.5, color = "red", label = "Observed orbit")
plot!(p_overlay, sol_new[1, :], sol_new[2, :], linewidth = 1.5, color = "green", label = "Reconstructed orbit")

####--------------------------------------------------------------------------------------------####

lv_orbits_sm = plot(sol.t, obs_noisy[1,:], lw=2, label="x true",
    grid=true, gridalpha=0.3, legendfontsize=9,
    legend=:outertop, legendcolumns=2, legendborder=false,
    legend_foreground_color=:transparent,
    title="Lotka-Volterra Dynamics (sm-MALA)", fontsize=10)
plot!(sol.t, obs_noisy[2,:], lw=2, label="y true")
plot!(sol.t, sol_new[1,:], lw=2, ls=:solid, label="x sm-MALA", color=:blue)
plot!(sol.t, sol_new[2,:], lw=2, ls=:solid, label="y sm-MALA", color=:red)
xlabel!("Time")

####--------------------------------------------------------------------------------------------####

function f_sse(p)
    sse = 0.0; resid = zeros(size(obs_noisy, 1)); 
    lv_end = ODEProblem(lotka_volterra!, u0, tspan, p);
    sol = solve(lv_end, Tsit5(), saveat=dt);
    F = cholesky(Symmetric(sigma_eta))
    for t in 1:size(obs_noisy, 2)
        @views resid .= obs_noisy[:, t] .- sol[:, t]
        y = F.U \ resid
        sse += dot(y, y)
    end
    return sse
end

f_sse(a)
f_sse(exp.(out.chain_par[end,:]))
f_sse(Theta_true)
