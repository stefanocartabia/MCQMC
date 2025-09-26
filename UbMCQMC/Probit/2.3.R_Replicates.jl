include("2.2.F_k_m.jl")

########################################### E[F_k_m] and Var[F_k_m] #################################################
#############################################       Biased       ####################################################


function R_Fkm_probit(mdl::UnMCQMC_probit, seq::Array{Float64,3}, n_iter::Int, k::Int, R::Int, f::Function)
    p = mdl.p
    R_Fkm = Matrix{Float64}(undef, p, R)   # <--- just p
    for r in 1:R
        R_Fkm[:,r] = Fkm_probit(mdl, seq[:,:,r], n_iter, k, f)
    end

    mu_pool = mean(R_Fkm; dims=2)
    sigma_pool = sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1))

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
end


######################################## E[F_k_m] and Var[F_k_m] #################################################
########################################     Unbiased    #########################################################
# This script aims to estimate the unbiased F_k_m over R indipendent chains to derive the mean and variance 


function R_UbFkm_probit(mdl::UnMCQMC_probit, seq::Array{Float64,3}, n_iter::Int, k::Int, R::Int, f::Function)
    p = mdl.p
    R_Fkm = Matrix{Float64}(undef, p, R)
    for r in 1:R
        R_Fkm[:,r] = UbFkm_probit(mdl, n_iter, seq[:,:,r], k, f)
    end

    mu_pool = mean(R_Fkm; dims=2)
    sigma_pool = sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1))

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
    
end


println("Check Probit: Imported estimator of E[F_k_m] and Var[F_k_m]")

