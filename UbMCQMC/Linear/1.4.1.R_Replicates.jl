include("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/UbMCQMC/Linear/1.3.F_k_m.jl")

########################################### E[F_k_m] and Var[F_k_m] #################################################
#############################################       Biased       ####################################################


function R_Fkm(R::Int, n_iter::Int,seq,mdl,k,f)

    p = mdl.p
    R_Fkm = Matrix{Float64}(undef, p+1, R)
    for r in 1:R
        R_Fkm[:,r] = Fkm(n_iter,seq[:,:,r],mdl,k,f)
    end

    mu_pool = dropdims(mean(R_Fkm; dims=2), dims=2)
    sigma_pool = dropdims(sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1)), dims=2)

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
    
end

######################################## E[F_k_m] and Var[F_k_m] #################################################
########################################     Unbiased    #########################################################
# This script aims to estimate the unbiased F_k_m over R indipendent chains to derive the mean and variance 

function R_UbFkm(R::Int, n_iter::Int,seq,mdl,k,f)

    p = mdl.p
    R_Fkm = Matrix{Float64}(undef, p+1, R)
    for r in 1:R
        R_Fkm[:,r] = UbFkm(n_iter,seq[:,:,r],mdl,k,f)
    end

    mu_pool = dropdims(mean(R_Fkm; dims=2), dims=2)
    sigma_pool = dropdims(sum((R_Fkm .- mu_pool).^2; dims=2) ./ (R*(R-1)), dims=2)

    return (
        mu_pool = mu_pool,
        sigma_pool = sigma_pool
    )
    
end


println("Check: Imported estimator of E[F_k_m] and Var[F_k_m]")





