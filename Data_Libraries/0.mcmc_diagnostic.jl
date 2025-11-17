
using FFTW
# function acovf(x::AbstractVector{<:Real})
#     n = length(x)
#     m = mean(x)
#     y = x .- m
#     m2 = 2^ceil(Int, log2(2n))          # padding a potenza di 2
#     fy = rfft(vcat(y, zeros(m2 - n)))
#     ac = irfft(abs.(fy).^2, m2)[1:n]
#     ac ./= (n:-1:1)                      # correzione per fine serie
#     return ac
# end

# # ESS con initial positive sequence (Geyer)
# function ess_ips(x::AbstractVector{<:Real})
#     n = length(x)
#     ac = acovf(x)
#     γ0 = ac[1]
#     ρ =  ac[2:end] ./ γ0            # autocorrelazioni ai lag ≥ 1
#     s = 0.0
#     k = 1
#     while k < length(ρ)
#         s_pair = ρ[k] + (k+1 <= length(ρ) ? ρ[k+1] : 0.0)
#         if s_pair <= 0
#             break
#         end
#         s += s_pair
#         k += 2
#     end
#     τ = 1 + 2s                           # time-series variance factor
#     return n / τ
# end

# # ESS per più catene: concateno dopo burn-in (assumendo stazionarietà/mescolamento simile)
# function ess_ips_multichain(chains::Vector{<:AbstractVector{<:Real}})
#     x = vcat(chains...)
#     return ess_ips(x)
# end



function acovf_fft(x::AbstractVector{<:Real};
                   maxlag::Int = 250,
                   biased::Bool = true)

    n = length(x)
    n <= 1 && error("Serve almeno 2 campioni")

    maxlag = min(maxlag, n - 1)

    μ = mean(x)
    y = x .- μ

    m2 = 2^ceil(Int, log2(2n))           # padding a potenza di 2
    fy = rfft(vcat(y, zeros(m2 - n)))
    ac = irfft(abs.(fy).^2, m2)[1:(maxlag + 1)]

    if biased
        ac ./= n                         # stimatore biased
    else
        ac ./= (n:-1:(n - maxlag))       # stimatore unbiased
    end

    return ac
end

"""
    ess_imps(x; maxlag=250)

Stima dell’ESS con Geyer Initial *Monotone* Positive Sequence.
Accetta:

- `x::AbstractVector`  -> singola catena (draws)
- `x::AbstractMatrix`  -> più catene (draws × chains)

Restituisce l’ESS per il *mean* (stile `ess_bulk` ma senza rank-normalization).
"""
function ess_imps(x::AbstractVector{<:Real}; maxlag::Int = 250)
    n = length(x)
    n < 4 && return 0.0

    ac = acovf_fft(x; maxlag=maxlag, biased=true)
    γ0 = ac[1]
    (γ0 <= 0 || !isfinite(γ0)) && return 0.0

    ρ = ac[2:end] ./ γ0   # lag ≥ 1

    # Geyer IMPS su coppie (ρ_k + ρ_{k+1})
    s = 0.0
    prev = Inf
    k = 1
    while k <= length(ρ)
        Γ = ρ[k] + (k+1 <= length(ρ) ? ρ[k+1] : 0.0)

        # sequenza monotona non-crescente
        if Γ > prev
            Γ = prev
        end

        # initial positive sequence
        if Γ <= 0
            break
        end

        s += Γ
        prev = Γ
        k += 2
    end

    τ = 1 + 2s
    ess = n / τ

    # per sicurezza limitiamo a [0, n]
    return max(0.0, min(ess, n))
end

function ess_imps(x::AbstractMatrix{<:Real}; maxlag::Int = 250)
    # x ha shape (draws, chains)
    n, m = size(x)
    n < 4 && return 0.0

    # centra ogni catena
    y = x .- mean(x, dims=1)

    # autocovarianze per catena
    maxlag = min(maxlag, n - 1)
    acs = [acovf_fft(view(y, :, j); maxlag=maxlag, biased=true) for j in 1:m]

    # varianza media (lag 0)
    γ0 = mean(ac[1] for ac in acs)
    (γ0 <= 0 || !isfinite(γ0)) && return 0.0

    # autocorrelazione media sui lag k
    ρ = similar(acs[1][2:end])
    for k in eachindex(ρ)
        ρ[k] = mean(ac[k+1] for ac in acs) / γ0
    end

    # IMPS sugli ρ medi
    s = 0.0
    prev = Inf
    k = 1
    while k <= length(ρ)
        Γ = ρ[k] + (k+1 <= length(ρ) ? ρ[k+1] : 0.0)

        if Γ > prev
            Γ = prev
        end

        if Γ <= 0
            break
        end

        s += Γ
        prev = Γ
        k += 2
    end

    τ = 1 + 2s
    ess = (n * m) / τ          # tutte le draw di tutte le catene

    return max(0.0, min(ess, n * m))
end