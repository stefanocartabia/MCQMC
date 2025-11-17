
# - Chen's Thesis, 8.2.1 Construction of Variate Matrix.
function variate_mat(seq::Vector, d::Int) 
    N = length(seq)
    M = Matrix{Float64}(undef, N, d)

    # y: smallest coprime with N
    y = minimum(k for k in d:N-1 if gcd(k, N) == 1)

    for i in 1:N
        for j in 1:d
            # idx = ((i−1)⋅y+j) (modN)    
            idx = mod1((i - 1) * y + j, N)
            M[i,j] = seq[idx]
        end
    end

    return M
end


#=
wcud_12 = variate_mat(vec_wcud_12,3)
size(wcud_12)

N = length(vec_wcud_12); y= 4
wcud_12[1,:] == vec_wcud_12[[1,2,3]]
wcud_12[2,:] == vec_wcud_12[[5,6,7]]
wcud_12[end,:] == vec_wcud_12[[4092,4093,4094]]    # 16377, 16378, 16379 (mod 4095)

# Check the variate matrix has no duplicates
M = variate_mat(vec_wcud_12,3)
unique_rows = unique(eachrow(M))
size(M,1) - length(unique_rows)

=#

# -Chen's Thesis, 8.2.2 Randomization

function digit_shift_mat(seq::Vector, d::Int, R::Int, k::Int=32)
    # 1. Construct the variate matrix
    M = variate_mat(seq, d)             # size (2^n)-1 × d
    N, d = size(M)
    scale = 2.0^k                       # used to truncate to k binary digits

    U = Array{Float64}(undef, d, N, R)

    # 3. Precompute binary weights for reconstruction
    #    pow2 = [2^-1, 2^-2, ..., 2^-k]
    pow2 = 2.0 .^ -(1:k)

    for r in 1:R
        # Step 1-2: Random digital shift
        w = rand(d)                             # uniform random numbers for each column
        zbits = falses(k, d)                    # matrix of boolean values, initialise with False
        for j in 1:d
            x = floor(Int, w[j] * scale)        # take first k bits of w[j]
            @inbounds for bit in 1:k
                zbits[bit, j] = (x >> (k-bit)) & 1 == 1
            end
        end

        # Step 3: Matrix Binary Expension
        Mbits = falses(k, N, d)                 # # Mbits[bit, i, j]
        for j in 1:d
            for i in 1:N
                x = floor(Int, M[i,j] * scale)  # truncate to k-bit integer
                @inbounds for bit in 1:k
                    Mbits[bit, i, j] = (x >> (k-bit)) & 1 == 1
                end
            end
        end

        # Step 4: XOR or addition in mod2

        Mshift = falses(k, N, d)
        for j in 1:d
            @views Mshift[:, :, j] .= xor.(Mbits[:, :, j], zbits[:, j])
        end

        # Convert shifted bits back into real numbers in [0,1)
        for j in 1:d
            # Multiply bits (N×k) with pow2 (k×1) → N×1 vector
            U[j, :, r] = (Mshift[:, :, j]' * pow2)
        end
    end
    return U
end

###################################################################################################################################################################################
############################################################################ VARIATE MATRIX ALREADY DEFINED #######################################################################
###################################################################################################################################################################################

function digital_shift(M::Matrix, R::Int, k::Int=32)
    # 1. Construct the variate matrix
    # M is the variate matrix of size (2^n)-1 × d
    N, d = size(M)
    scale = 2.0^k                       # used to truncate to k binary digits

    U = Array{Float64}(undef, d, N, R)

    # 3. Precompute binary weights for reconstruction
    #    pow2 = [2^-1, 2^-2, ..., 2^-k]
    pow2 = 2.0 .^ -(1:k)

    for r in 1:R
        # Step 1-2: Random digital shift
        w = rand(d)                             # uniform random numbers for each column
        zbits = falses(k, d)                    # matrix of boolean values, initialise with False
        for j in 1:d
            x = floor(Int, w[j] * scale)        # take first k bits of w[j]
            @inbounds for bit in 1:k
                zbits[bit, j] = (x >> (k-bit)) & 1 == 1
            end
        end

        # Step 3: Matrix Binary Expension
        Mbits = falses(k, N, d)                 # # Mbits[bit, i, j]
        for j in 1:d
            for i in 1:N
                x = floor(Int, M[i,j] * scale)  # truncate to k-bit integer
                @inbounds for bit in 1:k
                    Mbits[bit, i, j] = (x >> (k-bit)) & 1 == 1
                end
            end
        end

        # Step 4: XOR or addition in mod2

        Mshift = falses(k, N, d)
        for j in 1:d
            @views Mshift[:, :, j] .= xor.(Mbits[:, :, j], zbits[:, j])
        end

        # Convert shifted bits back into real numbers in [0,1)
        for j in 1:d
            # Multiply bits (N×k) with pow2 (k×1) → N×1 vector
            U[j, :, r] = (Mshift[:, :, j]' * pow2)
        end
    end
    return U
end



###################################################################################################################################################################################
############################################################################   Algorithm 9 Tobias Schwedes   ######################################################################
###################################################################################################################################################################################

function tobias_wcud(seq::Vector{Float64},dtilde::Int, N::Int, L::Int)

    P = length(seq)
    b = dtilde + 1              # since alpha = 1
    n_needed = L * N * b        # total number of CUD points needed

    # Check if base sequence is long enough
    if P < n_needed
        error("Sequence length $P is too short: need at least $n_needed points.")
    end

    # Step 2: Trim to a multiple of b
    T = (P ÷ b) * b             # T = floor(P / b) * b

    if T < L * N
        error("After trimming, base length T=$T is too small for $(L*N) blocks.")
    end

    u = seq[1:T]

    # Step 3–5: generate b cyclic shifts and concatenate them
    big = Vector{Float64}(undef, b * T)
    k = 1

    for shift in 0:(b-1)    # shift = 0,...,b-1
        start = shift + 1   # starting index of the rotation

        @inbounds begin
            # from start to T
            for t in start:T
                big[k] = u[t]; k += 1
            end
            # from 1 to start-1
            for t in 1:(start-1)
                big[k] = u[t]; k += 1
            end
        end
    end

    # Step 6: keep only what Algorithm 13 needs
    if n_needed > length(big)
        error("Internal error: concatenated sequence too short.")
    end

    big = big[1:n_needed]

    # Reshape into matrix (L*N) × b
    M = reshape(big, b, L*N)'
    return M
end