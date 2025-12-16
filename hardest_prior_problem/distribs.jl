#=
distribs:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-25
=#

using IsApprox, Random, Ket
import LinearAlgebra


function discrete_prob_sets(N, step = 0.1, sum = 1)
    #= 
    generates all sets of discrete probability distributions
    for a given number of states and precision
    =#
    sum = round(sum, digits = 6)

    if N == 1
        return [[sum]]
    end
    
    res = Vector{Vector{Float16}}()
    for p in range(start = 0, step = step, stop = sum)
        cur_rec = discrete_prob_sets(N - 1, step, sum - p)
        for vec in cur_rec
            mid_res = [p]
            append!(mid_res, vec)
            append!(res, [mid_res])
        end
    end
    return res
end

function random_state(d)
    # returns a random quantum density matrix

    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

function qubit_bloch_sphere(theta, phi)
    # recovers regular qubit representation from a bloch sphere one

    return [cos(theta / 2), sin(theta / 2) * exp(im * phi)]
end

function iterate_over_disc(start = 0, stop = 1, step = 0.1, phi = 0)
    # generates a set of qubits distributed uniformly across the bloch sphere, except |0>

    start += step
    result = []
    for idx in start:step:stop
        result = [result; [qubit_bloch_sphere(idx * pi, phi)]]
        result = [result; [qubit_bloch_sphere(idx * pi, phi + pi)]]
        # println("Result:\n", result)
    end
    return result
end

function rotate_states(rho_set, precision = 5)
    # rotate states to diagonalize the first state
    t, z, vals = LinearAlgebra.schur(rho_set[1])
    rotated_rho = [z^(-1) * rho_set[i] * z for i in 1:size(rho_set)[1]]
    rotated_rho = [round.(rotated_rho[i], digits = precision) for i in 1:size(rho_set)[1]]
    return rotated_rho
end

function trace_distance(rho, sigma)
    return LinearAlgebra.tr(LinearAlgebra.sqrt((rho - sigma)' * (rho - sigma))) / 2
end

function distance_map(rho_set, precision = 5)
    d = size(rho_set)[1]
    println(d)
    result = Matrix{Float64}(undef, (d, d))
    println(result)

    for i in 1:d
        println(result[i,i])
        result[i,i] = 0
        for j in i+1:d
            t = round(trace_distance(rho_set[i], rho_set[j]), digits = precision)
            result[i,j] = t
            result[j,i] = t
        end
    end
    return result
end

# need a func for generating set of N density matrices

# ans = discrete_prob_sets(3)
# println("done.")

# a = Matrix{ComplexF64}([0.59673 + 0.0im 0.171285 + 0.216656im; 0.171285 - 0.216656im 0.40327 + 0.0im])
# b = Matrix{ComplexF64}([0.284332 + 0.0im 0.159722 - 0.416004im; 0.159722 + 0.416004im 0.715668 + 0.0im])
# c, d = rotate_states([a, b])

# distance_map([a, b, c, d])