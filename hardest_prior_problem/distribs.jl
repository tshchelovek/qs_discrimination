#=
distribs:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-25
=#

using IsApprox, Random
import LinearAlgebra

function discrete_prob_sets(N, step = 0.1, sum = 1)
    #= 
    generates all sets of discrete probability distributions
    for a given number of states and precision
    =#

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

# need a func for generating set of N density matrices