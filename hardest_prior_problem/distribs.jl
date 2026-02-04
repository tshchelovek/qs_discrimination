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

function vecs_to_kets(vecs)
    # by a 3D vector return a ket
    n = length(vecs)
    res = []
    for i in 1:n
        vec = vecs[i]
        if vec[3] == -1
            v = [0, 1]
        else
            sqz = sqrt(2 * (1 + vec[3]))
            v = [sqz / 2, (vec[1] + 1im * vec[2]) / sqz]
        end
        # println(v * v')
        # println(Ket.Hermitian(v * v'))
        append!(res, [v * v'])
    end
    return res
end

# need a func for generating set of N density matrices

# ans = discrete_prob_sets(3)
# println("done.")

# a = Matrix{ComplexF64}([0.59673 + 0.0im 0.171285 + 0.216656im; 0.171285 - 0.216656im 0.40327 + 0.0im])
# b = Matrix{ComplexF64}([0.284332 + 0.0im 0.159722 - 0.416004im; 0.159722 + 0.416004im 0.715668 + 0.0im])
# c, d = rotate_states([a, b])

# distance_map([a, b, c, d])

# vecs_to_kets([[1,0,0], [-1,0,0]])
# vecs_to_kets([[1,0,0], [-.5, .86602540378443864675, 0], [-.5, -.86602540378443864675, 0]])
# vecs_to_kets([[.577350269189625763, .577350269189625763, .577350269189625763], [.577350269189625763, -.577350269189625763, -.577350269189625763], [-.577350269189625763, .577350269189625763, -.577350269189625763], [-.577350269189625763, -.577350269189625763, .577350269189625763]])
# vecs_to_kets([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
# vecs_to_kets([[0, 0, 1.], [.79422389401839640649, 0, -.60762521851076509842], [-.39711194700919820325, .68781806851253095399, -.60762521851076509842], [-.39711194700919820325, -.68781806851253095399, -.60762521851076509842], [.96164648479875939718, 0, .27429188517743176508], [-.48082324239937969859, .83281028529573164803, .27429188517743176508], [-.48082324239937969859, -.83281028529573164803, .27429188517743176508]])
# vecs_to_kets([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-.5, .86602540378443864675, 0], [-.5, -.86602540378443864675, 0]])

vecs_to_kets([[.57735026918962576449, .57735026918962576449, .57735026918962576449], [.57735026918962576449, .57735026918962576449, -.57735026918962576449], [.57735026918962576449, -.57735026918962576449, .57735026918962576449], [-.57735026918962576449, .57735026918962576449, .57735026918962576449], [.57735026918962576449, -.57735026918962576449, -.57735026918962576449], [-.57735026918962576449, .57735026918962576449, -.57735026918962576449], [-.57735026918962576449, -.57735026918962576449, .57735026918962576449], [-.57735026918962576449, -.57735026918962576449, -.57735026918962576449]])