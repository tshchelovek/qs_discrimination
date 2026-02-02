#=
hardest_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-18
=#

using JuMP, MosekTools, IsApprox, Random
import LinearAlgebra
import Dualization
import SCS
import Mosek

module SDPModels
    include("sdp_models.jl")
end
module Distribs
    include("distribs.jl")
end

# function prior_set(N, d, ρ, p_set)
#     sol_dual_set = Vector{Any}(undef, length(p_set))

#     for i in range(start = 1, stop = length(p_set))
#         sol_dual = dual_model(N, d, ρ, p_set[i])
#         sol_dual_set[i] = sol_dual
#     end
#     return sol_dual_set
# end

function prior_min(N, d, ρ)
    res_min = 2
    sol_min = [res_min, ρ[1]]
    p_min = [1, 0]

    for i in range(start = 1, stop = length(p_set))
        println(p_set[i])
        sol_dual = SDPModels.dual_model(N, d, ρ)
        if sol_dual[1] < res_min
            sol_min = sol_dual
            res_min = sol_dual[1]
            p_min = p_set[i]
        end
    end
    return sol_min, p_min
end

# function greedy_prior_set(N, d, ρ, step = 0.1)
#     p_set = Distribs.discrete_prob_sets(N, step)
#     sol_dual_set = Vector{Any}(undef, length(p_set))

#     for i in range(start = 1, step = 1, stop = length(p_set))
#         print(p_set[i], ": ")
#         sol_dual = SDPModels.dual_model_with_p(N, d, ρ, p_set[i])
#         sol_dual_set[i] = tuple(sol_dual, p_set[i])
#         println(sol_dual[1])
#     end
#     return sol_dual_set
# end

# function greedy_prior_min(N, d, ρ, step = 0.1, precision = 4)
#     res_min = 2
#     sol_min = [res_min, ρ[1]]
#     p_min = [1, 0]

#     p_set = Distribs.discrete_prob_sets(N, step)

#     for i in range(start = 1, stop = length(p_set))
#         sol_dual = SDPModels.dual_model_with_p(N, d, ρ, p_set[i])
#         sol_dual = (round(sol_dual[1], digits = precision), sol_dual[2])
#         if sol_dual[1] < res_min
#             sol_min = sol_dual
#             res_min = sol_dual[1]
#             p_min = p_set[i]
#         end
#     end
#     return sol_min, p_min
# end

# function greedy_prior(N, d, ρ, step = 0.1, min = true)
#     p_set = Distribs.discrete_prob_sets(N, step)

#     if min == true
#         return greedy_prior_min(N, d, ρ, p_set)
#     else
#         return greedy_prior_set(N, d, ρ, p_set)
#     end
# end

function sdp_solver()
    N, d = 3, 2

    # ρ = [random_state(d) for i in 1:N]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]
    # ρ = Matrix{ComplexF64}[[0.59673 + 0.0im 0.171285 + 0.216656im; 0.171285 - 0.216656im 0.40327 + 0.0im], [0.284332 + 0.0im 0.159722 - 0.416004im; 0.159722 + 0.416004im 0.715668 + 0.0im]]
    ρ = Matrix{ComplexF64}[[0.283105 + 0.0im -0.444254 - 0.074799im; -0.444254 + 0.074799im 0.716895 + 0.0im], [0.751264 + 0.0im 0.423131 - 0.088468im; 0.423131 + 0.088468im 0.248736 + 0.0im], [0.83673 + 0.0im 0.068563 - 0.363197im; 0.068563 + 0.363197im 0.16327 + 0.0im]]

    # p = [[0, 1]]
    # p = [[0.5, 0.5]]
    # p = [[0.001, 0.999]]
    # p = [[0.2, 0.3, 0.5]]
    # p = [0.8, 0.2]
    p = [[0.2, 0.38, 0.42]]

    # greedy_dual, greedy_probs = greedy_prior_min(N, d, ρ, 0.005)
    greedy_dual, greedy_probs = greedy_prior_set(N, d, ρ, 0.1)

    # primal_optimal = SDPModels.primal_model(N, d, ρ, p)
    # println("Primal problem for optimal distribution ", p, ": ", primal_optimal[1])
    # println("Measurement: ", primal_optimal[2])

    # primal_uniform = SDPModels.primal_model(N, d, ρ, [1/2, 1/2])
    # println("Primal problem for uniform distribution [0.5, 0.5]: ", primal_uniform[1])
    # println("Measurement: ", primal_uniform[2])

    # return greedy_dual, greedy_probs
    return
end

function simulator(filename, N, d, step = 0.1, precision = 4)
    filename_read = string("data/double_dual/", filename)
    filename_write = string("data/dual/", filename, "_", string(step))

    cases = Fio.read_txt(filename_read, N, d)

    for i in 1:size(cases)[1]
        rho, objective_value, objective_solution, prior = cases[i]
        # (obj_value, obj_solution), obj_prior = greedy_prior_min(N, d, rho, step, precision)
        (obj_value, obj_solution), obj_prior = SDPModels.dual_model(N, d, rho)
        println(obj_value)
        # Fio.write_txt(filename_write, rho, obj_value, obj_solution, obj_prior)
    end

    return
end

function simulator_rotated(filename, N, d, precision = 4)
    filename_read = string("data/dual/", filename)
    filename_write = string("data/dual/rotated_", filename)

    cases = Fio.read_txt(filename_read, N, d)
    to_sort = Dict()

    for i in 1:size(cases)[1]
        rho, objective_value, objective_solution, prior = cases[i]
        # (obj_value, obj_solution), obj_prior = greedy_prior_min(N, d, rho, step, precision)
        rho_rotated = Distribs.rotate_states(rho)
        # to_sort[prior] = tuple(rho_rotated, objective_value, objective_solution)
        Fio.write_txt(filename_write, rho_rotated, objective_value, [], prior)
    end
    # print(to_sort)

    return
end

function simulator_classical(N, d)

    ρ = [Matrix{Float64}(LinearAlgebra.diagm(LinearAlgebra.diag(Ket.random_state(d)))) for i in 1:N]
    # println("Generated states:\n", ρ)
    # println(LinearAlgebra.tr(ρ[1] * ρ[1]))
    
    dual_value, dual_solution = SDPModels.double_dual_model(N, d, ρ)
    # println("Double dual: ", dual_value, "\n", dual_solution)

    # println("Sanity check: 2 >= ", dual_value * N)

    return ρ, dual_value, dual_solution
end

function main()
    N = 2
    d = 2
    
    simulator_rotated("pure", N, d)

    # simulator("mixed", N, d, 0.01)
    # simulator("pure", N, d, 0.01)
    simulator("classical", N, d, 0.01)

    # Fio.analyze_value_txt("data/dual/pure", N, d) # / Fio.analyze_value_txt("data/primal/pure", N, d)

    # sdp_solver()
end

main()