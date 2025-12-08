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

module FixedPrior
    include("fixed_prior.jl")
end
module Distribs
    include("distribs.jl")
end

function prior_set(N, d, ρ, p_set)
    sol_dual_set = Vector{Any}(undef, length(p_set))

    for i in range(start = 1, stop = length(p_set))
        sol_dual = dual_model(N, d, ρ, p_set[i])
        sol_dual_set[i] = sol_dual
    end
    return sol_dual_set
end

function prior_min(N, d, ρ, p_set)
    res_min = 2
    sol_min = [res_min, ρ[1]]
    p_min = [1, 0]

    for i in range(start = 1, stop = length(p_set))
        println(p_set[i])
        sol_dual = FixedPrior.dual_model(N, d, ρ, p_set[i])
        if sol_dual[1] < res_min
            sol_min = sol_dual
            res_min = sol_dual[1]
            p_min = p_set[i]
        end
    end
    return sol_min, p_min
end

function greedy_prior_set(N, d, ρ, step = 0.1)
    p_set = Distribs.discrete_prob_sets(N, step)
    sol_dual_set = Vector{Any}(undef, length(p_set))

    for i in range(start = 1, stop = length(p_set))
        sol_dual = dual_model(N, d, ρ, p_set[i])
        sol_dual_set[i] = sol_dual
    end
    return sol_dual_set
end

function greedy_prior_min(N, d, ρ, step = 0.1)
    res_min = 2
    sol_min = [res_min, ρ[1]]
    p_min = [1, 0]

    p_set = Distribs.discrete_prob_sets(N, step)

    for i in range(start = 1, stop = length(p_set))
        println(p_set[i])
        sol_dual = FixedPrior.dual_model(N, d, ρ, p_set[i])
        if sol_dual[1] < res_min
            sol_min = sol_dual
            res_min = sol_dual[1]
            p_min = p_set[i]
        end
    end
    return sol_min, p_min
end

# function greedy_prior(N, d, ρ, step = 0.1, min = true)
#     p_set = Distribs.discrete_prob_sets(N, step)

#     if min == true
#         return greedy_prior_min(N, d, ρ, p_set)
#     else
#         return greedy_prior_set(N, d, ρ, p_set)
#     end
# end

function sdp_solver()
    N, d = 2, 2

    # ρ = [random_state(d) for i in 1:N]
    ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    # p = [[0, 1]]
    p = [[0.5, 0.5]]
    # p = [[0.001, 0.999]]
    # p = [[0.2, 0.3, 0.5]]

    greedy_dual, greedy_probs = greedy_prior_min(N, d, ρ, p)

    check_primal = FixedPrior.primal_model(N, d, ρ, greedy_probs)
    # println("checking primal problem: ", check_primal[1])

    return greedy_dual, greedy_probs
end

sdp_solver()