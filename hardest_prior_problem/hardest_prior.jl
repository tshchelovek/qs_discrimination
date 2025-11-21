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

function random_state(d)
    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

function prob_recursion(N, sum, step = 0.01)
    if N == 1
        return [[sum]]
    end
    
    res = Vector{Vector{Float16}}()
    for p in range(start = 0, step = step, stop = sum)
        cur_rec = prob_recursion(N - 1, sum - p, step)
        for vec in cur_rec
            mid_res = [p]
            append!(mid_res, vec)
            append!(res, [mid_res])
        end
    end
    return res
end

function greedy_prior_set(N, d, ρ, step = 0.1)
    # ONLY WORKS FOR N = 2
    p0_set = [i for i in range(0, step = step, 1 - step)]
    
    sol_dual_set = Vector{Any}(undef, length(p0_set))
    for i in range(start = 1, stop = length(p0_set))
        p0 = p0_set[i]
        p = [p0, 1 - p0]
        sol_dual = dual_model(N, d, ρ, p)
        sol_dual_set[i] = sol_dual
    end
    return sol_dual_set
end

function greedy_prior_min(N, d, ρ, step = 0.1)
    # ONLY WORKS FOR N = 2
    p0_set = [i for i in range(0, step = step, 1 - step)]

    res_min = 1
    sol_min = [res_min, ρ[1]]
    p_set = [1, 0]
    for i in range(start = 1, stop = length(p0_set))
        p0 = p0_set[i]
        p = [p0, 1 - p0]
        sol_dual = FixedPrior.dual_model(N, d, ρ, p)
        if sol_dual[1] < res_min
            sol_min = sol_dual
            res_min = sol_dual[1]
            p_set = [p0_set[i], 1 - p0_set[i]]
        end
    end
    return sol_min, p_set
end

function greedy_prior(N, d, ρ, step = 0.1, min = true)
    if min == true
        return greedy_prior_min(N, d, ρ, step)
    else
        return greedy_prior_set(N, d, ρ, step)
    end
end

function sdp_solver()
    N, d = 3, 2

    # ρ = [random_state(d) for i in 1:N]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    # p = [0, 1]
    # p = [0.5, 0.5]
    # p = [0.001, 0.999]

    greedy_dual, greedy_probs = greedy_prior_min(N, d, ρ, 0.05)

    check_primal = FixedPrior.primal_model(N, d, ρ, greedy_probs)
    println(check_primal[1])

    # println(isapprox(sol_primal[1], sol_dual[1]))
    # println(" = ", greedy_dual[1], "\n  prob_dual = ", sol_dual[1])

    return greedy_dual, greedy_probs
end

sdp_solver()
# greedy_prior()