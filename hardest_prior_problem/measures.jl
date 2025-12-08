#=
measurements:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-02
=#

import LinearAlgebra

function PGM_observables(ρ, p_set = [])
    N = size(ρ)[1]

    if p_set == []
        p_set = [1 / N for i in 1:N]
    end

    lambda = sum(ρ)
    lambda = inv(lambda)
    lambda = sqrt(lambda)

    M = [lambda * ρ[i] * lambda for i in 1:N]
    return M
end

function PGM(ρ, p_set = [])
    N = size(ρ)[1]
    if p_set == []
        p_set = [1 / N for i in 1:N]
    end

    M = PGM_observables(ρ, p_set)
    p_obj = [LinearAlgebra.tr(ρ[i] * M[i]) * p_set[i] for i in 1:N]

    return sum(p_obj), M
end

function main()
    N, d = 3, 2

    # ρ = [random_state(d) for i in 1:N]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    ρ = [[1 0 ; 0 0], [0.81 0.39 ; 0.39 0.19], [0.64 0.48 ; 0.48 0.36]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    # p = [[0, 1]]
    # p = [[0.5, 0.5]]
    # p = [[0.001, 0.999]]
    # p = [[0.2, 0.3, 0.5]]

    # greedy_dual, greedy_probs = greedy_prior_min(N, d, ρ, p)

    # dual_value, dual_solution = FixedPrior.double_dual_model(N, d, ρ)
    # println("checking primal problem: ", check_primal[1])

    # return greedy_dual, greedy_probs

    PGM(ρ)

end

main()