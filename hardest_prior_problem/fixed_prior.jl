#=
hardest_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-12
=#

using JuMP, MosekTools, IsApprox
import LinearAlgebra
import Dualization
import SCS
import Mosek

function random_state(d)
    x = randn(ComplexF64, (d, d))
    y = x * x'
    return LinearAlgebra.Hermitian(y / LinearAlgebra.tr(y))
end

function primal_model(N, d, ρ, p)

    model = Model(SCS.Optimizer)
    set_silent(model)

    E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(E) == LinearAlgebra.I)

    @objective(
        model,
        Max,
        sum(p[i] * real(LinearAlgebra.tr(ρ[i] * E[i])) for i in 1:N),
    )

    optimize!(model)
    @assert is_solved_and_feasible(model)
    solution_summary(model)

    objective_value(model)

    0.5 + 0.25 * sum(LinearAlgebra.svdvals(ρ[1] - ρ[2]))

    solution = [value.(e) for e in E]
    return objective_value(model), solution
end

function dual_model(N, d, ρ, p)

    model_dual = Model(SCS.Optimizer)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())

    cs = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    # print(model_dual)

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)
    solution_summary(model_dual)

    objective_value(model_dual)

    # print("Minimal probability p = ", objective_value(model_dual), " is achieved for\n y = ", value.(y))
    solution = [objective_value(model_dual), value.(y)]
    return objective_value(model_dual), value.(y)
end

function double_dual_model(N, d, ρ)

    model = Model(SCS.Optimizer)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
    q = [@variable(model) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs1 = [@constraint(model, q[i] <= 0) for i in 1:N]
    # cs2 = [@constraint(model, q[i] + real(LinearAlgebra.tr(ρ[i] * M[i])) == q[1] + real(LinearAlgebra.tr(ρ[1] * M[1]))) for i in 2:N]

    @objective(
        model,
        Max,
        q[1] + real(LinearAlgebra.tr(ρ[1] * M[1])),
    )

    print(model)

    optimize!(model)
    @assert is_solved_and_feasible(model)
    solution_summary(model)

    objective_value(model)

    # 0.5 + 0.25 * sum(LinearAlgebra.svdvals(ρ[1] - ρ[2]))

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function double_dual_no_q_model(N, d, ρ)

    model = Model(SCS.Optimizer)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) >= real(LinearAlgebra.tr(ρ[1] * M[1]))) for i in 2:N]

    @objective(
        model,
        Max,
        real(LinearAlgebra.tr(ρ[1] * M[1])),
    )

    print(model)

    optimize!(model)
    @assert is_solved_and_feasible(model)
    solution_summary(model)

    objective_value(model)

    # 0.5 + 0.25 * sum(LinearAlgebra.svdvals(ρ[1] - ρ[2]))

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function sdp_solver()
    N, d = 2, 2

    # ρ = [random_state(d) for i in 1:N]
    ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    #  ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    # p = [0, 1]
    p = [0.5, 0.5]
    # p = [0.001, 0.999]

    # sol_primal = primal_model(N, d, ρ, p)
    # sol_dual = dual_model(N, d, ρ, p)
    sol_double_dual = double_dual_no_q_model(N, d, ρ)

    # println(isapprox(sol_primal[1], sol_dual[1]))
    # println("prob_primal = ", sol_primal[1], "\n  prob_dual = ", sol_dual[1])
end


sdp_solver()