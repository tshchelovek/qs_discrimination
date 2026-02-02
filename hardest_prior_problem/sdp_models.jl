#=
models:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-12
=#

using JuMP, MosekTools, IsApprox
import LinearAlgebra
import Dualization
import SCS
import Mosek

module Distribs
    include("distribs.jl")
end

optimizer = SCS.Optimizer

function primal_model(N, d, ρ, p = [1/N for i in 1:N], opt = optimizer)

    model = Model(opt)
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

    solution = [value.(e) for e in E]
    return objective_value(model), solution
end

function dual_model_with_p(N, d, ρ, p, opt = optimizer)

    model_dual = Model(opt)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())

    cs = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)

    return objective_value(model_dual), value.(y)
end

function dual_model(N, d, ρ, opt = optimizer)

    model_dual = Model(opt)
    set_silent(model_dual)

    y = @variable(model_dual, [1:d, 1:d] in HermitianPSDCone())
    p = [@variable(model_dual) for i in 1:N]

    cs1 = [@constraint(model_dual, LinearAlgebra.Hermitian(p[i] * ρ[i]) <= y, HermitianPSDCone()) for i in 1:N]
    cs2 = [@constraint(model_dual, p[i] >= 0) for i in 2:N]
    @constraint(model_dual, sum(p) == 1)

    @objective(
        model_dual, 
        Min, 
        LinearAlgebra.tr(y))

    optimize!(model_dual)
    @assert is_solved_and_feasible(model_dual; dual = true)

    return objective_value(model_dual), value.(y), value.(p)
end

function double_dual_model_with_q(N, d, ρ, opt = optimizer)

    model = Model(opt)
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

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function double_dual_model(N, d, ρ, opt = optimizer)

    model = Model(opt)
    set_silent(model)

    M = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]

    @constraint(model, sum(M) == LinearAlgebra.I)
    cs = [@constraint(model, real(LinearAlgebra.tr(ρ[i] * M[i])) >= real(LinearAlgebra.tr(ρ[1] * M[1]))) for i in 2:N]

    @objective(
        model,
        Max,
        real(LinearAlgebra.tr(ρ[1] * M[1])),
    )

    # print(model)

    optimize!(model)
    @assert is_solved_and_feasible(model)

    solution = [value.(m) for m in M]
    return objective_value(model), solution
end

function sdp_solver(N = 2, d = 2)

    # ρ = [Distribs.random_state(d) for i in 1:N]
    # ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5]] # |0> and |+>
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [1/2 0 ; 0 1/2]]
    #  ρ = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    # ρ = [[1 0 ; 0 0], [0.81 0.39 ; 0.39 0.19], [0.64 0.48 ; 0.48 0.36]]
    # ρ = [[0.65831 + 0.0im -0.0 - 0.0im; 0.0 + 0.0im 0.34169 - 0.0im], [0.37922 - 0.0im -0.17691 + 0.42249im; -0.17691 - 0.42249im 0.62078 + 0.0im]]
    # ρ = [[1 0 ; 0 0], [0 0 ; 0 1], [0.5 0.5 ; 0.5 0.5], [0.5 -0.5 ; -0.5 0.5]]
    # ρ = [[1 0 ; 0 0], [0.1464466094067263 0.35355339059327384 ; 0.35355339059327384 0.8535533905932737], [0.1464466094067263 -0.35355339059327384 ; -0.35355339059327384 0.8535533905932737]]

    z = zeros(d, d)
    z[1,1]=1
    ρ = [z, LinearAlgebra.I(d)/d]
    # ket 0 and fully mixed give as close as you want to deterministic

    # p = [0, 1]
    # p = [0.5, 0.5]
    # p = [0.001, 0.999]
    p = [1 / N for i in 1:N]

    # sol_primal = primal_model(N, d, ρ, p)
    # sol_dual = dual_model_with_p(N, d, ρ, p)
    sol_dual = dual_model(N, d, ρ)
    # sol_double_dual = double_dual_model(N, d, ρ)

    # println("prob_primal = ", sol_primal[1], "\n  prob_dual = ", sol_dual[1])
end


sdp_solver(2, 3)