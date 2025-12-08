#=
just a calculator:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-11-12
=#

using JuMP
import LinearAlgebra
import SCS

# M = [[0.24999939042521085 + 0.0im -0.24999913525896275 + 0.0im; -0.24999913525896275 + 0.0im 0.24999939042521085 + 0.0im], [0.7500006092572504 + 0.0im 0.24999913472792443 + 0.0im; 0.24999913472792443 - 0.0im 0.7500006092572504 + 0.0im]]
# ρ = [[1 0; 0 0], [1 0; 0 0]]

M = [[0.853551627355789 + 0.0im -0.3535516302948938 + 0.0im; -0.3535516302948938 + 0.0im 0.14644836638863346 + 0.0im],
    [0.14644836614662668 + 0.0im 0.35355163052118227 + 0.0im; 0.35355163052118227 - 0.0im 0.8535516267920369 + 0.0im]]
ρ = [[1 0; 0 0], [1 0; 0 0]]

LinearAlgebra.tr(ρ[2] * M[1])

(sqrt(2)-1)/sqrt(2)/2



# f = open("qs_discrimination/hardest_prior_problem/distributions/2_00.txt", "r")
# m = read(f, String)
# # m = raw"[1, 0; 0.9, 0.1; 0.8, 0.2; 0.7, 0.3; 0.6, 0.4; 0.5, 0.5; 0.4, 0.6; 0.3, 0.7; 0.2, 0.8; 0.1, 0.9; 0, 1]"
# # A = Meta.parse(m) 
# m_spl = split(m, '\n')
# probs = Matrix{Float16}(undef, length(m_spl))
# for i in 1:1:length(m_spl)
#     line_spl = split(m_spl[i])
#     probs[i] = [Float16(line_spl[1]), Float16(line_spl[2])]
#     print(probs[i])
# end

# x = Matrix{Float64}(undef, 1, 0)
# append!(x, 1)
# append!(x, Matrix{Float64}(reduce(hcat, [[5, 4]])))
# append!(x[1], Matrix{Float64}(reduce(hcat, [5, 4])))
x = Vector{Vector{Int64}}()
# append!(x[1], [1])
# append!(x, [[5, 4]])
# append!(x[1], [1])
# println(x)

prob_recursion(3, 1, 0.2)


M = [[0.640000272025707 + 0.0im -0.1600006833360107 + 0.0im; -0.1600006833360107 + 0.0im 0.03999974346694657 + 0.0im], 
     [0.31999975198746244 + 0.0im 0.3200011567343735 + 0.0im; 0.3200011567343735 - 0.0im 0.3200008315665967 + 0.0im], 
     [0.03999997737555185 + 0.0im -0.16000047478529925 + 0.0im; -0.16000047478529925 + 0.0im 0.6399994273708338 + 0.0im]]
rho = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]

lam = [0.86 -0.15; -0.15 0.86]
sum_rho = sum(rho)
lam * rho[3] * lam

sqrt(1 - 0.8 * 0.8)
