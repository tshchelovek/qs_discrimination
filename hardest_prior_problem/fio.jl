#=
fio:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-03
=#

using Parsers, LinearAlgebra
using JSON3
module Distribs
    include("distribs.jl")
end

function parsing_json()
    # json_string = """{"a": 1, "b": "hello, world"}"""
    # open("double_dual_3_2.json", "w") do f
    #     JSON3.write(f, json_string)
    #     # JSON3.pretty(f, JSON3.write(x))
    #     println(f)
    # end
end

function write_txt(name, rho, objective_value, objective_solution = [[]], prior = [[]])
    N = size(rho)[1]
    d = size(rho[1])[1]
    file_name = string(name, "_", d, "_", N, ".txt")
    # file_name = string("data/", name, "_", d, "_", N, ".txt")

    precision = 4
    rho = [round.(rho[i], digits = precision + 2) for i in 1:size(rho)[1]]
    objective_value = round(objective_value, digits = precision)
    objective_solution = [round.(objective_solution[i], digits = precision) for i in 1:size(objective_solution)[1]]
    prior = [round.(prior[i], digits = precision) for i in 1:size(prior)[1]]

    try
        open(file_name, "a") do file
            write(file, string(rho), ":")
            write(file, string(objective_value), ":")
            write(file, string(objective_solution), ":")
            write(file, string(prior), "\n")
        end
    catch e
        open(file_name, "w") do file
            write(file, string(rho), ":")
            write(file, string(objective_value), ":")
            write(file, string(objective_solution), ":")
            write(file, string(prior), "\n")
        end
    end
end

function read_txt(file_name, d = 0, N = 0)
    if N > 0 && d > 0
        file_name = string(file_name, "_", d, "_", N, ".txt")
    end
    println(file_name)
    try
        open(file_name, "r") do file
            result = []
            for line in eachline(file)
                rho_str, objective_value_str, objective_solution_str, prior_str = rsplit(line, ":")
                tpl = tuple(eval(Meta.parse(rho_str)), 
                            eval(Meta.parse(objective_value_str)), 
                            eval(Meta.parse(objective_solution_str)), 
                            eval(Meta.parse(prior_str)))
                push!(result, tpl)
            end
            return result
        end
    catch e
        println("File ", file_name, " doesn't exist!")
    end
end

function analyze_value_txt(file_name, d = 0, N = 0)
    if d > 0 && N > 0
        file_name = string(file_name, "_", d, "_", N, ".txt")
    end
    try
        open(file_name, "r") do file
            unif = [1/N for i in 1:N]
            furth_distrib = [1/N for i in 1:N]
            println(furth_distrib)
            furth_dist = 0
            min_val = 1
            min_rho = Vector{Vector{Complex}}()
            result = 0
            len = 0
            for line in eachline(file)
                rho_str, objective_value_str, objective_solution_str, prior_str = rsplit(line, ":")
                objective_value = Meta.parse(objective_value_str)
                prior = Meta.parse(prior_str)
                # println(prior)

                # dist = norm(unif - prior)
                # if dist > furth_dist
                #     furth_dist = dist
                #     furth_distrib = prior
                # end
                if min_val > objective_value
                    min_val = objective_value
                    min_rho = Meta.parse(rho_str)
                end
                result += objective_value
                len += 1
            end
            return tuple(min_val, min_rho, round(result / len, digits = 4), furth_distrib)
        end
    catch e
        println("File ", file_name, " doesn't exist!")
    end
end

function analyze_prior_txt(file_name, d = 0, N = 0)
    if d > 0 && N > 0
        file_name = string(file_name, "_", d, "_", N, ".txt")
    end
    try
        open(file_name, "r") do file
            unif = Vector{Float64}([1/N for i in 1:N])
            furth_distrib = unif
            # println(LinearAlgebra.norm(unif - furth_distrib))
            furth_dist = 0
            min_val = 1
            min_rho = Vector{Vector{Complex}}()
            result = 0
            len = 0
            for line in eachline(file)
                rho_str, objective_value_str, objective_solution_str, prior_str = rsplit(line, ":")
                objective_value = eval(Meta.parse(objective_value_str))
                prior = eval(Meta.parse(prior_str))
                println(prior_str)
                println(prior)
                println(type(prior))
                println(Vector{Float64}(prior[1]))
                # println(LinearAlgebra.norm(Vector{Float64}(prior)))
                # println(LinearAlgebra.norm(unif - prior))

                # dist = norm(unif - prior)
                # if dist > furth_dist
                #     furth_dist = dist
                #     furth_distrib = prior
                # end
                if min_val > objective_value
                    min_val = objective_value
                    min_rho = eval(Meta.parse(rho_str))
                end
                result += objective_value
                len += 1
            end
            return tuple(min_val, min_rho, round(result / len, digits = 4), furth_distrib)
        end
    catch e
        println("File ", file_name, " doesn't exist!")
    end
end

function main()
    M = [[0.640000272025707 + 0.0im -0.1600006833360107 + 0.0im; -0.1600006833360107 + 0.0im 0.03999974346694657 + 0.0im], 
         [0.31999975198746244 + 0.0im 0.3200011567343735 + 0.0im; 0.3200011567343735 - 0.0im 0.3200008315665967 + 0.0im], 
         [0.03999997737555185 + 0.0im -0.16000047478529925 + 0.0im; -0.16000047478529925 + 0.0im 0.6399994273708338 + 0.0im]]
    rho1 = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [0 0 ; 0 1]]
    rho2 = [[1 0 ; 0 0], [0.5 0.5 ; 0.5 0.5], [1 1 ; 0 1]]
    obj_val = 1
    prior = [1/3, 1/3, 1/3]

    # write_txt("data/test", rho1, obj_val, M, prior)
    # write_txt("data/test", rho2, obj_val, M, prior)
    # cases = read_txt("data/test", 3, 2)
    # rho, objective_value, objective_solution, prior = cases[end]
    # objective_solution

    # min_val, min_rho, ave_val = analyze_value_txt("data/double_dual/mixed_2_2.txt")
    # min_val, min_rho, ave_val, distrib = analyze_prior_txt("data/dual/mixed", 2, 2)
    # min_rho = Distribs.rotate_states([min_rho])
    # ADD WORST DISTRIB !!!!

end

main()