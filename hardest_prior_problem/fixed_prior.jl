#=
fixed_prior:
- Julia version: 1.10.5
- Author: ChernyshovaP
- Date: 2025-12-08
=#

module SDPModels
    include("sdp_models.jl")
end
module Measures
    include("measures.jl")
end
module Fio
    include("fio.jl")
end


function simulator(filename, N, d)
    filename_read = string("data/double_dual/", filename)
    filename_write = string("data/primal/", filename)

    cases = Fio.read_txt(filename_read, N, d)

    for i in 1:size(cases)[1]
        rho, objective_value, objective_solution, prior = cases[i]
        obj_value, obj_solution = SDPModels.primal_model(N, d, rho)
        # Fio.write_txt(filename_write, rho, obj_value, obj_solution, [1/N for i in 1:N])
    end

    return
end

function main()
    N = 2
    d = 10
    
    # simulator("mixed", N, d)
    # simulator("pure", N, d)

    Fio.analyze_value_txt("data/double_dual/pure", N, d) / Fio.analyze_value_txt("data/primal/pure", N, d)
end

main()