using CUTEst

using LinearAlgebra

using NLPModels
using OptimizationProblems

using Plots
using Logging
using StatsPlots
using Distributed
using DataFrames
using BenchmarkProfiles

include("Framework.jl")

memory = 460
maxiter = 3000
scaling = false

probs = CUTEst.select(min_var=1, max_var=1, contype="unc")
probs = ["MUONSINELS", "MUONSINELS"]

global b_res = []
global c_res = []
global l_res = []
global d_res = []

global i = 1

for prob in probs
    if prob == "BLEACHNG"
        continue
    end

    global nlp = CUTEstModel(prob)

    println("_______________________________________________________________")
    println("$i/$(length(probs)) Currently working on problem number $i : ", prob)

    try
        b, l, c, d = evaluate_model(nlp, memory, maxiter, scaling)
        append!(b_res, [b prob])
        append!(l_res, l)
        append!(c_res, c)
        append!(d_res, d)
    catch e
        println(e)
        println("WARNING : Skipped problem : ", prob)
    end

    global i += 1
    finalize(nlp)

end
println("_______________________________________________________________")


prob_list = b_res[4:4:end]
nb_prob = length(prob_list)
deleteat!(b_res,4:4:20)

println("\n ITERATION COMPARISON")
df_ITER = DataFrame(PROBLEMS = prob_list,
                BFGS = b_res[1:3:end],
                LBFGS = l_res[1:3:end],
                LBFGS_C = c_res[1:3:end],
                CHOLESKY = d_res[1:3:end]
                )
println(df_ITER)

println("\n GRADIENT NORM COMPARISON")
df_GRAD_Norm = DataFrame(PROBLEMS = prob_list,
                            BFGS = b_res[2:3:end],
                            LBFGS = l_res[2:3:end],
                            LBFGS_C = c_res[2:3:end],
                            CHOLESKY = d_res[2:3:end]
                        )
println(df_GRAD_Norm)

println("\n GRADIENT NORM COMPARISON")
df_OPTIMAL = DataFrame(PROBLEMS = prob_list,
                       BFGS     = !iszero(b_res[3:3:end]),
                       LBFGS    = !iszero(l_res[3:3:end]),
                       LBFGS_C  = !iszero(c_res[3:3:end]),
                       CHOLESKY = !iszero(d_res[3:3:end])
                       )
println(df_OPTIMAL)

println("\n Comparison between LBFGS, LBFGS_C and CHOLESKY")
df_gen = DataFrame(PROBLEMS = prob_list,
                   iter_BFGS = b_res[1:3:end],
                   iter_LBFGS = l_res[1:3:end],
                   iter_LBFGS_C = c_res[1:3:end],
                   iter_CHOLESKY = d_res[1:3:end],
                   norm_BFGS = b_res[2:3:end],
                   norm_LBFGS = l_res[2:3:end],
                   norm_LBFGS_C = c_res[2:3:end],
                   norm_CHOLESKY = d_res[2:3:end],
                   optimal_BFGS     = !iszero(b_res[3:3:end]),
                   optimal_LBFGS    = !iszero(l_res[3:3:end]),
                   optimal_LBFGS_C  = !iszero(c_res[3:3:end]),
                   optimal_CHOLESKY = !iszero(d_res[3:3:end])
                  )
println(df_gen)


#solvers = OrderedDict{Symbol,Function}(
solvers = Dict{Symbol,Any}(
        :BFGS    => InverseBFGSOperator,
        :L_BFGS  => InverseLBFGSOperator,
        :C_BFGS  => CompactInverseBFGSOperator,
        :CH_BFGS => ChBFGSOperator
    )

col_names = ["NB_ITER", "NORM_G", "F"]
bdf = DataFrame(reshape(b_res, 3, nb_prob)', col_names)
ldf = DataFrame(reshape(l_res, 3, nb_prob)', col_names)
cdf = DataFrame(reshape(c_res, 3, nb_prob)', col_names)
ddf = DataFrame(reshape(d_res, 3, nb_prob)', col_names)


stats = OrderedDict{Symbol, DataFrame}()

stats[BFGS   ] = bdf
stats[L_BFGS ] = bdf
stats[C_BFGS ] = bdf
stats[CH_BFGS] = bdf
