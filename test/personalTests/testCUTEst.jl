using CUTEst
using Plots
using Logging
using NLPModels
using StatsPlots
using Distributed
using LinearOperators
using OptimizationProblems

include("../../src/BFGS/BFGS.jl")



# function test_algo(algo::Function, nlp, B, maxiter)
#     reset!(nlp)

#     #logger = Logging.ConsoleLogger(stderr, Logging.Warn)
#     #Logging.with_logger(logger) do
#     iter, f, normg, B, x = algo(nlp, B₀ = B, maxiter = maxiter)
#     ##@show iter, f, normg
#     #end

#     return iter, f, normg, B, x
# end

function test_algo(algo::Function, nlp, B, maxiter)
    reset!(nlp)

    stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
    stp.meta.max_iter = maxiter

    reinit!(stp)

    let stp = stp
        logger = Logging.ConsoleLogger(stderr, Logging.Warn)
        Logging.with_logger(logger) do
            stp = algo(nlp, stp=stp, B₀=B)
            #@show stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        end
    end

    return stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
end

function evalmodel_nolog(nlp, m, maxiter, scaling)
    n = nlp.meta.nvar

    B = InverseBFGSOperator(Float64, n; scaling = scaling)
    test_algo(bfgs_StopLS, nlp, B, maxiter)

    iB = InverseLBFGSOperator(Float64, n, mem=m; scaling = scaling)
    test_algo(bfgs_StopLS, nlp, iB, maxiter)

    cB = CompactInverseBFGSOperator(Float64, n, mem = m, scaling = scaling)
    test_algo(bfgs_StopLS, nlp, cB, maxiter)

end

function evalmodel(nlp, m, maxiter, scaling)
    n = nlp.meta.nvar

    @info "Version encapsulée opérateur, formule O(n^2)"
    B = InverseBFGSOperator(Float64, n; scaling = scaling)
    iterB, fB, normgB = test_algo(bfgs_StopLS, nlp, B, maxiter)
    println("| Nb iter : $iterB | Grad norm : $normgB | Fct value : $fB |")

    @info "Version L-BFGS"
    iB = InverseLBFGSOperator(Float64, n, mem=m; scaling = scaling)
    iteriB, fiB, normgiB = test_algo(bfgs_StopLS, nlp, iB, maxiter)
    println("| Nb iter : $iteriB | Grad norm : $normgiB | Fct value : $fiB |")

    @info "Version compact L-BFGS"
    cB = CompactInverseBFGSOperator(Float64, n, mem = m, scaling = scaling)
    itercB, fcB, normgcB = test_algo(bfgs_StopLS, nlp, cB, maxiter)
    println("| Nb iter : $itercB | Grad norm : $normgcB | Fct value : $fcB |")

    @info "Version Cholesky"
    cHB = ChBFGSOperator(Float64, n; scaling = scaling);
    itercHB, fcHB, normgcHB = test_algo(bfgs_StopLS, nlp, cHB, maxiter)
    println("| Nb iter : $itercHB | Grad norm : $normgcHB | Fct value : $fcHB |")

    println("_______________________________________________________________")

    [iterB normgB], [iteriB normgiB], [itercB normgcB], [itercB normgcHB]
end


memory = 460
maxiter = 3000
scaling = false

# pas avec contraintes
probs = CUTEst.select(min_var=1, max_var=800, contype="unc")

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
        evalmodel_nolog(nlp, memory, maxiter, scaling)
        reset!(nlp)
        b, l, c, d = evalmodel(nlp, memory, maxiter, scaling)
        #println([b prob])
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


using DataFrames
println("\n ITERATION COMPARISON")
df_ITER = DataFrame(PROBLEMS = b_res[3:3:end],
                BFGS = b_res[1:3:end],
                LBFGS = l_res[1:2:end],
                LBFGS_C = c_res[1:2:end],
                CHOLESKY = d_res[1:2:end]
                )
print(df_ITER)

println("\n GRADIENT NORM COMPARISON")
df_GRAD_Norm = DataFrame(PROBLEMS = b_res[3:3:end],
                            BFGS = b_res[2:3:end],
                            LBFGS = l_res[2:2:end],
                            LBFGS_C = c_res[2:2:end],
                            CHOLESKY = d_res[2:2:end]
                        )
print(df_GRAD_Norm)

println("\n Comparison between LBFGS, LBFGS_C and CHOLESKY")
df_gen = DataFrame(PROBLEMS = b_res[3:3:end],
                   iter_BFGS = b_res[1:3:end],
                   iter_LBFGS = l_res[1:2:end],
                   iter_LBFGS_C = c_res[1:2:end],
                   iter_CHOLESKY = d_res[1:2:end],
                   norm_BFGS = b_res[2:3:end],
                   norm_LBFGS = l_res[2:2:end],
                   norm_LBFGS_C = c_res[2:2:end],
                   norm_CHOLESKY = d_res[2:2:end]
                  )
print(df_gen)
