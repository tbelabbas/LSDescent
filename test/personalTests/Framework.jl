using Logging

using OneDmin
using Stopping

using LinearOperators

include("../../src/BFGS/BFGS.jl")


function test_algo(algo::Function, nlp, B, maxiter)
    reset!(nlp)

    stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
    stp.meta.max_iter = maxiter

    reinit!(stp)

    let stp = stp
        #logger = Logging.ConsoleLogger(stderr, Logging.Warn)
        logger = Logging.NullLogger()
        Logging.with_logger(logger) do
            stp = algo(nlp, stp=stp, B₀=B)
            #@show stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        end
    end

    return stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score, stp.meta.optimal
end


function evalmodel(nlp, m, maxiter, scaling, verbose)
    reset!(nlp)
    n = nlp.meta.nvar

    verbose && @info "Version encapsulée opérateur, formule O(n^2)"
    B = InverseBFGSOperator(Float64, n; scaling = scaling)
    iterB, fB, normgB, optimalB = test_algo(bfgs_StopLS, nlp, B, maxiter)
    verbose && println("| Nb iter : $iterB | Grad norm : $normgB | Fct value : $fB |")

    verbose && @info "Version L-BFGS"
    iB = InverseLBFGSOperator(Float64, n, mem=m; scaling = scaling)
    iteriB, fiB, normgiB, optimaliB = test_algo(bfgs_StopLS, nlp, iB, maxiter)
    verbose && println("| Nb iter : $iteriB | Grad norm : $normgiB | Fct value : $fiB |")

    verbose && @info "Version compact L-BFGS"
    cB = CompactInverseBFGSOperator(Float64, n, mem = m, scaling = scaling)
    itercB, fcB, normgcB,optimalcB = test_algo(bfgs_StopLS, nlp, cB, maxiter)
    verbose && println("| Nb iter : $itercB | Grad norm : $normgcB | Fct value : $fcB |")

    verbose && @info "Version Cholesky"
    cHB = ChBFGSOperator(Float64, n; scaling = scaling);
    itercHB, fcHB, normgcHB, optimalchB = test_algo(bfgs_StopLS, nlp, cHB, maxiter)
    verbose && println("| Nb iter : $itercHB | Grad norm : $normgcHB | Fct value : $fcHB |")

    verbose && println("_______________________________________________________________")

    [iterB normgB optimalB], [iteriB normgiB optimaliB], [itercB normgcB optimalcB], [itercB normgcHB optimalchB]
end


function evaluate_model(nlp, memory, maxiter, scaling, verbose = false)
    evalmodel(nlp, memory, maxiter, scaling, false)
    return evalmodel(nlp, memory, maxiter, scaling, verbose)
end