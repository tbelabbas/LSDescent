
using Pkg
Pkg.activate(".")

# Ce script utilise plutôt des fichiers locaux...
#using LSDescent

using NLPModels, JuMP,  NLPModelsJuMP
using SolverCore, Logging
using LinearAlgebra
using LinearOperators


function test_algo_stp(algo::Function, nlp, stp, B )
    reset!(nlp)

    #logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        stp = algo(nlp, stp=stp,  B₀ = B)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f |")
    end

end




# test all solvers with a well known test function
using OptimizationProblems


#include("woods.jl")
#include("genrose.jl")
#nlp = MathOptNLPModel(PureJuMP.dixmaank(40), name="dixmaank")
#nlp = MathOptNLPModel(PureJuMP.dixmaang(100), name="dixmaang")
#nlp = MathOptNLPModel(PureJuMP.srosenbr(80), name="srosenbr")
nlp = MathOptNLPModel(PureJuMP.woods(n=4), name="woods")
#nlp = MathOptNLPModel(PureJuMP.genrose(80), name="genrose")


n = nlp.meta.nvar
println(n)

include("../../src/BFGS/Type.jl")
include("../../src/BFGS/TypeCompact.jl")
include("../../src/BFGS/AcceptAll.jl")

maxiter = 2000
#scaling = false
scaling = true

println()
using Stopping
include("../../src/BFGS/bfgsStop.jl")

using OneDmin
include("../../src/BFGS/bfgsStopLS.jl")

stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.max_iter = maxiter


reinit!(stp)

mem = 500

@info "Version BFGS"

reinit!(stp)
B = InverseBFGSOperator(Float64, n; scaling = scaling)
test_algo_stp(bfgs_StopLS, nlp, stp, B)


@info "Version L-BFGS"

reinit!(stp)
B = InverseLBFGSOperator(Float64, n, mem=mem; scaling = scaling)
test_algo_stp(bfgs_StopLS, nlp, stp, B)

@info "Version Compact L-BFGS"

reinit!(stp)
B = CompactInverseBFGSOperator(Float64, n, mem=mem; scaling = scaling)
test_algo_stp(bfgs_StopLS, nlp, stp, B)


@info "Version CH-BFGS"

reinit!(stp)
B = ChBFGSOperator(Float64, n; scaling = scaling)
test_algo_stp(bfgs_StopLS, nlp, stp, B)

;
