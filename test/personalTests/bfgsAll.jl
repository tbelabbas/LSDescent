using NLPModels, JuMP,  NLPModelsJuMP
using OptimizationProblems
using SolverCore, Logging
using LinearOperators
using LinearAlgebra

using Stopping
using OneDmin


include("../../src/BFGS/BFGS.jl")
include("bfgs_Stop_compare.jl")

#nlp = MathOptNLPModel(PureJuMP.dixmaank(40), name="dixmaank")
#nlp = MathOptNLPModel(PureJuMP.dixmaang(100), name="dixmaang")
nlp = MathOptNLPModel(PureJuMP.srosenbr(80), name="srosenbr")
#nlp = MathOptNLPModel(PureJuMP.woods(40), name="woods")
#nlp = MathOptNLPModel(PureJuMP.genrose(4), name="genrose")


n = nlp.meta.nvar
maxiter = 1500
scaling = false
mem = 400

stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.max_iter = maxiter
reinit!(stp)
reset!(nlp)


chB = ChBFGSOperator(Float64, n; scaling = scaling);
B = InverseBFGSOperator(Float64, n; scaling = scaling);
lB = InverseLBFGSOperator(Float64, n, mem=mem; scaling = scaling);
cB = CompactInverseBFGSOperator(Float64, n, mem=mem; scaling = scaling)

let chB=chB, B=B, lB=lB, cB=cB
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        iter, f, normg, B, x = bfgsA(nlp,  chB = chB, bB = B, B = lB, cB = cB, maxiter=maxiter)
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f ")
    end
end
# iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score