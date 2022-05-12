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
#nlp = MathOptNLPModel(PureJuMP.srosenbr(80), name="srosenbr")
nlp = MathOptNLPModel(PureJuMP.woods(80), name="woods")
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

All_B = []
All_Bc = []
All_Bb = []
All_Bch = []

@info "Version L-BFGS"
let chB=chB, B=B, lB=lB, cB=cB, All_B=All_B, stp=stp
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        All_B = bfgsA(nlp, All_B, stp=stp, B₀=lB, maxiter=maxiter)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f ")
    end
end

reinit!(stp)
reset!(nlp)

@info "Version L-BFGS compact"
let cB=cB, All_Bc=All_Bc, stp=stp
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        All_Bc = bfgsA(nlp, All_Bc, stp=stp, B₀=cB, maxiter=maxiter)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f ")
    end
end

reinit!(stp)
reset!(nlp)

@info "Version BFGS"
let B=B, All_Bb=All_Bb, stp=stp
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        All_Bb = bfgsA(nlp, All_Bb, stp=stp, B₀=B, maxiter=maxiter)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f ")
    end
end

reinit!(stp)
reset!(nlp)

@info "Version Cholesky"
let chB=chB, All_Bch=All_Bch, stp=stp
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        All_Bch = bfgsA(nlp, All_Bch, stp=stp, B₀=chB, maxiter=maxiter)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        println("| Nb iter : $iter | Grad norm : $normg | Fct value : $f ")
    end
end
# iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score

j = length(All_B)

for i in 1:min(j, length(All_Bb), length(All_Bc), length(All_Bch))
    h = plot(Gray.(Matrix(All_B[i])-Matrix(All_Bc[i])))
    png("compact_lbfgs_BEFORE_iter$(i)")

    h = plot(Gray.(Matrix(All_B[i])-Matrix(All_Bb[i])))
    png("bfgs_lbfgs_BEFORE_iter_$(i)")

    h = plot(Gray.(Matrix(All_B[i])-Matrix(All_Bch[i])))
    png("cholesky_lbfgs_BEFORE_iter_$(i)")

    h = plot(Gray.(Matrix(All_B[i])-Matrix(All_B[i])))
    png("lbfgs_lbfgs_BEFORE_iter_$(i)")
end

for i in j:length(All_Bb)
    h = plot(Gray.(Matrix(All_B[j])-Matrix(All_Bb[i])))
    png("AFTER_bfgs_lbfgs_iter_$(i)")
end

for i in j:length(All_Bch)
    h = plot(Gray.(Matrix(All_B[j])-Matrix(All_Bch[i])))
    png("AFTER_cholesky_lbfgs_iter_$(i)")
end