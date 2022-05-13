using LinearAlgebra

using JuMP
using Logging
using NLPModels
using DataFrames
using NLPModelsJuMP
using QuadraticModels


include("genQuad.jl")
include("Framework.jl")

n = 25
memory  = 400
maxiter = 400
scaling = false

nb_probs = 150

nb_conds = 5
min_cond = 1
max_cond = 10

conds = []
for i in range(min_cond, max_cond, length=4)
    append!(conds, 10^i)
end


m = length(conds)
global BFGS_prct    = zeros(m, 3)
global L_BFGS_prct  = zeros(m, 3)
global CL_BFGS_prct = zeros(m, 3)
global CH_BFGS_prct = zeros(m, 3)

global j = 1
for λ_option in 1:500:1001
    global i = 1
    for cond in conds
        @info "Working with : Conditioning : $cond | Minimal λ : $(λ_option)"

        bp = lp = cp = chp = 0.0
        for prob in 1:nb_probs

            H, c, xopt = genQuad(n, cond, λ_option)
            nlp = QuadraticModel(c, H)
            reset!(nlp)

            global b = 0
            global l = 0
            global c = 0
            global ch = 0

            logger = Logging.NullLogger()
            Logging.with_logger(logger) do
                b, l, c, ch = evaluate_model(nlp, memory, maxiter, scaling)
            end

            # if optimal, increment sum of optimal problems
            !iszero(b[3])  && (bp += 1)
            !iszero(l[3])  && (lp += 1)
            !iszero(c[3])  && (cp += 1)
            !iszero(ch[3]) && (chp += 1)
        end

        BFGS_prct[i, j] = bp * 100 / nb_probs
        L_BFGS_prct[i, j] = lp * 100 / nb_probs
        CL_BFGS_prct[i, j] = cp * 100 / nb_probs
        CH_BFGS_prct[i, j] = chp * 100 / nb_probs
        global i+=1
    end
    global j+=1
end


# @info ""
# display(hcat(conds, BFGS_prct))
# display(hcat(conds, L_BFGS_prct))
# display(hcat(conds, CL_BFGS_prct))
# display(hcat(conds, CH_BFGS_prct))

df_λ_1 = DataFrame(Conditioning=conds,
                   BFGS_λ_1=BFGS_prct[:,1],
                   LBFGS_λ_1=L_BFGS_prct[:,1],
                   CBFGS_λ_1=CL_BFGS_prct[:,1],
                   ChBFGS_λ_1=CH_BFGS_prct[:,1]
                  )


df_λ_2 = DataFrame(Conditioning=conds,
                   BFGS_λ_2=BFGS_prct[:,2],
                   LBFGS_λ_2=L_BFGS_prct[:,2],
                   CBFGS_λ_2=CL_BFGS_prct[:,2],
                   ChBFGS_λ_2=CH_BFGS_prct[:,2]
                  )

df_λ_3 = DataFrame(Conditioning=conds,
                   BFGS_λ_3=BFGS_prct[:,3],
                   LBFGS_λ_3=L_BFGS_prct[:,3],
                   CBFGS_λ_3=CL_BFGS_prct[:,3],
                   ChBFGS_λ_3=CH_BFGS_prct[:,3]
                  )


println(df_λ_1)
println(df_λ_2)
println(df_λ_3)
;