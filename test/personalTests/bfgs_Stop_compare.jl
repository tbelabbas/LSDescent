using Plots, Colors

function bfgs_compare(nlp       :: AbstractNLPModel{T, S};
                     x         :: S = copy(nlp.meta.x0),
                     stp       :: NLPStopping = NLPStopping(nlp,
                                                      NLPAtX(nlp.meta.x0)),
                     scaling   :: Bool = true,
                     LS_algo   :: Function = bracket_s2N,
                     LS_logger :: AbstractLogger = Logging.NullLogger(),
                     chB ,
                     bB ,
                     lB ,
                     cB ,
                     kwargs...      # eventually options for the line search
                     ) where {T, S}

    n = length(x)
    f = obj(nlp,x)
    ∇f = grad(nlp, x)

    xt = similar(x)
    ∇ft = similar(∇f)

    stp.stopping_user_struct["BFGS"] = lB

    τ₀ = 0.0005
    τ₁ = 0.9999

#    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)
    #update_and_stop!(stp,  x = x, fx = f, gx = ∇f)
    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)

    while !OK
        chd = - chB*∇ft
        bd = - bB*∇ft
        d = - lB*∇ft
        cd = - cB*∇ft

        display(chd)
        display(bd)
        display(d)
        display(cd)
        # Simple line search call
        # returns  t, xt = x + t*d,  ft=f(xt), ∇ft = ∇f(xt)
        reset!(ϕ)
        t, xt, ft, ∇ft = linesearch(ϕ, ϕstp, x, d, f, ∇f, τ₀, τ₁, logger = LS_logger, algo = LS_algo; kwargs...)
        fail_sub_pb = ~ϕstp.meta.optimal

        if fail_sub_pb
            OK = true
            stp.meta.fail_sub_pb = true
        #end
        # try anyway the step
        else
            # Update BFGS approximation.
            chB = push!(chB, t * chd, ∇ft - ∇f)
            bB = push!(bB, t * bd, ∇ft - ∇f)
            lB = push!(lB, t * d, ∇ft - ∇f)
            cB = push!(cB, t * cd, ∇ft - ∇f)
            #B = push!(B, t * d, ∇ft - ∇f, scaling)

            #move on
            x .= xt
            f = ft
            ∇f .= ∇ft

            OK = update_and_stop!(stp, x = x, gx = ∇f, fx = f)

            gr()
            h = plot(Gray.(Matrix(lB)-Matrix(cB)))
            png(".\\plot")

        end
        norm∇f = stp.current_state.current_score
        #@show f
    end

    if !stp.meta.optimal
        @warn status(stp,list=true)
    end

    #return tuple(stp, B)
    return stp
end


function bfgsA(nlp     :: AbstractNLPModel;
    x       :: Vector{T}=copy(nlp.meta.x0),
    ϵ       :: T = 1e-6,
    rtol    :: T = 1e-15,
    maxiter :: Int = 200,
    scaling :: Bool = true,
    Lp      :: Real = 2, # norm Lp
    B, cB, bB, chB
    ) where T

    f = obj(nlp,x)
    ∇f = grad(nlp, x)
    norm₀ = norm(∇f, Lp)
    xt = similar(x)
    ∇ft = similar(∇f)

    τ₀ = 0.0005
    τ₁ = 0.9999

    iter = 0

    while (norm(∇f, Lp) > ϵ) && (norm(∇f, Lp) > rtol*norm₀) && (iter <= maxiter)
        d = - B*∇f
        cd = - cB*∇f
        bd = - bB*∇f
        chd = - chB*∇f

        display(norm(d - cd))

        #------------------------------------------
        # Hard coded line search
        hp0 = ∇f'*d
        t=1.0
        # Simple Wolfe forward tracking
        xt = x + t*d
        ∇ft = grad(nlp,xt)
        hp = ∇ft'*d
        ft = obj(nlp, xt)
        #  while  ~wolfe & armijo
        nbW = 0
        while (hp <= τ₁ * hp0) && (ft <= ( f + τ₀*t*hp0)) && (nbW < 10)
            t *= 5
            xt = x + t*d
            ∇ft = grad(nlp,xt)
            hp = ∇ft'*d
            ft = obj(nlp, xt)
            nbW += 1
        end
        tw = t

        # Simple Armijo backtracking
        nbk = 0
        while (ft > ( f + τ₀*t*hp0)) && (nbk < 20)
            t *= 0.5 # 0.4
            xt = x + t*d
            ft = obj(nlp, xt)
            nbk += 1
        end
        #------------------------------------------

        if t!=tw   ∇ft = grad(nlp, xt) end

        # Update BFGS approximation.
        B = push!(B, t * d, ∇ft - ∇f)
        cB = push!(cB, t * cd, ∇ft - ∇f)
        bB = push!(bB, t * bd, ∇ft - ∇f)
        chB = push!(chB, t * chd, ∇ft - ∇f)
        #B = push!(B, t * d, ∇ft - ∇f, scaling)

        gr()
        h = plot(Gray.(Matrix(B)-Matrix(cB)))
        png("compact_lbfgs_iter_$(iter)")

        h = plot(Gray.(Matrix(B)-Matrix(bB)))
        png("bfgs_lbfgs_iter_$(iter)")

        h = plot(Gray.(Matrix(B)-Matrix(chB)))
        png("cholesky_lbfgs_iter_$(iter)")

        h = plot(Gray.(Matrix(B)-Matrix(B)))
        png("lbfgs_lbfgs_iter_$(iter)")
        #move on
        x .= xt
        f = ft
        ∇f .= ∇ft

        iter += 1
    end

    if iter > maxiter
        @warn "Maximum d'itérations"
    end

    return iter, f, norm(∇f, Lp), B, x
end





