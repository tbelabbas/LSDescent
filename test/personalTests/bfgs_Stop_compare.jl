export bfgs_Stop, L_bfgs_Stop, M_bfgs_Stop, Ch_bfgs_Stop, C_bfgs_Stop


function bfgsA(nlp :: AbstractNLPModel,
                   All_B;
                   x :: Vector{T}=copy(nlp.meta.x0),
                   stp :: NLPStopping = NLPStopping(nlp,
                                                    NLPAtX(nlp.meta.x0)),
                   scaling :: Bool = true,
                   B₀ :: Union{AbstractLinearOperator,
                          AbstractMatrix,
                          UniformScaling{T},
                          Nothing}              = nothing,
                    maxiter = 1600
                   ) where T


    stp.meta.max_iter = maxiter
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f = obj(nlp,x)
    ∇f = grad(nlp, x)
    n = length(x)

    xt = similar(x)
    ∇ft = similar(∇f)

    B = B₀
    stp.stopping_user_struct["BFGS"] = B

    @show B.data.scaling

    τ₀ = 0.0005
    τ₁ = 0.9999

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)
    @info log_row(Any[0, f, norm(∇f)])

    while !OK
        d = - B*∇f

        #------------------------------------------
        # Hard coded line search
        hp0 = ∇f'*d
        t=1.0
        # Simple Wolfe forward tracking
        xt = x + t*d
        ft = obj(nlp, xt)
        if ft <= (f + τ₀*hp0)
            ∇ft = grad(nlp,xt)
            hp = ∇ft'*d
            #  while  ~wolfe & armijo
            nbW = 0
            while (hp <= τ₁ * hp0) && (ft <= (f + τ₀*t*hp0)) && (nbW < 10)
                t *= 5
                xt = x + t*d
                ∇ft = grad(nlp,xt)
                hp = ∇ft'*d
                ft = obj(nlp, xt)
                nbW += 1
                @debug "W", ft
            end
        end
        tw = t

        # Simple Armijo backtracking
        nbk = 0
        while ft > ( f + τ₀*t*hp0) && (nbk < 52)
            t *= 0.5 #0.4
            xt = x + t*d
            ft = obj(nlp, xt)
            nbk += 1
            @debug "A", ft
        end
        #------------------------------------------

        if nbk >= 51 # too small stepsize, abort
            OK = true
            stp.meta.fail_sub_pb = true
        else
            if t!=tw   ∇ft = grad(nlp, xt) end

            # Update BFGS approximation.
            B = push!(B, t * d, ∇ft - ∇f)

            #move on
            x .= xt
            f = ft
            ∇f .= ∇ft

            OK = update_and_stop!(stp, x = x, gx = ∇f, fx=f)

            push!(All_B, Matrix(B))
        end

        #@show f
        @info log_row(Any[stp.meta.nb_of_stop, f, norm(∇f), t, hp0])
    end

    if !stp.meta.optimal
        #@warn "Optimalité non atteinte"
        @warn status(stp,list=true)
    end

    return stp, All_B
end
