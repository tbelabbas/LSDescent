"""generates a quadratic convex objective q(x) with prescribed condition number and lowest eigenvalue of the matrix H.

 q(x) = ½ x'Hx + c'x

genQuad(n, condition, lmin = 0.1)

   - n = dimension
   - condition = condition number of matrix H
   - lmin = smallest eigenvalue of matrix H

returns H, c, and the solution xsol """
function genQuad(n, condition, lmin = 0.1)

    Λ = rand(n)
    range = maximum(Λ) - minimum(Λ)
    Λ = lmin*((condition-1)/range)*(Λ .- minimum(Λ)) .+ lmin

    M = rand(n,n)
    Q, = qr(M)

    H = Q*diagm(Λ)*Q'
    l,v = eigen(H)

    xsol = rand(n)
    c = -H*xsol

    return H, c, xsol
end

