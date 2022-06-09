# AAA algorithm from the paper "The AAA Alorithm for Rational Approximation"
# by Y. Nakatsukasa, O. Sete, and L.N. Trefethen, SIAM Journal on Scientific Computing
# 2018 

"""
    AAAapprox{T <: AbstractArray} <: BRInterp

Struct for the [`aaa`](@ref). 

# Fields 
- `x`: Support points.
- `f`: Function values at support points. 
- `w`: Weights. 
- `errvec`: Vector of errors at each iteration.
"""
struct AAAapprox{T <: AbstractArray} <: BRInterp
    x::T
    f::T
    w::T
    errvec::T
end

(a::AAAapprox)(zz) = reval(zz, a.x, a.f, a.w)

# Handle function inputs as well
function aaa(Z::AbstractArray{T,1}, F::S;  tol=1e-13, mmax=100, verbose=false, clean=true) where {T, S<:Function}
    aaa(Z, F.(Z), tol=tol, mmax=mmax, verbose=verbose, clean=clean)
end

"""
    aaa(Z::AbstractArray{T, 1}, F::AbstractArray{S, 1}; <keyword arguments>)

Applies the AAA algorithm to the data `(Z, T)`.

# Arguments 
- `Z::AbstractArray{T,1}`: Vector of sample points.
- `F::AbstractArray{S, 1}`: Vector of data values at the corresponding points in `Z`.

# Keyword Arguments 
- `tol = 1e-13`: Relative tolerance.
- `mmax = 100`: Maximum type is `(mmax - 1, mmax - 1)`.
- `verbose = false`: Print info while calculating.
- `clean = true`: Detect and remove Froissart doublets.
- `cleanup_fnc = res -> findall(abs.(res) .< tol)`: A function which takes in a 
    vector of residues `res` and returns a vector of indices in `res` that should be cleaned.

# Outputs 
- `r`: An AAA approximant as a callable struct with fields.
"""
function aaa(Z::AbstractArray{T,1}, F::AbstractArray{S,1}; tol=1e-13, mmax=100,
             verbose=false, clean=true, cleanup_fnc = res -> findall(abs.(res) .< tol)) where {S, T}
    # filter out any NaN's or Inf's in the input
    keep = isfinite.(F)
    F = F[keep]
    Z = Z[keep]
    
    M = length(Z)                    # number of sample points
    mmax = min(M, mmax)              # max number of support points
    
    reltol = tol*norm(F, Inf)
    SF = spdiagm(M, M, 0 => F)       # left scaling matrix
    
    F, Z = promote(F, Z)
    P = promote_type(S, T)
    
    J = [1:M;]
    z = P[]                          # support points
    f = P[]                          # function values at support points
    C = P[]
    w = P[]
    
    errvec = P[]
    R = F .- mean(F)
    @inbounds for m = 1:mmax
        j = argmax(abs.(F .- R))               # select next support point
        push!(z, Z[j])
        push!(f, F[j])
        deleteat!(J, findfirst(isequal(j), J))   # update index vector

        # next column of Cauchy matrix
        C = isempty(C) ? reshape((1 ./ (Z .- Z[j])), (M,1)) : [C (1 ./ (Z .- Z[j]))]

        Sf = diagm(f)                         # right scaling matrix
        A = SF * C - C * Sf                   # Loewner matrix
        G = svd(A[J, :])
        
        # A[J, :] might not have full rank, so svd can return fewer than m columns
        #w = G.V[:, m]                        
        w = G.V[:, end]                       # weight vector = min sing vector
        
        N = C * (w .* f)                      # numerator 
        D = C * w
        R .= F
        R[J] .= N[J] ./ D[J]                  # rational approximation
        
        err = norm(F - R, Inf)
        verbose && println("Iteration: ", m, "  err: ", err)
        errvec = [errvec; err]                # max error at sample points
        err <= reltol && break                # stop if converged
    end
    r = AAAapprox(z, f, w, errvec)

    # remove Frois. doublets if desired.  We do this in place
    if clean
        pol, res, zer = prz(r)            # poles, residues, and zeros
        ii = cleanup_fnc(res)  # find negligible residues
        length(ii) != 0 && cleanup!(r, pol, res, zer, Z, F, cleanup_fnc)
    end
    return r
end

"""
    prz(r::AAAapprox)

Returns the poles, residues, and zeros for the AAA approximant `r`.
"""
function prz(r::AAAapprox)
    z, f, w = r.x, r.f, r.w        
    m = length(w)
    B = diagm(ones(m+1))
    B[1, 1] = 0.0
    E = [0.0  transpose(w); ones(m) diagm(z)]
    pol, _ = eigen(E, B)
    pol = pol[isfinite.(pol)] 
    dz = 1e-5 * exp.(2im*pi*[1:4;]/4)
    
    # residues
    res = r(pol .+ transpose(dz)) * dz ./ 4 
        
    E = [0 transpose(w .* f); ones(m) diagm(z)]
    zer, _ = eigen(E, B)
    zer = zer[isfinite.(zer)]
    pol, res, zer
end

"""
    reval(zz, z, f, w)

Evaluates the AAA approximant at `zz`.
"""
function reval(zz, z, f, w)
    # evaluate r at zz
    zv = size(zz) == () ? [zz] : vec(zz)  
    CC = 1.0 ./ (zv .- transpose(z))           # Cauchy matrix
    r = (CC * (w .* f)) ./ (CC * w)            # AAA approx as vector
    r[isinf.(zv)] .= sum(f .* w) ./ sum(w)
    
    ii = findall(isnan.(r))                    # find values NaN = Inf/Inf if any
    @inbounds for j in ii
        if !isnan(zv[j]) && ((v = findfirst(isequal(zv[j]), z)) !== nothing)
            r[j] = f[v]  # force interpolation there
        end
    end
    r = size(zz) == () ? r[1] : reshape(r, size(zz))             # the AAA approximation
end


# Only calculate the updated z, f, and w
"""
    cleanup!(r, pol, res, zer, Z, F, cleanup_fnc)

Cleans up the Froissart doublets in the AAA approximant `r`. Operates in-place.

# Arguments 
- `r`: The AAA approximant.
- `pol`: Poles of `r`.
- `res`: Residues for the poles in `pol`.
- `zer`: Zeros of `r`.
- `Z`: The sample points. 
-` F`: The data values at the corresponding points in `Z`.
- `cleanup_fnc`: A function which takes in a 
    vector of residues `res` and returns a vector of indices in `res` that should be cleaned.
"""
function cleanup!(r, pol, res, zer, Z, F, cleanup_fnc)
    z, f, w = r.x, r.f, r.w
    m = length(z)
    M = length(Z)
    ii = cleanup_fnc(res) # find negligible residues
    ni = length(ii)
    ni == 0 && return
    println("$ni Froissart doublets. Number of residues = ", length(res))

    @inbounds for j = 1:ni
        azp = abs.(z .- pol[ii[j]] )
        jj = findall(isequal(minimum(azp)), azp)
        deleteat!(z, jj)    # remove nearest support points
        deleteat!(f, jj)
    end    

    @inbounds for j = 1:length(z)
        jj = findall(isequal(z[j]), Z)
        deleteat!(F, jj)
        deleteat!(Z, jj)
    end
    m = m - length(ii)
    SF = spdiagm(M-m, M-m, 0 => F)
    Sf = diagm(f)
    C = 1 ./ (Z .- transpose(z))
    A = SF*C - C*Sf
    G = svd(A)
    w[:] .= G.V[:, m]
    println("cleanup: ", size(z), "  ", size(f), "  ", size(w))
    return nothing
end
