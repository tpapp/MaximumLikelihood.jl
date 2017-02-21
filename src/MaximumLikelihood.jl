module MaximumLikelihood

using Optim
using PDMats
using ValidatedNumerics
using StatsFuns
using AutoAligns
import ForwardDiff

export estimate_ML

"""
Maximum likelihood estimator.

θ  ML estiamte
Σ  covariance matrix (estimated at the mode, using Fisher information)
varnames  variable names
optimization_result  contains convergence information from Optim.optimize
"""
immutable ML_estimator{T, ΣT <: AbstractPDMat, oT}
    θ::Vector{T}
    Σ::ΣT
    varnames::Vector{String}
    optimization_result::oT
end

"""
Default variable names, when not provided.
"""
default_varnames(n) = [string("θ", i) for i in 1:n]

"""
Estimate parameters using maximum likelihood, with the log likelihood
function `log_ℓ` and initial guess `initial_θ`.

Return an `ML_estimator`, with `varnames` optimally
specified. `method` is passed to `Optim.optimize`. `log_ℓ` needs to
work with ForwardDiff.
"""
function estimate_ML(log_ℓ, initial_θ;
                     method = BFGS(),
                     varnames = default_varnames(length(initial_θ)))
    o = optimize(θ -> -log_ℓ(θ), initial_θ,
                 method, Optim.Options(autodiff = true))
    Optim.converged(o) ||
        error("""Maximum likelihood did not converge.
Check concavity and initial value.""")
    θ = o.minimizer
    Σ = inv(PDMat(Symmetric(-ForwardDiff.hessian(log_ℓ, θ))))
    ML_estimator(θ, Σ, varnames, o)
end

"""
Univariate confidence intervals.
"""
function confidence_intervals(ml::ML_estimator, p=0.025)
    @assert p < 0.5
    s = norminvcdf(1-p)
    [θ ± σ*s for (θ, σ) in zip(ml.θ, diag(ml.Σ))]
end
                     
function Base.show(io::IO, ml::ML_estimator; p = 0.025)
    @assert p < 0.5
    aa = AutoAlign(align = Dict(1 => left, :default => right))
    p_string(p) = string(round(100*p, 1), "%")
    v_string(v) = @sprintf("%.5g", v) # significant digits hardcoded
    println(aa, "name", "  ", "est", "  ", p_string(p), "  ", p_string(1-p))
    for (varname, θ, ci) in zip(ml.varnames, ml.θ, confidence_intervals(ml, 0.025))
        println(aa, varname, "", v_string(θ), "", v_string(ci.lo), "", v_string(ci.hi))
    end
    print(io, aa)
end

end # module
