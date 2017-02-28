using MaximumLikelihood
using Base.Test
using Distributions

@testset "multivariate normal likelihood" begin
    d = MultivariateNormal([1,2], 3)
    ℓ(θ) = logpdf(d, θ)
    ML = MaximumLikelihood.estimate_ML( ℓ, zeros(2))
    
    @test ML.θ ≈ [1.0, 2.0]
    @test full(ML.Σ) ≈ diagm([9.0, 9.0])
end
