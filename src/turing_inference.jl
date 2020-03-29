using Turing: Tracker

function turing_inference(
    prob::DiffEqBase.DEProblem,
    alg,
    t,
    data,
    priors;
    likelihood_dist_priors = [InverseGamma(2, 3)],
    likelihood = (u,p,t,σ) -> MvNormal(u, σ[1]*ones(length(u))),
    num_samples=1000, sampler = Turing.NUTS(0.65),
    syms = [Turing.@varname(theta[i]) for i in 1:length(priors)],
    sample_u0 = false, 
    progress = true,
    solve_progress = false, 
    cb = θ -> false,
    kwargs...,
)
    N = length(priors)
    Turing.@model mf(x, ::Type{T} = Float64) where {T <: Real} = begin
        theta = Vector{T}(undef, length(priors))
        for i in 1:length(priors)
            theta[i] ~ NamedDist(priors[i], syms[i])
        end
        σ = Vector{T}(undef, length(likelihood_dist_priors))
        for i in 1:length(likelihood_dist_priors)
            σ[i] ~ likelihood_dist_priors[i]
        end
        nu = length(prob.u0)
        u0 = convert.(T, sample_u0 ? theta[1:nu] : prob.u0)
        p = convert.(T, sample_u0 ? theta[(nu + 1):end] : theta)
        _saveat = isnothing(t) ? Float64[] : t
        sol = concrete_solve(prob, alg, u0, p; saveat = _saveat, progress = solve_progress, kwargs...)
        failure = size(sol, 2) < length(_saveat)
        if failure
            @logpdf() = T(0) * sum(p) + T(-Inf)
            return
        end
        if ndims(sol) == 1
            x ~ likelihood(sol[:], theta, Inf, σ)
        else
            for i = 1:length(t)
                x[:, i] ~ likelihood(sol[:, i], theta, _saveat[i], σ)
            end
        end
        cb(Tracker.data.([u0; p; σ]))
        return
    end
    
    # Instantiate a Model object.
    model = mf(data)
    if num_samples > 0
        chn = sample(model, sampler, num_samples; progress = progress)
    else
        chn = Chains(zeros(0, 0, 0))
    end
    vi = Turing.VarInfo(model)
    backend = Turing.Core.getADbackend(sampler)
    function nlogp(θ)
        spl = Turing.SampleFromPrior()
        vi′ = Turing.VarInfo(vi, spl, θ)
        model(vi′, spl); cb(θ)
        -Turing.getlogp(vi′)
    end
    function ∇nlogp(θ)
        -Turing.gradient_logp(backend, θ, vi, model)[2]
    end
    return setinfo(chn, (model = model, nlogp = nlogp, ∇nlogp = ∇nlogp))
end