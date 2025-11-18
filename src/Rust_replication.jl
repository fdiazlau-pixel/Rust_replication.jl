module Rust_replication
#############################
# rust1987_nfxp_fast.jl — Rust (1987) Step 2 (NFXP), fast version
#############################

using CSV, DataFrames, LinearAlgebra, Optim, Printf, Logging
# hello xdxdxdxdxdx
# ---------------------------------
# Model container & builder (compat)
# ---------------------------------
export RustModel, build_model, prepare_panel, estimate, run_rust1987, nll_at_rust

struct RustModel
    N::Int                   # mileage bins (e.g. 90)
    β::Float64               # discount factor
    Δ::Float64               # bin width in miles (5000.0)
    p0::Float64              # Pr(Δ=0 bins)
    p1::Float64              # Pr(Δ=1 bin)
    p2::Float64              # Pr(Δ≥2 bins)
    xgrid::Vector{Float64}   # 0:Δ:(N-1)*Δ (miles since last replacement)
    Pkeep::Matrix{Float64}   # N×N transition for keep (kept for compatibility)
    Prep::Matrix{Float64}    # N×N transition for replace (kept for compatibility)
end

function build_model(; N::Int=90, β::Float64=0.9999, Δ::Float64=5000.0,
                        θ30::Float64=0.3919, θ31::Float64=0.5953)
    p0, p1 = θ30, θ31
    p2 = max(0.0, 1.0 - p0 - p1) # guard tiny negatives
    xgrid = collect(0.0:Δ:Δ*(N-1))

    # Dense matrices are kept for API compatibility, but the fast solver doesn't use them.
    Pkeep = zeros(N, N)
    for x in 1:N
        Pkeep[x, min(x,   N)] += p0
        Pkeep[x, min(x+1, N)] += p1
        Pkeep[x, min(x+2, N)] += p2
    end

    Prep = zeros(N, N)
    Prep[:, 1] .= p0
    if N ≥ 2; Prep[:, 2] .= p1; end
    if N ≥ 3; Prep[:, 3] .= p2; end

    return RustModel(N, β, Δ, p0, p1, p2, xgrid, Pkeep, Prep)
end

# --------------------------
# Preferences and utilities
# --------------------------
# IMPORTANT: your state grid is in MILES.
# To match Rust’s linear spec in comparable utility units, use:
#   c(x_miles, θ11) = 1e-6 * θ11 * x_miles
@inline maintenance_cost(x_miles::Float64, θ11::Float64) = 1e-6 * θ11 * x_miles

# Deterministic utility u(x,i): i=0 keep, i=1 replace (kept for completeness)
@inline function u(model::RustModel, idx::Int, i::Int, θ11::Float64, RC::Float64)
    x = model.xgrid[idx]
    return (i == 1) ? (-RC - maintenance_cost(0.0, θ11)) : (-maintenance_cost(x, θ11))
end

# --------------------------
# Numerically stable helpers
# --------------------------
# log(exp(a)+exp(b)) using “subtract max” + log1p
@inline function logsumexp2(a::Float64, b::Float64)
    m = ifelse(a > b, a, b)
    return m + log1p(exp(-abs(a - b)))
end

# Logistic (stable for large |z|)
@inline function safe_logistic(z::Float64)
    z ≥ 0 ? 1.0/(1.0 + exp(-z)) : exp(z)/(1.0 + exp(z))
end

# Stable log-sigmoid and log(1-sigmoid)
@inline logσ(z::Float64)   = z ≥ 0 ? -log1p(exp(-z))     : (z - log1p(exp(z)))
@inline log1mσ(z::Float64) = z ≥ 0 ? (-z - log1p(exp(-z))) : (-log1p(exp(z)))

# --------------------------
# Fast inner fixed point (EV)
# EV(x,i) = E_y[ log Σ_j exp(u(y,j)+β EV(y,j)) | x,i]
# O(N) per iteration, no large temporaries
# --------------------------
function solve_EV(model::RustModel, θ11::Float64, RC::Float64;
                  tol::Float64=1e-4, maxit::Int=300_000, mix::Float64=0.5,
                  EV0::Union{Nothing,AbstractMatrix}=nothing)

    N, β = model.N, model.β
    p0, p1, p2 = model.p0, model.p1, model.p2
    x = model.xgrid

    # Start from provided warm start or zeros
    EV = EV0 === nothing ? zeros(Float64, N, 2) : copy(EV0)
    w  = zeros(Float64, N)

    # Deterministic utilities:
    # keep: u0(y) = -1e-6*θ11*x[y]; replace: u1 is constant = -RC (since maintenance_cost(0,θ11)=0)
    a   = -1e-6 * θ11
    u1c = -RC

    for _ in 1:maxit
        # w(y) = log Σ_j exp(u(y,j)+β EV(y,j)) with stable 2-term logsumexp
        @inbounds for y in 1:N
            v0 = a * x[y] + β * EV[y, 1]  # keep
            v1 = u1c       + β * EV[y, 2]  # replace
            w[y] = logsumexp2(v0, v1)
        end

        # Replacement column is identical across states:
        rep_val = p0*w[1] + p1*w[min(2, N)] + p2*w[min(3, N)]

        # Mix in-place: EV .= (1-mix)*EV + mix*EVcand without forming EVcand
        diff = 0.0

        # Keep column: 3-point stencil with saturation to N
        @inbounds for i in 1:(N-2)
            newv  = p0*w[i] + p1*w[i+1] + p2*w[i+2]
            old   = EV[i, 1]
            mixed = (1 - mix)*old + mix*newv
            d = abs(mixed - old); if d > diff; diff = d; end
            EV[i, 1] = mixed
        end
        @inbounds begin
            # i = N-1
            newv  = p0*w[N-1] + p1*w[N] + p2*w[N]
            old   = EV[N-1, 1]
            mixed = (1 - mix)*old + mix*newv
            d = abs(mixed - old); if d > diff; diff = d; end
            EV[N-1, 1] = mixed

            # i = N: p0+p1+p2=1 ⇒ newv = w[N]
            newv  = w[N]
            old   = EV[N, 1]
            mixed = (1 - mix)*old + mix*newv
            d = abs(mixed - old); if d > diff; diff = d; end
            EV[N, 1] = mixed
        end

        # Replace column: fill with the same value
        @inbounds for i in 1:N
            old   = EV[i, 2]
            mixed = (1 - mix)*old + mix*rep_val
            d = abs(mixed - old); if d > diff; diff = d; end
            EV[i, 2] = mixed
        end

        if diff < tol
            return EV
        end
    end
    error("EV did not converge within maxit (increase maxit or adjust mix).")
end

# --------------------------
# Choice probabilities (two‑choice logit, stable)
# --------------------------
function ccp(model::RustModel, EV::AbstractMatrix{<:Real}, θ11::Float64, RC::Float64)
    N, β = model.N, model.β
    P = similar(EV)
    a   = -1e-6 * θ11
    u1c = -RC
    @inbounds for x in 1:N
        v0 = a * model.xgrid[x] + β * EV[x, 1]
        v1 = u1c                 + β * EV[x, 2]
        z  = v1 - v0                 # difference-only logistic
        p1 = safe_logistic(z)        # P(replace | x)
        P[x, 2] = p1
        P[x, 1] = 1.0 - p1
    end
    return P
end

# --------------------------
# Robust data preparation
# --------------------------
_to_int_vec(v) = eltype(v) <: Integer ? Int.(v) :
                 eltype(v) <: Real    ? round.(Int, v) :
                 begin
                    s = lowercase.(strip.(String.(v)))
                    [si in ("1","true","replace") ? 1 : 0 for si in s]
                 end

"""
    prepare_panel(df, model, state_col, decision_col, x_is_bin)

Return (states, decisions) with states in 1..N and decisions in {0,1}.
If `x_is_bin=false`, interpret `state_col` as raw miles and map by floor(miles/Δ)+1.
Detect and recover if the mapping collapses to a single bin.
"""
function prepare_panel(df::DataFrame, model::RustModel,
                       state_col::Symbol, decision_col::Symbol, x_is_bin::Bool)

    dcol = df[!, decision_col]
    decisions = _to_int_vec(dcol)

    scol = df[!, state_col]
    vreal = eltype(scol) <: Real ? Float64.(scol) : parse.(Float64, String.(scol))

    states = Vector{Int}()
    interpreted_as = ""

    if x_is_bin
        mn, mx = minimum(vreal), maximum(vreal)
        if 0.0 ≤ mn && mx ≤ model.N + 5
            s = round.(Int, vreal)
            states = (minimum(s) == 0) ? (s .+ 1) : s
            interpreted_as = "bin index (x_is_bin=true)"
        else
            states = clamp.(floor.(Int, vreal ./ model.Δ) .+ 1, 1, model.N)
            interpreted_as = "miles (auto-corrected despite x_is_bin=true)"
        end
    else
        states = clamp.(floor.(Int, vreal ./ model.Δ) .+ 1, 1, model.N)
        interpreted_as = "miles (x_is_bin=false)"
    end

    if !isempty(states) && minimum(states) == maximum(states)
        s_alt = round.(Int, vreal)
        if 0 ≤ minimum(s_alt) ≤ (model.N + 5) && maximum(s_alt) ≤ (model.N + 5)
            states = (minimum(s_alt) == 0) ? (s_alt .+ 1) : s_alt
            interpreted_as *= " → recovered as bin index"
        end
    end

    if isempty(states) || minimum(states) == maximum(states)
        error("State column ‘$(state_col)’ collapsed to a single bin. ",
              "If it’s a 5k‑mile bin index, call with x_is_bin=true; ",
              "if it is raw miles, call with x_is_bin=false.")
    end

    @info "Interpreting ‘$(state_col)’ as $(interpreted_as). Range = " *
          "[$(minimum(states)),$(maximum(states))]"
    return states, decisions
end

# --------------------------
# Count observations by (state, decision) once
# --------------------------
function counts_by_state_dec(states::Vector{Int}, decisions::Vector{Int}, N::Int)
    C = zeros(Float64, N, 2)
    @inbounds for t in eachindex(states)
        C[states[t], decisions[t] + 1] += 1.0
    end
    return C
end

# --------------------------
# Negative log-likelihood at given (log-params), fast (O(N))
# --------------------------
function nll_logparams(p::Vector{Float64}, model::RustModel,
                       states::Vector{Int}, decisions::Vector{Int};
                       inner_tol::Float64=1e-4, inner_maxit::Int=300_000, mix::Float64=0.5)

    counts = counts_by_state_dec(states, decisions, model.N)
    RC  = exp(p[1]); θ11 = exp(p[2])
    EV  = solve_EV(model, θ11, RC; tol=inner_tol, maxit=inner_maxit, mix=mix)

    β   = model.β
    a   = -1e-6 * θ11
    u1c = -RC
    x   = model.xgrid

    ll = 0.0
    @inbounds for s in 1:model.N
        v0 = a * x[s] + β * EV[s, 1]
        v1 = u1c      + β * EV[s, 2]
        z  = v1 - v0
        ll += counts[s, 2] * logσ(z) + counts[s, 1] * log1mσ(z)
    end
    return -ll
end

# --------------------------
# Estimation (two-stage, derivative-free; warm starts; O(N) objective)
# --------------------------
function estimate(model::RustModel;
                  df::DataFrame,
                  state_col::Symbol = :state,
                  decision_col::Symbol = :decision,
                  x_is_bin::Bool = true,
                  initial::Tuple{<:Real,<:Real} = (10.0, 2.0),
                  inner_tol::Float64 = 1e-4,
                  inner_maxit::Int = 300_000,
                  mix::Float64 = 0.5)

    states, decisions = prepare_panel(df, model, state_col, decision_col, x_is_bin)
    @info "Replacement share: $(round(sum(decisions)/length(decisions), digits=4))"

    # Pre-aggregate once
    C = counts_by_state_dec(states, decisions, model.N)

    # Warm-start cache for EV across objective evaluations
    EV_cache = zeros(Float64, model.N, 2)

    obj = function (p::Vector{Float64})
        RC  = exp(p[1]); θ11 = exp(p[2])
        EV  = solve_EV(model, θ11, RC; tol=inner_tol, maxit=inner_maxit, mix=mix, EV0=EV_cache)

        β   = model.β
        a   = -1e-6 * θ11
        u1c = -RC
        x   = model.xgrid

        ll = 0.0
        @inbounds for s in 1:model.N
            v0 = a * x[s] + β * EV[s, 1]
            v1 = u1c      + β * EV[s, 2]
            z  = v1 - v0
            ll += C[s, 2] * logσ(z) + C[s, 1] * log1mσ(z)
        end
        EV_cache .= EV
        return -ll
    end

    # Bounds in log-parameter space: RC ∈ [0.1, 50], θ11 ∈ [0.1, 10]
    lower = log.([0.1, 0.1])
    upper = log.([50.0, 10.0])

    # Initial point, clamped into the box
    x0 = clamp.(log.(collect(initial)), lower, upper)

    # --- Stage 1: unconstrained Nelder–Mead warm start ---
    res1 = Optim.optimize(obj, x0, NelderMead(), Optim.Options(; iterations = 5_000))

    # Clamp the warm start into the box; ensure finite and strictly inside
    x1 = Optim.minimizer(res1)
    if !all(isfinite, x1)
        x1 = x0
    end
    ε = 1e-8
    x1 = min.(max.(x1, lower .+ ε), upper .- ε)

    # --- Stage 2: bounded search (Fminbox) ---
    res2 = Optim.optimize(obj, lower, upper, x1,
                          Fminbox(NelderMead()), Optim.Options(; iterations = 20_000))

    x̂ = Optim.minimizer(res2)
    return (RC = exp(x̂[1]), θ11 = exp(x̂[2]), nll = Optim.minimum(res2), result = res2)
end

# --------------------------
# High-level runner
# --------------------------
function run_rust1987(csv_path::AbstractString;
                      state_col::Symbol = :state,
                      decision_col::Symbol = :decision,
                      x_is_bin::Bool = true,
                      inner_tol::Float64 = 1e-4,
                      inner_maxit::Int = 300_000,
                      mix::Float64 = 0.5)

    df = CSV.read(csv_path, DataFrame)
    @printf "Loaded %d rows. Columns: %s\n" nrow(df) string(names(df))

    model = build_model(; N=90, β=0.9999, Δ=5000.0, θ30=0.3919, θ31=0.5953)

    est = estimate(model;
                   df=df, state_col=state_col, decision_col=decision_col, x_is_bin=x_is_bin,
                   initial=(10.0, 2.0), inner_tol=inner_tol, inner_maxit=inner_maxit, mix=mix)

    @printf "\n=== Rust (1987) — NFXP Step 2 (fast) ===\n"
    @printf "β = %.4f, θ30 = %.4f, θ31 = %.4f (p2 = %.4f)\n" model.β model.p0 model.p1 model.p2
    @printf "Bins: %d, Δ=%.0f miles\n" model.N model.Δ
    @printf "Estimates: RC = %.4f, θ11 = %.6f\n" est.RC est.θ11
    @printf "Negative log-likelihood (nll): %.6f\n\n" est.nll

    return (model=model, estimates=est)
end

# --------------------------
# Optional sanity check: nll at Rust's published numbers
# --------------------------
function nll_at_rust(df::DataFrame, model::RustModel;
                     RC::Float64=10.075, θ11::Float64=2.293,
                     state_col::Symbol=:state, decision_col::Symbol=:decision,
                     x_is_bin::Bool=true,
                     inner_tol::Float64=1e-4, inner_maxit::Int=300_000, mix::Float64=0.5)
    states, decisions = prepare_panel(df, model, state_col, decision_col, x_is_bin)
    return nll_logparams([log(RC), log(θ11)], model, states, decisions;
                         inner_tol=inner_tol, inner_maxit=inner_maxit, mix=mix)
end

end # module Rust_replication


