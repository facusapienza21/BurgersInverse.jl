# using ADTypes
# using DifferentiationInterface
# using ModelingToolkit, 
using DifferentialEquations
using TaylorDiff
# using ForwardDiff
# using Enzyme, Zygote, ReverseDiff
# using SciMLSensitivity
using Plots
using LaTeXStrings
using Plots.PlotMeasures


# Let's import utils so we can bring the real solution
include("utils.jl")

function solve_Burgers_reverse(t::Float64, λ::Float64) 

    τ = -log(-t)

    function burgers_reverse_ODE!(du,u,p,t)
        du .= (p .* u) ./ (exp(t) .* u .-p .- 1)
    end

    tspan = [-log(2.0), τ]

    # τ = t
    # function burgers_ODE!(du,u,p,t)
    #     du .= (λ .* u) ./ ((1 .+ λ) .* t .+ u)
    # end
    # tspan = [-2.0, t]

    # u0 = [1.0]
    # prob = ODEProblem(burgers_reverse_ODE!, u0, tspan, [λ])
    # sol = solve(prob, Vern9(), adaptive=false, dt=0.01, force_dtmin=true)
    # prob = ODEProblem(burgers_ODE!, u0, tspan, [λ])
    # sol = solve(prob, Vern9(), adaptive=false, dt=0.01, force_dtmin=true)
    
    u0 = [big(1.0)] 
    prob = ODEProblem(burgers_reverse_ODE!, u0, tspan, [big(λ)])
    # prob = ODEProblem(burgers_ODE!, u0, tspan, [big(λ)])
    sol = solve(prob, Feagin14(), adaptive=false, dt=0.01, force_dtmin=true);

    # sol = solve(prob, Rodas5P(), adaptive=false, dt=Δt, force_dtmin=true)
    return sol

end

function ODETaylorAD(sol::ODESolution, t::Float64, P::Int)

    # Obtain important information from numerical solver
    # We change variables here
    τ = -log(-t)
    λ = sol.prob.p[1]
    λ_taylor = TaylorScalar(λ, tuple(big(0.0)))

    # @infiltrate
    # Let's bypass the numeric solution by the real solution
    # THIS NOW GIVES NICE SMOOTH SOLUTION!
    # @warn "By-passing exact solution of Burger's Equation"
    # u = u_true(t, Float64(λ))
    # u = big(u_true(t, Float64(λ)))
    u = only(sol(τ))

    # Here we bypass for now the value of the function
    # f₀(x, t) = (λ * x) / ((1+λ) * t + x)
    # f₀(x, t) = λ / ((1.0+λ) * t + x) * x
    f₀(x,t,λ) = λ / ((big(1.0)+λ) * t + x) * x
    # f = sol.prob.f
    # f₀(x) = (only ∘ f)(x, sol.prob.p, t)


    # Create originat Taylor scalar 
    u_taylor = TaylorScalar(u, tuple(f₀(u, t, λ)))
    # t_taylor = TaylorScalar(t, tuple(1.0))

    # I can replace this with TaylorDiff.make_seed! 
    t_taylor = TaylorScalar(big(t), tuple(big(1.0)))

    # Iterate to obtain higher-order derivatives
    for i in 1:(P-1)
        # f_taylor = f₀(u_taylor, t_taylor)
        f_taylor = f₀(u_taylor, t_taylor, λ_taylor)
        u_taylor = TaylorScalar(u, (u_taylor.partials..., (1/(i+1)) * f_taylor.partials[end]))
        t_taylor = TaylorScalar(big(t), tuple(big.(float.([j==1 for j in 1:(i+1)]))...))
        # t_taylor = TaylorScalar(t, tuple(float.([j==1 for j in 1:(i+1)])...))
        λ_taylor =  TaylorScalar(λ, tuple(big.(zeros(i+1))...))
    end
    # return u_taylor
    return TaylorDiff.extract_derivative(u_taylor, Val(P))
end

# lambdas to try
λ_index = collect(-0.5:0.01:6.5)
λ_integer = round.(Int, λ_index)
λ_derivative = 3 .+ 2 .* λ_integer 
λs = 1 ./ (2 .+ 2 .* λ_index)

λ_derivative_value = Float64[]

for i in eachindex(λs)

    # We need a custom final time for each order
    if λs[i] > 1//3
        t_final = -0.001
    elseif λs[i] > 1//5
        t_final = -0.01
    elseif λs[i] > 1//7
        t_final = -0.02
    elseif λs[i] > 1//9
        t_final = -0.05
    elseif λs[i] > 1//13
        t_final = -0.1
    else
        t_final = -0.15
    end

    sol = solve_Burgers_reverse(t_final, λs[i])
    # sol_AD = ODETaylorAD(sol, -0.02, λ_derivative[i])
    # @show length(sol_AD.partials)
    # @show sol.u[end-5:end]
    ts = -exp.(.-sol.t)
    # ∂u  = [ODETaylorAD(sol, t, λ_derivative[i]).partials[end] for t in ts]
    ∂u  = [ODETaylorAD(sol, t, λ_derivative[i]) for t in ts]
    ∂u₊_finite_differeces = finite_differences(∂u, ts, 1)
    @printf "λ: %.3f \t %2d-th order derivative: %.9f \n"  λs[i] (λ_derivative[i]+1) ∂u₊_finite_differeces[end-2]
    push!(λ_derivative_value, ∂u₊_finite_differeces[end-2])
end

derivative_lambda_plot = plot(λs, abs.(λ_derivative_value), yscale=:log10, xscale=:log2,markershape=:circle, label="Maximum residual around t=0")
# vline!([1//2], label=L"\lambda = 1/2")
λ_cont = [1//2, 1//4, 1//6, 1//8, 1//10, 1//12, 1//14]
vline!(λ_cont, label="1/2") 
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, xticks=(λ_cont,[L"\frac{1}{2}",L"\frac{1}{4}",L"\frac{1}{6}",L"\frac{1}{8}",L"\frac{1}{10}",L"\frac{1}{12}",L"\frac{1}{14}"]), xformatter = xi -> convert(Rational, xi),
        legend=true, size=(1200,700), dpi=600, xlabel=L"\lambda", ylabel=L"\frac{\partial^3 u}{\partial t^3}", margin=7mm)
Plots.savefig(derivative_lambda_plot, "plot_taylor_lambda.pdf")

# I can detect this, as long as I can solve without blowing up everythign

λ₀ = 0.4
sol_ = solve_Burgers_reverse(-0.01, λ₀)
# ts = collect(-2.0:0.001:-0.01)
ts = -exp.(.-sol_.t)
∂u  = [ODETaylorAD(sol_, t, 3) for t in ts]
# ∂u₊ = [ODETaylorAD(sol_, t, 4).partials[end] for t in ts]
# ∂u₊_finite_differeces = finite_differences(∂u, ts, 1)
derivatives_plot = plot(ts, ∂u, lw = 5, label = "Third derivative")
# plot!(ts, ∂u₊, lw = 5, label = "Fourth derivative")
# plot!(ts, ∂u₊./maximum(abs.(∂u₊)), lw = 5, label = "Fourth derivative")
# plot!(ts, ∂u₊_finite_differeces, lw = 5, label = "Fourth derivative - Finite differeences")
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, legend=true, size=(1200,700), dpi=600)
Plots.savefig(derivatives_plot, "plot_Taylor_third.pdf")

# MAKE THIS WORK WITH BIG, SINCE I THINK THIS IS NOT SOLVING
# WELL THE 4TH DERIVATIVE BASED ON THE 3RD

# CHECK ALL THE CODE AGAIN! YOU TOUCHED TOO MUCH

# Quality check

using ForwardDiff

λ₀ = 0.5
t₀ = -0.001
u₀ = u_true(t₀, 0.5)

f(x, t) = (λ₀) / ((1+λ₀) * t + x) * x

∂f∂u = ForwardDiff.derivative(u -> f(u, t₀), u₀)
∂f∂t = ForwardDiff.derivative(t -> f(u₀, t), t₀)

# Let's be sure that TaylorAD works here
# t₀_taylor = TaylorScalar(big(t₀), tuple(big(1.0)))
t₀_taylor = TaylorScalar(t₀, tuple(1.0))
∂f∂t_taylor = f(u₀, t₀_taylor).partials[end]

u₀_taylor = TaylorScalar(u₀, tuple(1.0))
∂f∂u_taylor = f(u₀_taylor, t₀).partials[end]

@show ∂f∂t 
@show ∂f∂t_taylor
@assert ∂f∂t ≈ ∂f∂t_taylor

@show ∂f∂u
@show ∂f∂u_taylor
@assert ∂f∂u ≈ ∂f∂u_taylor 

dfdt = ∂f∂u * f(u₀, t₀) + ∂f∂t
@show dfdt

sol_ = solve_Burgers_reverse(t₀, λ₀)
d²udt² = ODETaylorAD(sol_, t₀, 2)
@show d²udt²

@assert d²udt² ≈ dfdt

# It seems that the last derivative does not capture the divergence using TaylorAD