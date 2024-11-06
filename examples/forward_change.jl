using Pkg; Pkg.activate(".")

# using OrdinaryDiffEq 
# using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using DifferentialEquations
using Plots
using Plots.PlotMeasures
using NonlinearSolve
using Statistics, LinearAlgebra
using FiniteDifferences
using Random, Distributions
using Infiltrator
using LaTeXStrings

include("./utils.jl")
# include("./plot.jl")

# boundary condition is reversed
tspan = [-log(2.0), 8.0]
u0 = [1.0] 

λ = [0.520]

function τ(t)
    return - log(-t)
end

function time(τ)
    return - exp(-τ)
end

"""
Out-place Burgers update
"""
function burgers_reverse_ODE!(du,u,p,t)
    du .= (p .* u) ./ (exp(t) .* u .-p .- 1)
end

prob = ODEProblem(burgers_reverse_ODE!, [big(only(u0))], tspan, [big(only(λ))])

# sol = solve(prob, Tsit5(), adaptive=true, dt=0.1, force_dtmin=true)
sol = solve(prob, Feagin14(), adaptive=false, dt=0.1, force_dtmin=true);

# Now, let's compute \partial ∂u/∂t using change of variables
u = first.(sol.u)

∂³u∂t³ = exp.(3 .* sol.t) .* (finite_differences(u, 1, 4, 1.0) .+ 2 .* finite_differences(u, 2, 4, 1.0) .+ finite_differences(u, 3, 4, 1.0))

# Now, this drastically fails... so I think I should used this as a solver and comeback to the original 
# reference system t and compute derivaves there.

# Let's first check we got the solution 
t = time.(sol.t)
f = first.(sol.(t, Val{1}))
ts = LinRange(t[begin], t[end], 13131)

sol_plot = plot(t, u, linewidth = 10, title = "Burgers Equation", xaxis = "y", yaxis = "U(y)", label = "Numerical Solution")      
plot!(ts, y -> u_true(y, only(λ)), lw = 5, label = "True Solution")
Plots.savefig(sol_plot, "plot_solution_52.pdf")

derivatives_plot = plot(τ.(t[begin:end-5]), abs.(finite_differences(u, t, 3))[begin:end-5], lw = 5, label = "Derivatives")
# derivatives_plot = plot(sol.t, ∂³sol, lw = 5, label = "Derivatives")
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
legend=true, size=(1200,700), yscale=:log10, dpi=600)

# This seems to be working!!! 
∂³res = finite_differences(u, t, 4) .- finite_differences(f, t, 3)
residual_plot = plot(τ.(t[begin:end-5]), ∂³res[begin:end-5], lw = 5, label = "Residual")
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
legend=true, size=(1200,700), dpi=600)
Plots.savefig(residual_plot, "plot_residual_52.pdf")

λ_scan = LinRange(0.40, 0.60, 2000)
# λ_scan = LinRange(0.10, 0.55, 5000)

last_res = BigFloat[]

d = 3

for _λ in λ_scan
    _prob = ODEProblem(burgers_reverse_ODE!, [big(only(u0))], tspan, [big(only(_λ))])
    _sol = solve(_prob, Feagin14(), adaptive=false, dt=0.1, force_dtmin=true);
    _t = time.(_sol.t)
    _f = first.(_sol.(_t, Val{1}))
    _u = first.(_sol.u)
    _∂³res = finite_differences(_u, _t, d+1) .- finite_differences(_f, _t, d)
    push!(last_res, mean(abs.(_∂³res[end-20:end])))
end

residual_lambda_plot = plot(λ_scan, last_res, yscale=:log10, markershape=:circle, label="Maximum residual around t=0")
vline!([1//2], label=L"\lambda = 1/2") 
# vline!([1//2, 1//4, 1//6], label="1/2") 
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        legend=true, size=(1200,700), dpi=600, xlabel=L"\lambda", ylabel=L"\frac{\partial^3 res}{\partial t^3}", margin=7mm)
Plots.savefig(residual_lambda_plot, "plot_residual_lambda.pdf")