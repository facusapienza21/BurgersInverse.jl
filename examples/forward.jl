using Pkg; Pkg.activate(".")

# using OrdinaryDiffEq 
# using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using DifferentialEquations
using Plots
using NonlinearSolve
using Statistics, LinearAlgebra
using FiniteDifferences
using Random, Distributions
using Infiltrator

include("./utils.jl")
include("./plot.jl")

tspan = [-2.0, 0.0]
# u0 = [0.0f0] 
u0 = [1.0f0] 
λ₀ = [0.3f0]

"""
Out-place Burgers update
"""
function burgers_ODE!(du,u,λ,t)
    du .= λ.*u ./ (u .+ (big(1.0f0).+λ).*t)
end

"""
In-place Burgers update
"""
function burgers_ODE(u,λ,t)
    return λ.*u ./ (u .+ (big(1.0f0).+λ).*t)
end


prob = ODEProblem(burgers_ODE!, [big(only(u0))], tspan, λ)
sol = solve(prob, Feagin14(), dt=Δt, adaptive=false);

plot(sol)

function forward(λ, Δt=1//10000; make_plot=false)

    prob = ODEProblem(burgers_ODE!, [big(only(u0))], tspan, λ)

    # sol = solve(prob, Tsit5(), dt=Δt, adaptive=false, force_dtmin=false)
    sol = solve(prob, Feagin14(), dt=Δt, adaptive=false);

    # Trying BVP with different solver

    # function bc2a!(resid_a, u_a, p) # u_a is at the beginning of the time span
    #     resid_a[1] = u_a[1] - 1.0 # the solution at the beginning of the time span should be 1
    # end
    # function bc2b!(resid_b, u_b, p) # u_b is at the ending of the time span
    #     resid_b[1] = u_b[1] - 0.0 # the solution at the end of the time span should be 0
    # end
    
    # bvp_Burgers = TwoPointBVProblem(burgers_ODE!, (bc2a!, bc2b!), [1.0], tspan, λ;
    #     bcresid_prototype = (zeros(1), zeros(1)))

    # sol= solve(bvp_Burgers, MIRK3(), dt = 0.005, adaptive=true)


    # @show sol.t[end] - sol.t[end-1]
    # @show length(sol.t)
    # @show sol.u[end]

    if make_plot

        # We are going to evaluate the solution and its derivatives in a new grid, in order
        # to emulate the "test" procedure.
        n_nodes = 13131 
        # ts = sort(rand(Uniform(tspan[1], tspan[2]), n_nodes))
        ts = LinRange(tspan[1], tspan[2], n_nodes)
        u = first.(sol.(ts))
        f = first.(sol.(ts, Val{1}))
        δt = (tspan[2]-tspan[1]) / n_nodes
        # Plot solution

        # @infiltrate

        sol_plot = plot(sol, linewidth = 10, title = "Burgers Equation",
            xaxis = "y", yaxis = "U(y)", label = "Numerical Solution") # legend=false
        plot!(ts, y -> u_true(y, only(λ)), lw = 5, label = "True Solution")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
            xlimits=(tspan[1], tspan[2]), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(sol_plot, "plot_sol.pdf")

        # Plot residual 
        residual_plot = plot(ts, u_true.(ts, Ref(λ)) - u, lw = 5, label = "Residual")
        # residual_plot = plot(sol.t, u_true.(sol.t, Ref(λ)) - first.(sol.u), lw = 5, label = "Residual")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(residual_plot, "plot_residual.pdf")
        
        # Plot derivatives 

        derivatives_plot = plot(ts, f, lw = 5, label = "First derivative (f)")
        # derivatives_plot = plot(sol.t, ∂³sol, lw = 5, label = "Derivatives")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(derivatives_plot, "plot_derivatives_first_f.pdf")

        # ∂³sol = finite_differences(first.(sol.u), 3, 12, 1.0)
        ∂³sol = finite_differences(u, 3, 12, 1.0)
        ylim = 1.1 * Float32(maximum(abs.(∂³sol[begin+20:end-20])))
        @show ∂³sol[end-20:end]

        derivatives_plot = plot(ts, ∂³sol, lw = 5, label = "Derivatives")
        # derivatives_plot = plot(sol.t, ∂³sol, lw = 5, label = "Derivatives")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), ylimits=(-ylim, ylim), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(derivatives_plot, "plot_derivatives.pdf")

        # Plot solution residuals
        # f_sol = first.(sol.(sol.t, Val{1}))
        f_sol = burgers_ODE(u, Ref(λ), ts)
        # f_sol[begin] = f_sol[end] = 0.0
        # res = finite_differences(u, 1, 12, δt) .- f
        res = residual(finite_differences(u, 1, 4, δt), f)
        ylim = 1.1 * Float32(maximum(abs.(res[begin+20:end-20])))
        # res ./= first.(sol.u)

        # derivatives_plot = plot(sol.t, ∂³res, lw = 5, label = "Derivatives Residual")
        residual_plot = plot(ts, res, lw = 5, label = "Residual")
        plot!(ts, movingaverage(res, Int(round(length(res)/100))), label="Running mean")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), ylimits=(-ylim,ylim),legend=true, size=(1200,700), dpi=600)
        Plots.savefig(residual_plot, "plot_residual.pdf")

        # Plot derivatives of residual
        ∂³res = finite_differences(first.(sol.u), 4, 5, 1.0) / Δt .- finite_differences(first.(sol.(sol.t, Val{1})), 3, 4, 1.0)
        ∂³res ./= first.(sol.u)
        ylim = 1.1 * maximum(abs.(∂³res[begin+20:end-20]))

        derivatives_residual_plot = plot(sol.t, ∂³res, lw = 5, label = "Derivatives Residual")
        plot!(sol.t, movingaverage(∂³res, Int(round(length(res)/100))), label="Running mean")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), ylimits=(-ylim, ylim), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(derivatives_residual_plot, "plot_derivatives_residual.pdf")

        plot_burgers = plot(sol_plot, 
                            derivatives_plot,
                            residual_plot,
                            derivatives_residual_plot,
                            layaut=(2,2))
        Plots.savefig(plot_burgers, "plot_burgers_combo.pdf")


        # Does the residual change with order of magnitude?
        # order_try =  2:16  
        # res_median = Float64[]
        # for order in order_try
        #     res = res = residual(finite_differences(u, 1, order, δt), f)
        #     push!(res_median, median(abs.(res)))
        # end
        # order_plot = scatter(order_try, res_median, lw = 5, label = "Median Error", yscale=:log10)
        # plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, legend=true, size=(1200,700), dpi=600)
        # Plots.savefig(order_plot, "plot_order.pdf")
    end

    return sol
end

sol = forward(0.5; make_plot=true)

# forward(0.5; make_plot=true);

# forward(0.6; make_plot=true);

# λ_scan = LinRange(0.45, 0.55, 11)
# λ_scan = LinRange(0.45, 0.55, 100)

# time_steps = Float64[]
# abs_error = Float64[]
# last_step = Float64[]
# last_u = BigFloat[]

# ts = LinRange(tspan[1], tspan[2], 1313)

# for λ in λ_scan
#     println("Trying...λ = ", λ)
#     _sol = forward(λ, 1//100000; make_plot=false)
#     # push!(time_steps, length(_sol.u))
#     # push!(last_step, _sol.t[end-1]-_sol.t[end-2])
#     push!(last_u, only(_sol.u[end]))
#     # push!(abs_error, mean((u_true.(ts, Ref(λ)) .- first.(_sol(ts))).^2))
# end

# # plot(λ_scan, abs.(last_u), yscale=:log10, markershape=:circle)
# plot(λ_scan, abs.(last_u), yscale=:log10, markershape=:circle)
# vline!([1//2,1//4,1//6]) 
# plot!(fontfamily="Computer Modern",
#     #title="PIL51",
#     titlefontsize=18,
#     tickfontsize=15,
#     legendfontsize=15,
#     guidefontsize=18,
#     # ylimits=(-1e-12,1e-12),
#     xlimits=(λ_scan[begin], λ_scan[end]),
#     legend=true,
#     # margin= 7mm,
#     size=(1200,700),
#     dpi=600)
# plot(λ_scan, last_step)
# plot(λ_scan, time_steps)
# plot(λ_scan, abs_error)


