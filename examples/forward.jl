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


tspan = [-2.0, 0.0]
# u0 = [0.0f0] 
u0 = [1.0f0] 
λ₀ = [0.3f0]


"""
True solution of the Burgers equation
"""
function u_true(y, λ)
    if y <= 0.0 
        burgers_constraint(u,x) = x .+ u .+ u.^(1.0+1.0/λ)
        implicit_prob = NonlinearProblem(burgers_constraint, [0.0], y)
        sol = solve(implicit_prob)
        return only(sol.u)
    else
        return -u_true(-y, λ)
    end
end

###
# Tons of derivatives calculation 

# function f₀(u,t;λ)
#     return (1+λ) * t + u 
# end

# function f(u,t;λ)
#     return λ * u / f₀(u,t;λ)
# end

# function ∂f∂u(u,t;λ)
#     return ( λ * f₀(u,t;λ) - λ * u ) /f₀(u,t;λ)^2
# end

# function ∂f∂t(u,t;λ)
#     return - λ * (1 + λ) * u / f₀(u,t;λ)^2
# end

# function ∂²f∂t∂u(u,t;λ)
#     return 
# end

# function ∂²f∂t²(u,t;λ)
#     return 
# end

# function ∂²f∂u²(u,t;λ)
#     return 
# end

###

### Finite order derivative 

# function ∂³(x)
#     return - 0.5 .* circshift(x,-2) .+ circshift(x,-1) .- circshift(x,1) .+ 0.5 .* circshift(x,2)
# end

function burgers_ODE!(du,u,λ,t)
    du .= λ.*u ./ (u .+ (big(1.0f0).+λ).*t)
end

function burgers_ODE(u,λ,t)
    return λ.*u ./ (u .+ (big(1.0f0).+λ).*t)
end

function residual(a, b)
    @assert length(a) == length(b)
    res = zeros(length(a))
    for i in 1:length(a)
        if (abs(a[i]-999.0) > 0.1) & (abs(b[i] - 999.0) > 0.1)
            res[i] = a[i] - b[i]
        else
            res[i] = nothing
        end
    end
    return res
end

movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]

function finite_differences(x, derivative, order, Δx)
    method = central_fdm(order, derivative)
    grid = method.grid
    coefs = method.coefs
    res = zeros(length(x))
    for i in 1:length(grid)
        res .+= coefs[i] .* circshift(x, -grid[i])
    end
    largest_index = grid[end]
    res[begin:largest_index] .= 999
    res[end-largest_index:end] .= 999
    return res / (Δx^derivative)
end 

function forward(λ; make_plot=false)

    Δt = 1//100000

    prob = ODEProblem(burgers_ODE!, [big(only(u0))], tspan, λ)

    # sol = solve(prob, Tsit5(), dt=Δt, adaptive=false, force_dtmin=false)
    sol = solve(prob, Feagin14(), dt=Δt, adaptive=false);

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
        res = residual(finite_differences(u, 1, 12, δt), f)
        ylim = 1.1 * Float32(maximum(abs.(res[begin+20:end-20])))
        # res ./= first.(sol.u)

        # derivatives_plot = plot(sol.t, ∂³res, lw = 5, label = "Derivatives Residual")
        derivatives_plot = plot(ts, res, lw = 5, label = "Residual")
        plot!(ts, movingaverage(res, Int(round(length(res)/100))), label="Running mean")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), ylimits=(-ylim,ylim),legend=true, size=(1200,700), dpi=600)
        Plots.savefig(derivatives_plot, "plot_residual.pdf")

        # Plot derivatives of residual
        ∂³res = finite_differences(first.(sol.u), 4, 5, 1.0) / Δt .- finite_differences(first.(sol.(sol.t, Val{1})), 3, 4, 1.0)
        ∂³res ./= first.(sol.u)
        ylim = 1.1 * maximum(abs.(∂³res[begin+20:end-20]))

        derivatives_plot = plot(sol.t, ∂³res, lw = 5, label = "Derivatives Residual")
        plot!(sol.t, movingaverage(∂³res, Int(round(length(res)/100))), label="Running mean")
        plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18,
        xlimits=(tspan[1], tspan[2]), ylimits=(-ylim, ylim), legend=true, size=(1200,700), dpi=600)
        Plots.savefig(derivatives_plot, "plot_derivatives_residual.pdf")

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

sol = forward(0.4; make_plot=true)

# forward(0.5; make_plot=true);

# forward(0.6; make_plot=true);

# λ_scan = LinRange(0.45, 0.55, 11)
# λ_scan = LinRange(0.10, 0.60, 100)

# time_steps = Float64[]
# abs_error = Float64[]
# last_step = Float64[]
# last_u = BigFloat[]

# for λ in λ_scan
#     println("Trying...λ = ", λ)
#     _sol = forward(λ; make_plot=false)
#     # push!(time_steps, length(_sol.u))
#     # push!(last_step, _sol.t[end-1]-_sol.t[end-2])
#     push!(last_u, only(_sol.u[end]))
#     # push!(abs_error, mean((u_true.(_sol.t, Ref(λ)) .- first.(_sol.u)).^2))
# end

# plot(λ_scan, abs.(last_u), yscale=:log10, markershape=:circle)
# vline!([1//2,1//4,1//6]) 
# plot!(fontfamily="Computer Modern",
#     #title="PIL51",
#     titlefontsize=18,
#     tickfontsize=15,
#     legendfontsize=15,
#     guidefontsize=18,
#     # ylimits=(-1e-12,1e-12),
#     # xlimits=(tspan[1], tspan[2]),
#     legend=true,
#     # margin= 7mm,
#     size=(1200,700),
#     dpi=600)
# plot(λ_scan, last_step)
# plot(λ_scan, time_steps)
# plot(λ_scan, abs_error)


