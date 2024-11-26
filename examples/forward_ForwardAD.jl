using ForwardDiff
using SciMLSensitivity
using Zygote 
using Plots
using DifferentialEquations

"""
Refereced from: https://discourse.julialang.org/t/automatic-differentiation-for-higher-order-derivatives/62337/7
"""

# function ndiff(f,x,n)
#     tag = ForwardDiff.Dual{nothing}(x,one(x))
#     for _ in 1:n
#         tag = ForwardDiff.Dual{nothing}(tag,one(tag))
#     end
#     rawres = f(tag)
#     return rawres
# end

function ndiff(f,x,n)
    _funcs = Function[f]
    for i in 1:n
        push!(_funcs, (t) -> ForwardDiff.derivative(_funcs[i], t))
    end
    return [_func(x) for _func in _funcs]
end

# The cost of this grows exponentially, which I think is something to expect... But the derivatives seems to be analytical
# @show ndiff(x->sin(x), 0.0, 5)

# Let's try now by differentiationg solver 

# function solve_Burgers(t::AbstractVector{T}) where T

#     function burgers_reverse_ODE!(du,u,p,t)
#         du .= (p .* u) ./ (exp(t) .* u .-p .- 1)
#     end

#     tspan = [-log(2.0), T(only(t))]
#     u0 = [1.0] 
#     λ = [0.5]
#     prob = ODEProblem(burgers_reverse_ODE!, u0, tspan, λ)
#     sol = solve(prob, Tsit5())
#     # sol = solve(prob, Feagin14(), adaptive=false, dt=0.1, force_dtmin=true);
#     return sol.u[end][1]
# end



function solve_Burgers(t::T, λ, accuracy="Feagin") where T

    τ = -log(-T(t))

    function burgers_reverse_ODE!(du,u,p,t)
        du .= (p .* u) ./ (exp(t) .* u .-p .- 1)
    end

    tspan = [-log(2.0), τ]
    
    if accuracy == "Feagin"
        u0 = [big(1.0)] 
        prob = ODEProblem(burgers_reverse_ODE!, u0, tspan, [big(λ)])
        sol = solve(prob, Feagin14(), adaptive=false, dt=Δt, force_dtmin=true);
        return first.(sol.u)[end][1]
    else
        # For Burgers, try ode15s -> Try Rodas5P() since this is the recommended improvement of this solver coming from Matlab
        # Vern9() is better than Tsit5()
        u0 = [1.0]
        prob = ODEProblem(burgers_reverse_ODE!, u0, tspan, [λ])
        sol = solve(prob, Vern9(), adaptive=false, dt=Δt, force_dtmin=true)
        # sol = solve(prob, Rodas5P(), adaptive=false, dt=Δt, force_dtmin=true)
        return first.(sol.u)[end][1]
    end
end




# t₀ = 8.0

# @show solve_Burgers(t₀, 0.50)

# g1 = Zygote.gradient(x -> solve_Burgers(x), t₀)
# @show g1 

# λ₀ = 0.5
# g2 = ForwardDiff.derivative(x -> solve_Burgers(x, λ₀), t₀)
# @show g2
# @show ForwardDiff.gradient(x -> solve_Burgers(x), [5.0])

# Compute up to five order derivatives

# g45 = ndiff(x->solve_Burgers(x, 0.45), t₀, 4)
# g50 = ndiff(x->solve_Burgers(x, 0.50), t₀, 4)
# g55 = ndiff(x->solve_Burgers(x, 0.55), t₀, 4)

# @show g45
# @show g50
# @show g55

make_derivative_plots = false

if make_derivative_plots

    # Let's check this is giving us the right result
    ts = collect(-2:0.05:-0.01)
    ∂³u = zeros(length(ts))
    for i in eachindex(ts)
        ∂³u[i] = ndiff(x->solve_Burgers(x, 0.40), ts[i], 3)[end]
    end
    # ∂³u = map(t -> ndiff(x->solve_Burgers(x, 0.40), t, 3)[end], ts)

    derivatives_plot = plot(ts, ∂³u, lw = 5, label = "Third derivative (f)")
    plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, legend=true, size=(1200,700), dpi=600)
    Plots.savefig(derivatives_plot, "plot_AD_third.pdf")

    # Let's try now to do fourth derivative

    ∂⁴u = zeros(length(ts))
    for i in eachindex(ts)
        ∂⁴u[i] = ndiff(x->solve_Burgers(x, 0.40), ts[i], 4)[end]
    end

    derivatives_plot = plot(ts, ∂⁴u, lw = 5, label = "Third derivative (f)")
    plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, legend=true, size=(1200,700), dpi=600)
    Plots.savefig(derivatives_plot, "plot_AD_forth.pdf")

end
# Lets try to see the changes for different λs

# Time to evaluate derivatives 
t₀ = -0.01
derivative = 2
# It is not clear if it matters or not to have small or large stepsize
Δt = 0.1

last_res = BigFloat[]
λ_scan = LinRange(0.47, 0.53, 11)

for _λ in λ_scan
    @show _λ
    # GC.gc()
    push!(last_res, ndiff(x->solve_Burgers(x, _λ, "Feagin"), t₀, derivative)[end])
end

last_plot = plot(λ_scan, abs.(last_res), lw = 5, label = "Last residual", yscale=:log10)
scatter!(λ_scan, abs.(last_res), markersize=10)
plot!(fontfamily="Computer Modern", titlefontsize=18, tickfontsize=15, legendfontsize=15, guidefontsize=18, legend=true, size=(1200,700), dpi=600)
Plots.savefig(last_plot, "plot_AD_residual.pdf")
