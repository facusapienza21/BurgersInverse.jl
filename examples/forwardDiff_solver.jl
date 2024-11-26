using DifferentialEquations
using ForwardDiff

"""
Example from: https://discourse.julialang.org/t/error-with-forwarddiff-no-method-matching-float64/41905
This works!
"""
function CostFun_ic(x::AbstractVector{T}) where T
           
    function SpringEqu!(du, u, x, t)
        du[1] = u[2]
        du[2] = -(x[1] / x[3]) * u[2] - (x[2] / x[3]) * u[1] + 50 / x[3]
    end
    
    u0 = T[2.0, 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(SpringEqu!, u0, tspan, x)
    sol = solve(prob)

    Simpos = zeros(T, length(sol.t))
    Simvel = zeros(T, length(sol.t))
    tout = zeros(T, length(sol.t))
    for i = 1:length(sol.t)
        tout[i] = sol.t[i]
        Simpos[i] = sol[1, i]
        Simvel[i] = sol[2, i]
    end

    totalCost = sum(Simpos)
    return totalCost
end

g1 = ForwardDiff.gradient(CostFun_ic, [2000.0, 20000.0, 80.0])

# Let's try now the same but with the time... 

function CostFun_time(x::AbstractVector{T}) where T
           
    function SpringEqu!(du, u, x, t)
        du[1] = u[2]
        du[2] = -(x[1] / x[3]) * u[2] - (x[2] / x[3]) * u[1] + 50 / x[3]
    end
    
    u0 = [2.0, 0.0]
    tspan = [0.0, T(only(x))]
    prob = ODEProblem(SpringEqu!, u0, tspan, [2000.0, 20000.0, 80.0])
    sol = solve(prob)

    Simpos = zeros(T, length(sol.t))
    Simvel = zeros(T, length(sol.t))
    tout = zeros(T, length(sol.t))
    for i = 1:length(sol.t)
        tout[i] = sol.t[i]
        Simpos[i] = sol[1, i]
        Simvel[i] = sol[2, i]
    end

    totalCost = sum(Simpos)
    return totalCost
end

g2 = ForwardDiff.gradient(CostFun_time, [1.0])