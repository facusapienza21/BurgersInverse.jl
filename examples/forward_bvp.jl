"""
Example trying to solve the Burgers equation as a boundary problem to impose u(0) = 0 
"""

using BoundaryValueDiffEq
using Plots

# const g = 9.81
# L = 1.0
# tspan = (0.0, pi / 2)

# function simplependulum!(du, u, p, t)
#     θ = u[1]
#     dθ = u[2]
#     du[1] = dθ
#     du[2] = -(g / L) * sin(θ)
# end

# function bc1!(residual, u, p, t)
#     residual[1] = u[end ÷ 2][1] + pi / 2 # the solution at the middle of the time span should be -pi/2
#     residual[2] = u[end][1] - pi / 2 # the solution at the end of the time span should be pi/2
# end
# bvp1 = BVProblem(simplependulum!, bc1!, [pi / 2, pi / 2], tspan)
# sol1 = solve(bvp1, MIRK4(), dt = 0.05)

# plot(sol1)

# function bc2a!(resid_a, u_a, p) # u_a is at the beginning of the time span
#     resid_a[1] = u_a[1] + pi / 2 # the solution at the beginning of the time span should be -pi/2
# end
# function bc2b!(resid_b, u_b, p) # u_b is at the ending of the time span
#     resid_b[1] = u_b[1] - pi / 2 # the solution at the end of the time span should be pi/2
# end
# bvp2 = TwoPointBVProblem(simplependulum!, (bc2a!, bc2b!), [pi / 2, pi / 2], tspan;
#     bcresid_prototype = (zeros(1), zeros(1)))
# sol2 = solve(bvp2, MIRK4(), dt = 0.05)
# plot(sol2)

### Now let's do this with Burgers! 

tspan = (-2.0, 0.0)
u0 = [1.0f0] 
λ₀ = [0.5f0]

function burgers_ODE!(du, u, λ, t)
    # For now we do this with regular precission
    du[1] = λ[1] * u[1] / (u[1] + (1.0f0+λ[1])*t)
end

function bc2a!(resid_a, u_a, p) # u_a is at the beginning of the time span
    resid_a[1] = u_a[1] - 1.0 # the solution at the beginning of the time span should be 1
end
function bc2b!(resid_b, u_b, p) # u_b is at the ending of the time span
    resid_b[1] = u_b[1] - 0.0 # the solution at the end of the time span should be 0
end

bvp_Burgers = TwoPointBVProblem(burgers_ODE!, (bc2a!, bc2b!), [1.0], tspan, λ₀;
    bcresid_prototype = (zeros(1), zeros(1)))

# Adaptive = false seems to fail in all cases
sol_Burgers2 = solve(bvp_Burgers, MIRK2(), dt = 0.005, adaptive=true)
sol_Burgers3 = solve(bvp_Burgers, MIRK3(), dt = 0.005, adaptive=true)
sol_Burgers4 = solve(bvp_Burgers, MIRK4(), dt = 0.005, adaptive=true)
# sol_Burgers_sh = solve(bvp_Burgers, Shooting(Tsit5()), dt = 0.005, adaptive=true)
sol_Burgers5 = solve(bvp_Burgers, LobattoIIIa2(), dt = 0.005, adaptive=true)
# sol_Burgers6 = solve(bvp_Burgers, RadauIIa2(), dt = 0.005, adaptive=true) # unstable
 
plot(sol_Burgers2)
plot!(sol_Burgers3)
plot!(sol_Burgers4)