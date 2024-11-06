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

movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]

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

function finite_differences(x::AbstractVector{<:Real}, derivative::Integer, order::Integer, Δx::Real)
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

"""
Customized stencil, noticed that here in princuple we don't specify order
"""
function finite_differences(x::AbstractVector{<:Real}, t::AbstractVector{<:Real}, t₀::Real,  derivative::Integer)
    weights = stencil(t, t₀, derivative)
    return sum(weights .* x) 
end

function finite_differences(x::AbstractVector{<:Real}, t::AbstractVector{<:Real}, derivative::Integer)
    res = zeros(length(x))
    wd = ceil(Int, derivative / 2)
    for i in 1:length(x)
        if i <= wd
            res[i] = finite_differences(x[begin:derivative+1], t[begin:derivative+1], t[i], derivative)
        elseif i >= (length(x) - wd)
            res[i] = finite_differences(x[end-derivative:end], t[end-derivative:end], t[i], derivative)
        else
            res[i] = finite_differences(x[i-wd:i+wd], t[i-wd:i+wd], t[i], derivative)
        end
    end
    return res
end

"""
Custom stencil function 
source: https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/5
"""
function stencil(x::AbstractVector{<:Real}, x₀::Real, m::Integer)
    ℓ = 0:length(x)-1
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    return A \ (ℓ .== m) # vector of weights w
end