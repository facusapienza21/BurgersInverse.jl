using Symbolics 

@variables t
D = Differential(t)

z = t + t^2
D(z)

@show expand_derivatives(D(z))

@variables x y
Jac = Symbolics.jacobian([x + x*y, x^2 + y], [x, y])

@variables λ
@variables u t
Symbolics.jacobian([ λ * u / ((1 + λ) * t + u)], [u t])