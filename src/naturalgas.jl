
function make_prob(T0, P, phi, p; tfinal=1.0)
    local fuel2air = 0.85 * 2 + 0.1 * 3.5 + 0.05 * 5
    X0 = zeros(ns);
    X0[species_index(gas, "CH4")] = phi * 0.85
    X0[species_index(gas, "C2H6")] = phi * 0.1
    X0[species_index(gas, "C3H8")] = phi * 0.05
    X0[species_index(gas, oxygen)] = fuel2air
    X0[species_index(gas, inert)] = fuel2air * 3.76

    X0 = X0 ./ sum(X0);
    Y0 = X2Y(gas, X0, dot(X0, gas.MW));
    u0 = vcat(Y0, T0);

    prob = ODEProblem(dudt!, u0, (0.0, tfinal), p);
    prob
end
