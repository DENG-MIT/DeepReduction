np = length(p);

function make_prob(T0, P, phi, p; tfinal=1.0)
    X0 = zeros(ns);
    X0[species_index(gas, fuel)] = phi
    X0[species_index(gas, oxygen)] = fuel2air
    X0[species_index(gas, inert)] = fuel2air * 3.76
    X0 = X0 ./ sum(X0);
    Y0 = X2Y(gas, X0, dot(X0, gas.MW));
    u0 = vcat(Y0, T0);

    mean_MW = 1.0 / dot(Y0, 1 ./ gas.MW)
    ρ_mass = P / R / T0 * mean_MW
    global ρ_mass

    prob = ODEProblem(dudt!, u0, (0.0, tfinal), p);
    prob
end


@inbounds function dudt!(du, u, p, t)
    Y = @view(u[1:ns])
    T = u[end]
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    # ρ_mass = P / R / T * mean_MW
    P = ρ_mass * R * T / mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    cv_mole, cv_mass = get_cv(cp_mole, cp_mass, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    u_mole = get_U(h_mole, Y)
    S0 = get_S(gas, T, P, X)
    _p = reshape(p, npr, 3)
    # kp = exp.(p)
    qdot = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true)
    @. qdot[1:npr] *= @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 1000.0 * 4184.0 / R / T))
    wdot = gas.reaction.vk * qdot
    Ydot = wdot / ρ_mass .* gas.MW
    Tdot = -dot(u_mole, wdot) / ρ_mass / cv_mass
    du .= vcat(Ydot, Tdot)
    return du
end

@inbounds function dudtp!(du, u, p, t)
    Y = @view(u[1:ns])
    T = u[end]
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    # ρ_mass = P / R / T * mean_MW
    P = ρ_mass * R * T / mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    cv_mole, cv_mass = get_cv(cp_mole, cp_mass, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    u_mole = get_U(h_mole, Y)
    S0 = get_S(gas, T, P, X)
    _p = reshape(p, npr, 3)
    # kp = exp.(p)
    qdot::typeof(p) = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true)
    @. qdot[1:npr] *= @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 1000.0 * 4184.0 / R / T))
    wdot = gas.reaction.vk * qdot
    Ydot = wdot / ρ_mass .* gas.MW
    Tdot = -dot(u_mole, wdot) / ρ_mass / cv_mass
    du .= vcat(Ydot, Tdot)
    return du
end


function sensBVP_mthreadp(ts, pred, p)
    idt = ts[end]
    Tign = pred[end, end]
    ng = length(ts)
    Fy = BandedMatrix(Zeros(ng * nu, ng * nu), (nu, nu));
    Fp = zeros(ng * nu, np)
    i = 1
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F])[ind_diag] .= ones_nu
    Fy[i * nu, i * nu] = -1.0
    Fy[i * nu, (i + 1) * nu] = 1.0

    dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt

    @threads for i = 2:ng
        u = @view(pred[:, i])
        du = similar(u)
        i_F = 1 + (i - 1) * nu:i * nu - 1
        @view(Fp[i_F, :]) .= jacobian((du, x) -> dudtp!(du, u, x, 0.0),
                                    du, p)::Array{Float64,2} .* (-idt)
    end
    @threads for i = 2:ng
        u = @view(pred[:, i])
        du = similar(u)
        i_F = 1 + (i - 1) * nu:i * nu - 1
        @view(Fy[i_F, i_F]) .= jacobian((du, x) -> dudt!(du, x, p, 0.0),
                                    du, u)::Array{Float64,2} .* (-idt)
    end

    for i = 2:ng
        u = @view(pred[:, i])
        du = similar(u)
        i_F = 1 + (i - 1) * nu:i * nu - 1
        @view(Fy[i_F, i * nu]) .= - dudt!(du, u, p, 0.0)
        @view(Fy[i_F, i_F])[ind_diag] .+= ones_nu ./ (dts[i - 1])
        @view(Fy[i_F, i_F .- nu])[ind_diag] .+= ones_nu ./ (-dts[i - 1])
        if i < ng
            Fy[i * nu, i * nu] = -1.0
            Fy[i * nu, (i + 1) * nu] = 1.0
        else
            Fy[i * nu, i * nu - 1] = 1.0
        end
    end
    dydp = - Fy \ sparse(Fp)
    return @view(dydp[end, :]) ./ idt
end
