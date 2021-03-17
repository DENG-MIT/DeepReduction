n_exp = Int64(conf["n_exp"])
n_train = Int64(n_exp * 0.7)

plan = randomLHC(n_exp, 4) ./ n_exp

conds = scaleLHC(plan,[(Float64(conf["temperature"]["lb"]),
                        Float64(conf["temperature"]["ub"])),
                       (Float64(conf["pressure"]["lb"]),
                        Float64(conf["pressure"]["ub"])),
                       (Float64(conf["phi"]["lb"]),
                        Float64(conf["phi"]["ub"])), (0.0, 1.0)])

for i_exp in 1:n_exp
    local T0 = conds[i_exp, 1]
    global P = conds[i_exp, 2] * one_atm
    local phi = conds[i_exp, 3]
    conds[i_exp, 4] = f_idt(T0, P, phi, p; dT=dT, tfinal=10.0)
    if i_exp < 10
        check_sol(T0, P, phi, p; i_exp=i_exp)
    end
    @printf("(%d) T %.1f [K] P %.1f [atm] phi %.1f IDT0 = %.2e [s] \n",
            i_exp, conds[i_exp, 1], conds[i_exp, 2],
            conds[i_exp, 3], conds[i_exp, 4])
end
# @save string(ckpt_path, "/../conds.bson") conds

@load string(ckpt_path, "/../conds.bson") conds
