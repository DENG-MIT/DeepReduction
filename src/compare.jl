include("header.jl")
include("simulator.jl")
include("visual.jl")

function get_conds!(conds)
    for i_exp in 1:size(conds)[1]
        local T0 = conds[i_exp, 1]
        global P = conds[i_exp, 2] * one_atm
        local phi = conds[i_exp, 3]
        conds[i_exp, 4] = f_idt(T0, P, phi, p; dT=dT)
        @printf("(%d) T %.1f [K] P %.1f [atm] phi %.1f IDT0 = %.2e [s] \n",
                i_exp, conds[i_exp, 1], conds[i_exp, 2],
                conds[i_exp, 3], conds[i_exp, 4])
    end
    return conds
end

conds = zeros(20, 4)
conds[:, 1] .= range(700.0, stop=1600, length=size(conds)[1])
conds[:, 2] .= 40.0
conds[:, 3] .= 0.9

# dir_master = conf["master"]
# @load string(ckpt_path, "/../../", dir_master, "/conds.bson") conds

gas = CreateSolution("mechanism/gri30.yaml");
ns = gas.n_species;
nr = gas.n_reactions;
p = zeros(nr * 3);
conds = get_conds!(conds)
conds_master = deepcopy(conds)

gas = CreateSolution("mechanism/gri30_sk23.yaml");
ns = gas.n_species;
nr = gas.n_reactions;
p = zeros(nr * 3);

conds = get_conds!(conds)
conds_sk = deepcopy(conds)

# gas = CreateSolution("mechanism/gri30_sk23_op.yaml");
# ns = gas.n_species;
# nr = gas.n_reactions;
# p = zeros(nr * 3);

@load string(ckpt_path, "/../p.bson") p
conds = get_conds!(conds)
conds_sk_op = deepcopy(conds)

if std(conds[:, 2]) < 1.e-3
    plt = Plots.plot(1000 ./ conds_master[:, 1], conds_master[:, 4],
                lw=1, ls=:solid, label="gri30")
else
    plt = Plots.scatter(1000 ./ conds_master[:, 1], conds_master[:, 4],
                label="gri30")
end
Plots.scatter!(plt, 1000 ./ conds_sk[:, 1], conds_sk[:, 4],
            label="sk23")
Plots.scatter!(plt, 1000 ./ conds_sk_op[:, 1], conds_sk_op[:, 4],
            label="sk23_op")
xlabel!(plt, "1000 / T [K]")
ylabel!(plt, "IDT [s]")
plot!(plt, yscale=:log10, legend=:left)
title!(plt, @sprintf("p=%.1f atm phi=%.1f CH4:0.85/C2H4:0.1/C3H8:0.05",
        conds[1, 2], conds[1, 3]))
png(plt, string(fig_path, "/regression_T"))
