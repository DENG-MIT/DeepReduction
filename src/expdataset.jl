
l_exp = []
push!(l_exp, readdlm(ckpt_path * "/../F24_p20bar_phi0.5.txt", ','))
push!(l_exp, readdlm(ckpt_path * "/../F24_p20bar_phi1.0.txt", ','))
push!(l_exp, readdlm(ckpt_path * "/../F24_p20bar_phi1.3.txt", ','))

exp_all = vcat(l_exp...)

n_exp = size(exp_all)[1]
conds = zeros(n_exp, 4)

conds[:, 1] .= 1000.0 ./ exp_all[:, 1]
conds[:, 4] .= exp_all[:, 2] ./ 1000
conds[:, 2] .= 20/ 1.013

conds[1:41, 3] .= 0.5 * 1.08831169
conds[42:68, 3] .= 1.0 * 1.08831169
conds[69:96, 3] .= 1.3 * 1.08831169

n_train = Int64(floor(n_exp * 0.8))

for i_exp in 1:n_exp
    local T0 = conds[i_exp, 1]
    local P = conds[i_exp, 2] * one_atm
    local phi = conds[i_exp, 3]
    @printf("(%d) T %.1f [K] P %.1f [atm] phi %.1f IDT0 = %.2e [s] \n",
            i_exp, conds[i_exp, 1], conds[i_exp, 2],
            conds[i_exp, 3], conds[i_exp, 4])
end
@save string(ckpt_path, "/../conds.bson") conds

Random.seed!(0)
l_all = randperm(n_exp)
l_exp_train = l_all[1:n_train]
l_exp_val = l_all[n_train+1:end]
