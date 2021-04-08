function updateyaml(mech, p)
    yaml = YAML.load_file(mech)
    n_species = length(yaml["phases"][1]["species"])
    n_reactions = length(yaml["reactions"])
    species_names = yaml["phases"][1]["species"]
    elements = yaml["phases"][1]["elements"]
    n_elements = length(elements)

    _p = reshape(p, npr, 3)

    for i = 1:npr
        p_vec = _p[i, :]
        fA = exp(p_vec[1])
        fb = p_vec[2]
        fEa = p_vec[3]
        reaction = yaml["reactions"][i]
        if haskey(reaction, "type")
            if (reaction["type"] == "falloff")
                yaml["reactions"][i]["low-P-rate-constant"]["A"] *= fA
                yaml["reactions"][i]["low-P-rate-constant"]["b"] += fb
                yaml["reactions"][i]["low-P-rate-constant"]["Ea"] += fEa
                yaml["reactions"][i]["high-P-rate-constant"]["A"] *= fA
                yaml["reactions"][i]["high-P-rate-constant"]["b"] += fb
                yaml["reactions"][i]["high-P-rate-constant"]["Ea"] += fEa
            else
                yaml["reactions"][i]["rate-constant"]["A"] *= fA
                yaml["reactions"][i]["rate-constant"]["b"] += fb
                yaml["reactions"][i]["rate-constant"]["Ea"] += fEa
            end
        else
            yaml["reactions"][i]["rate-constant"]["A"] *= fA
            yaml["reactions"][i]["rate-constant"]["b"] += fb
            yaml["reactions"][i]["rate-constant"]["Ea"] += fEa * 1000.0
        end
    end

    YAML.write_file(mech[1:end - 5] * "_op.yaml", yaml)
end

pyplot()

epoch = 58
@load string(ckpt_path, "/model_$(epoch).bson") p opt l_loss l_loss_val l_grad l_pnorm iter
# @load string(ckpt_path, "/mymodel.bson") p opt l_loss l_loss_val l_grad l_pnorm iter

# @load string("./results/JetA/checkpoint/model_103.bson") p opt l_loss l_loss_val l_grad l_pnorm iter
# p[33:48] .= p[33:48]/1000.0

writedlm(string(fig_path, "loss.txt"), hcat([l_loss, l_loss_val, l_grad, l_pnorm]...))

updateyaml(mech, p)

plt_loss = plot(l_loss, yscale=:log10, label="train", lw=2);
plot!(plt_loss, l_loss_val, yscale=:log10, label="validation", lw=2, ls=:dash);
plt_grad = plot(l_grad, yscale=:log10, label="grad_norm");
plt_pnorm = plot(l_pnorm .+ 1.e-8, yscale=:identity, label="p_norm");
xlabel!(plt_loss, "Epoch");
xlabel!(plt_grad, "Epoch");
xlabel!(plt_pnorm, "Epoch");
ylabel!(plt_loss, "Loss");
ylabel!(plt_grad, "Grad Norm");
ylabel!(plt_pnorm, "p Norm");
plt = plot(plt_loss, legend=:best);
plot!(
        plt,
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
        size = (350, 350),
    )
png(plt, string(fig_path, "/loss"));

plt = plot(plt_pnorm, legend=:best);
plot!(
        plt,
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
        size = (350, 350),
    )
png(plt, string(fig_path, "/pnorm"));


p_ga = readdlm(string(ckpt_path, "/../p_ga.txt"), '\t')
p_ga = reshape(p_ga, npr*3, 1)

l_phi = [0.5, 1.0, 1.3] .* 1.08831169
l_plt = []
for i in 1:3
    expdata = l_exp[i]
    l_idt = zeros(size(expdata)[1], 5)
    for j in 1:size(expdata)[1]
        local T0 = 1000.0 / expdata[j, 1]
        l_idt[j, 1] = expdata[j, 1]
        l_idt[j, 2] = expdata[j, 2]
        local P = 20.0 / 1.013 * one_atm
        local phi = l_phi[i]
        l_idt[j, 3] = f_idt(T0, P, phi, p.*0; dT=800) * 1000.0
        l_idt[j, 4] = f_idt(T0, P, phi, p; dT=800) * 1000.0
        l_idt[j, 5] = f_idt(T0, P, phi, p_ga; dT=800) * 1000.0
    end
    # plt = Plots.scatter(l_idt[:, 1], l_idt[:, 2], yscale=:log10, label="exp")
    # Plots.plot!(plt, l_idt[:, 1], l_idt[:, 3], lw=1, yscale=:log10, label="base")
    # Plots.plot!(plt, l_idt[:, 1], l_idt[:, 4], lw=3, yscale=:log10, label="train")
    # # plot!(plt, [minimum(l_idt), maximum(l_idt)], [minimum(l_idt), maximum(l_idt)], label="y=x")
    # xlabel!(plt, "1000 / T [1/K]")
    # ylabel!(plt, "IDT [ms]")
    # xlims!(plt, (0.7, 1.7))
    # title!(plt, "20 bar phi=$(l_phi[i])")
    # plot!(plt, legend=:topleft, framestyle = :box, foreground_color_legend = nothing)
    # push!(l_plt, plt)

    writedlm(string(fig_path, "phi=$(l_phi[i])"), l_idt)
end


plt = plot(l_plt..., layout=(3, 1), size=(400, 1200))
png(plt, string(fig_path, "/regression_T"))
gr()
