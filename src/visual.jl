function regression_plot(;max=10)
    @printf("...regression plot ...\n")
    l_idt = zeros(minimum([max, n_exp]), 3)
    for i_exp in 1:minimum([max, n_exp])
        local T0 = conds[i_exp, 1]
        local P = conds[i_exp, 2] * one_atm
        local phi = conds[i_exp, 3]
        local idt0 = conds[i_exp, 4]
        idt = f_idt(T0, P, phi, p; dT=dT)
        if i_exp < max
            check_sol(T0, P, phi, p; i_exp=i_exp)
        end
        @printf("%d idt %.2e idt0 %.2e \n", i_exp, idt, idt0)
        _idt0 = f_idt(T0, P, phi, p .* 0.0; dT=dT)
        l_idt[i_exp, :] .= [idt0, idt, _idt0]
    end
    plt = Plots.scatter(l_idt[:, 1], l_idt[:, 2], xscale=:log10, yscale=:log10, label="train-ref")
    Plots.scatter!(plt, l_idt[:, 1], l_idt[:, 3], xscale=:log10, yscale=:log10, label="sk-ref")
    plot!(plt, [minimum(l_idt), maximum(l_idt)], [minimum(l_idt), maximum(l_idt)], label="y=x")
    xlabel!(plt, "ref [log-s]")
    ylabel!(plt, "train [log-s]")
    plot!(plt, legend=:topleft)
    png(plt, string(fig_path, "/regression"))
end

function check_sol(T0, P, phi, p; i_exp=0)
    l_plt = []
    ts, pred = get_idt(T0, P, phi, p; dT=dT)
    ts .+= 1.e-8
    plt = plot(ts, pred[end, :], lw=2, label="Train")
    ylabel!(plt, "Temperature [K]")
    xlabel!(plt, "Time [s]")
    title!(plt, @sprintf("%s, IDT=%.2e [s] \n @%.1f K, %.1f atm, phi=%.1f",
                fuel, ts[end], T0, P / one_atm, phi))
    push!(l_plt, plt)
    for s in [fuel, oxygen]
        plt =
            plot(ts,
            pred[species_index(gas, "$s"), :],
            lw=2, label="Train")
        ylabel!(plt, "Y $s")
        xlabel!(plt, "Time [s]")
        push!(l_plt, plt)
    end
    pltsum = plot(l_plt..., legend=false, framestyle=:box, xscale=:log10)

    png(pltsum, string(fig_path, "/conditions/sol_$i_exp"))
end

function plot_loss()
    plt_loss = plot(l_loss, yscale=:log10, label="train");
    plot!(plt_loss, l_loss_val, yscale=:log10, label="val");
    plt_grad = plot(l_grad, yscale=:log10, label="grad_norm");
    plt_pnorm = plot(l_pnorm .+ 1.e-8, yscale=:identity, label="p_norm");
    xlabel!(plt_loss, "Epoch");
    xlabel!(plt_grad, "Epoch");
    xlabel!(plt_pnorm, "Epoch");
    ylabel!(plt_loss, "Loss");
    ylabel!(plt_grad, "Grad Norm");
    ylabel!(plt_pnorm, "p Norm");
    plt_all = plot([plt_loss, plt_grad, plt_pnorm]..., legend=:bottomleft);
    png(plt_all, string(fig_path, "/loss_grad"));
end
