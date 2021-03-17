
l_min = 1.e3;
l_loss = [];
l_loss_val = [];
l_grad = [];
l_pnorm = [];
iter = 1;
cb = function (p, loss, loss_val, g_norm; doplot=true)
    global l_loss, l_grad, iter
    push!(l_loss, loss)
    push!(l_loss_val, loss_val)
    push!(l_grad, g_norm)
    push!(l_pnorm, norm(p))

    if doplot & (iter % n_plot == 0)
        regression_plot(;max=10)

        plt_loss = plot(l_loss, yscale=:log10, label="train");
        plot!(plt_loss, l_loss_val, yscale=:log10, label="val");
        plt_grad = plot(l_grad, yscale=:log10, label="grad_norm");
        plt_pnorm = plot(l_pnorm, yscale=:log10, label="p_norm");
        xlabel!(plt_loss, "Epoch");
        xlabel!(plt_grad, "Epoch");
        xlabel!(plt_pnorm, "Epoch");
        ylabel!(plt_loss, "Loss");
        ylabel!(plt_grad, "Grad Norm");
        ylabel!(plt_pnorm, "p Norm");
        plt_all = plot([plt_loss, plt_grad, plt_pnorm]..., legend=:bottomleft);
        png(plt_all, string(fig_path, "/loss_grad"));

        @save string(ckpt_path, "/mymodel.bson") p opt l_loss l_loss_val l_grad l_pnorm iter
        if loss_val < l_min
            @save string(ckpt_path, "/modelmin.bson") p opt l_loss l_loss_val l_grad l_pnorm iter
            global l_min = loss_val
        end
    end
    iter += 1
    return false
end

if is_restart
    @load string(ckpt_path, "/mymodel.bson") p opt l_loss l_loss_val l_grad l_pnorm iter
    iter += 1
end
