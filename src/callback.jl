
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

        plot_loss()

        @save string(ckpt_path, "/mymodel.bson") p opt l_loss l_loss_val l_grad l_pnorm iter
        if loss_val < l_min
            @save string(ckpt_path, "/model_$(length(l_loss)).bson") p opt l_loss l_loss_val l_grad l_pnorm iter
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

function load_model(epoch)
    @load string(ckpt_path, "/model_$(epoch).bson") p opt l_loss l_loss_val l_grad l_pnorm iter
end
