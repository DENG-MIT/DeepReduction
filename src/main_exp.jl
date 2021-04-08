include("header.jl")
include("simulator.jl")
include("visual.jl")
include("updateyaml.jl")

# include("naturalgas.jl")
gas = CreateSolution(mech);
ns = gas.n_species;
nr = gas.n_reactions;

include("expdataset.jl")

include("sensBVP.jl")

npr = 16;
p = zeros(npr * 3);
include("sensBVPpart.jl")

regression_plot(; max=5)

opt = ADAMW(1.e-3, (0.9, 0.999), 1.e-6);

grad_max = 10^(2.5);

include("callback.jl")

ind_sl = 17:nr

epochs = ProgressBar(iter:1000);
l_epoch = ones(n_exp);
grad_norm = ones(n_exp);
for epoch in epochs
    global p
    l_epoch .= 1.0
    grad_norm .= 1.0
    for i_exp in randperm(n_exp)
        T0 = conds[i_exp, 1]
        P = conds[i_exp, 2] * one_atm
        phi = conds[i_exp, 3]
        idt0 = conds[i_exp, 4]
        ts, pred = get_idt(T0, P, phi, p; dT=dT, tfinal=10.0)
        ngfull = length(ts)
        idt = ts[end]
        l_epoch[i_exp] = (log(idt / idt0))^2
        @printf("%d ng: %d (%d) idt %.2e idt0 %.2e \n",
                i_exp, length(ts), ngfull, idt, idt0)

        if (idt < 9.9) & (length(ts) < 2000) & (i_exp in l_exp_train)
            grad = 2 * log(idt / idt0) * sensBVP_mthreadp(ts, pred, p)
            grad_norm[i_exp] = norm(grad, 2)
            if grad_norm[i_exp] > grad_max
                @. grad = grad / grad_norm[i_exp] * grad_max
            end
            # grad[ind_sl] .= 0.0
            if (iter > 1)
                update!(opt, p, grad)
            end
        end
    end
    loss = mean(l_epoch[l_exp_train])
    loss_val = mean(l_epoch[l_exp_val])
    g_norm = mean(grad_norm[l_exp_train])
    set_description(
        epochs,
        string(
            @sprintf("loss %.2e val %.2e pnorm %.2f gnorm %.2f",
                loss, loss_val, norm(p), g_norm
            )
        ),
    )
    cb(p, loss, loss_val, g_norm; doplot=true)
end

# @load string(ckpt_path, "/../p.bson") p

@save string(ckpt_path, "/../p.bson") p

regression_plot(; max=100)
