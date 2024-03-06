using DMUStudent.HW4: HW4, GridWorldEnv, RL
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using StaticArrays: SA, SVector
using Statistics: mean
using BenchmarkTools: @btime
using LinearAlgebra
using Plots
import POMDPTools


# HW4.evaluate("collin.hudson@colorado.edu",time=true)

# Copied from the SARSA Jupyter Notebook
function sarsa_episode!(Q, env; eps=0.1, gamma=0.99, alpha=0.1)
    start = time()
    
    function policy(s)
        if rand() < eps
            return rand(RL.actions(env))
        else
            return argmax(a->Q[(s, a)], RL.actions(env))
        end
    end

    s = RL.observe(env)
    a = policy(s)
    r = RL.act!(env, a)
    sp = RL.observe(env)
    hist = [s]

    while !RL.terminated(env)
        ap = policy(sp)

        Q[(s,a)] += alpha*(r + gamma*Q[(sp, ap)] - Q[(s, a)])

        s = sp
        a = ap
        r = RL.act!(env, a)
        sp = RL.observe(env)
        push!(hist, sp)
    end

    Q[(s,a)] += alpha*(r - Q[(s, a)])

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function sarsa!(env; n_episodes=100)
    Q = Dict((s, a) => 0.0 for s in RL.observations(env), a in RL.actions(env))
    episodes = []
    
    for i in 1:n_episodes
        RL.reset!(env)
        push!(episodes, sarsa_episode!(Q, env, eps=max(0.1, 1-i/n_episodes)))
    end
    
    return episodes
end
sarsa_episodes = sarsa!(HW4.gw, n_episodes=100_000);

# @manipulate for episode in 1:length(sarsa_episodes), step in 1:maximum(ep->length(ep.hist), sarsa_episodes)
#     ep = sarsa_episodes[episode]
#     i = min(step, length(ep.hist))
#     POMDPTools.render(m, (s=ep.hist[i],), color=s->maximum(map(a->ep.Q[(s,a)], RL.actions(env))))
# end

function evaluate(env, policy, n_episodes=1000, max_steps=1000, gamma=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        RL.reset!(env)
        s = RL.observe(env)
        while !RL.terminated(env)
            a = policy(s)
            r += gamma^t*RL.act!(env, a)
            s = RL.observe(env)
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

function plotEnv(env,episodes)
    p = plot(xlabel="steps in environment", ylabel="avg return")
    n = 20
    stop = 1000
    for (name, eps) in episodes
        Q = Dict((s, a) => 0.0 for s in RL.observations(env), a in RL.actions(env))
        xs = [0]
        ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env))))]
        for i in n:n:min(stop, length(eps))
            newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
            push!(xs, last(xs) + newsteps)
            Q = eps[i].Q
            push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env)))))
        end    
        plot!(p, xs, ys, label=name)
    end
    p
end
episodes = Dict("SARSA"=>sarsa_episodes)
plotEnv(HW4.gw,episodes)