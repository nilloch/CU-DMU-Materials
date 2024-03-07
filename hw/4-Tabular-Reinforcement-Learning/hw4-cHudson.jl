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
sarsa_episodes = sarsa!(HW4.gw, n_episodes=10_000);

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

function policyGrad_episode!(thetas,env,alpha,eps=0.1)
    function policy(s)
        if rand() < eps
            return rand(RL.actions(env))
        else
            return RL.actions(env)[argmax(thetas[s])]
        end
    end
    function logGradSum(x,totr)

    end

    s = RL.observe(env)
    a = policy(s)
    r = RL.act!(env, a)
    sp = RL.observe(env)
    hist = [s]
    tau = [(s,a,r)]
    totR = 0
    while !RL.terminated(env)
        ap = policy(sp)

        s = sp
        a = ap
        r = RL.act!(env, a)
        totr += r
        sp = RL.observe(env)
        push!(tau, (s,a,r))
        push!(hist, sp)
    end
    # Update Thetas based on trajectory (tau)
    for x in tau
        theta = thetas[(x[1])][getindex(RL.actions(env),x[2])]
        thetas[x[1]][getindex(RL.actions(env),x[2])] = theta + alpha*(totr)/theta
        totr -= x[3]
    end
    return (hist=hist, theta = copy(theta), time=time()-start)
end

function policyGrad!(env, alpha = 0.2, n_episodes=100)
    episodes = []
    thetas = Dict((s) => SA{Float64}[0.0,0.0,0.0,0.0] for s in RL.observations(env))
    for i in 1:n_episodes
        RL.reset!(env)
        tau = policyGrad_episode!(thetas, env)
        push!(episodes, tau)
        for ele in hist

        end
    end
    
    return episodes
end






episodes = Dict("SARSA"=>sarsa_episodes)
plotEnv(HW4.gw,episodes)