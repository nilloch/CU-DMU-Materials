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
function sarsa_episode!(Q, env; eps=0.05, gamma=0.99, alpha=0.2)
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

function plotEnv(env,episodes,stop=50000)
    p = plot(xlabel="steps in environment", ylabel="avg return")
    n = 1000
    for (name, eps) in episodes
        if(name == "SARSA")
            Q = Dict((s, a) => 0.0 for s in RL.observations(env), a in RL.actions(env))
            xs = [0.0]
            ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env))))]
            for i in n:n:min(stop, length(eps))
                newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
                push!(xs, last(xs) + newsteps)
                Q = eps[i].Q
                push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env)))))
            end
        else
            xs = [0.0]
            thetas = Dict((s) => 420*ones(4) for s in RL.observations(env))
            ys = [mean(evaluate(env, s->eps[1].policy(s,thetas)))]
            for i in n:n:min(stop, length(eps))
                newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
                push!(xs, last(xs) + newsteps)
                thetas = eps[i].thetas
                push!(ys, mean(evaluate(env, s->eps[i].policy(s,thetas))))
            end
        end    
        plot!(p, xs, ys, label=name)
    end
    display(p)
end
function plotEnvTime(env,episodes,stop=50000)
    p2 = plot(xlabel="time in environment", ylabel="avg return")
    n = 1000
    for (name, eps) in episodes
        if(name == "SARSA")
            Q = Dict((s, a) => 0.0 for s in RL.observations(env), a in RL.actions(env))
            xs = [0.0]
            ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env))))]
            for i in n:n:min(stop, length(eps))
                newsteps = sum(length(ep.time) for ep in eps[i-n+1:i])
                push!(xs, last(xs) + newsteps)
                Q = eps[i].Q
                push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], RL.actions(env)))))
            end
        else
            xs = [0.0]
            thetas = Dict((s) => 420*ones(4) for s in RL.observations(env))
            ys = [mean(evaluate(env, s->eps[1].policy(s,thetas)))]
            for i in n:n:min(stop, length(eps))
                newsteps = sum(length(ep.time) for ep in eps[i-n+1:i])
                push!(xs, last(xs) + newsteps)
                thetas = eps[i].thetas
                push!(ys, mean(evaluate(env, s->eps[i].policy(s,thetas))))
            end
        end    
        plot!(p2, xs, ys, label=name)
    end
    display(p2)
end
function getAIdx(a)
    idx = 1
    if(a[1] == 0) || (a[1] == -1)
        idx += 1
    end
    if(a[2] == -1)
        idx += 2
    elseif a[2] == 1
        idx += 1
    end
    return idx
end
function sampleAction(prob)
    sample = rand()
    if(sample <= prob[1])
        return 1
    elseif(sample <= (prob[1] + prob[2]))
        return 2
    elseif(sample <= (prob[1] + prob[2] + prob[3]))
        return 3
    else(sample <= (prob[1] + prob[2] + prob[3] + prob[4]))
        return 4
    end
end
function policyGrad_episode!(thetas,env,alpha,eps=0.1)
    function policy(s,thetas)
        softMax = exp.(thetas[s] .+ eps)/sum(exp.(thetas[s] .+ eps))
        return RL.actions(env)[sampleAction(softMax)]
    end
    function policyEps(s)
            return policy(s,thetas)
    end
    
    # function pgFunc(x)
    #     idx = getAIdx(x[2])
    #     if(idx == 1)
    #         return [1/thetas[x[1]][idx],0.0,0.0,0.0]
    #     elseif(idx == 2)
    #         return [0.0,1/thetas[x[1]][idx],0.0,0.0]
    #     elseif(idx == 3)
    #         return [0.0,0.0,1/thetas[x[1]][idx],0.0]
    #     elseif(idx == 4)
    #         return [0.0,0.0,0.0,1/thetas[x[1]][idx]]
    #     end
    # end #hahaha

    function pgFunc(x)
        a = getAIdx(x[2])
        if a == 1
            gradPolicy = [1 - exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]]))]
        elseif a == 2
            gradPolicy = [-exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), 1 - exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]]))]
        elseif a == 3
            gradPolicy = [-exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), - exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), 1 - exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]]))]
        elseif a == 4
            gradPolicy = [-exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), -exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]])), 1 - exp(thetas[x[1]][a])/sum(exp.(thetas[x[1]]))]
        else
            throw(error("not a valid action"))
        end
        return gradPolicy
    end

    start = time()
    hist = []
    thetasBatch = Dict((s) => zeros(4) for s in RL.observations(env))
    for batch in 1:1
        s = RL.observe(env)
        a = policyEps(s)
        r = RL.act!(env, a)
        sp = RL.observe(env)
        # hist = [s]
        push!(hist, s)
        tau = [(s,a,r)]
        rtogo = 0
        while !RL.terminated(env)
            ap = policyEps(sp)
            s = sp
            a = ap
            r = RL.act!(env, a)
            rtogo += r
            sp = RL.observe(env)
            push!(tau, (s,a,r))
            push!(hist, sp)
        end
        # Update Thetas based on trajectory (tau)
        for x in tau
            thetasBatch[x[1]] += alpha*pgFunc(x)*rtogo/20
            rtogo -= x[3]
        end
    end
    mergewith!(+,thetas,thetasBatch)
    return (hist=hist, thetas = copy(thetas), time=time()-start, policy = policy)
end

function policyGrad!(env, alpha = 0.2, eps = 0.05, n_episodes=100)
    episodes = []
    thetas = Dict((s) => 0.5*ones(4) for s in RL.observations(env))
    for i in 1:n_episodes
        RL.reset!(env)
        tau = policyGrad_episode!(thetas, env, max(0.025, alpha*(1-i/(n_episodes))), max(eps, 1-i/(n_episodes)))
        push!(episodes, tau)
    end
    
    return episodes
end



numEps = 150_000
# sarsa_episodes = sarsa!(HW4.gw, n_episodes=numEps)
# episodes = Dict("SARSA"=>sarsa_episodes)
policyGrad_episodes = policyGrad!(HW4.gw,0.1,0.05,numEps)
episodes = Dict("Policy Gradient"=>policyGrad_episodes)
# episodes = Dict("SARSA"=>sarsa_episodes, "Policy Gradient"=>policyGrad_episodes)
plotEnv(HW4.gw,episodes,numEps)
plotEnvTime(HW4.gw,episodes,numEps)