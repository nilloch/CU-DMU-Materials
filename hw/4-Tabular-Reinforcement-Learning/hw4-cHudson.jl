using DMUStudent.HW4: HW4, GridWorldEnv, RL
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using StaticArrays: SA, SVector
using Statistics: mean
using BenchmarkTools: @btime
using LinearAlgebra


# HW4.evaluate("collin.hudson@colorado.edu",time=true)
gwe = GridWorldEnv()

function solve!(gwe,T,R,gamma)
    policy = Dict{Tuple{S}, SVector}()
    policy' = Dict{Tuple{S}, SVector}()
    for s in states(gwe)
        policy[s] = rand(RL.actions(env))
        policy'[s] = rand(RL.actions(env))
    end
    while any(policy'[s] != policy[s] for s in states(gwe))
        policy = policy'
        Rp = Array{Float64, 1}(undef,length(states(gwe)))
        Tp = Array{Float64, 2}(undef,length(states(gwe)),length(states(gwe)))
        i = 1
        for s in states(gwe)
            Rp[i] = R[(s,policy[s])]
            j = 1
            for s' in states(gwe)
                Tp[i,j] = T[(s,policy[s],s')]
            end
            i += 1
        end
        Up = (I - gamma*Tp)\Rp
        for s in states(gwe)
            V = 0
            policy'[s] = argmax(R[(s,a)] + gamma*T)
        end
    end
end

function evalPolicy!(policy,s,gwe)
    if !haskey(policy, s)
        policy[s] = rand(RL.actions(env))
    end
    return policy[s]
end

function TMLMBRL(gwe,steps=100,evals=10,gamma=0.9)
    policy = Dict{Tuple{S}, SVector}()
    T = Dict{Tuple{S, A, S}, Float64}() #estimation of transition probability from s via a to s'
    R = Dict{Tuple{S, A}, Float64}()  #estimation of reward function for a at s
    N = Dict{Tuple{S, A, S}, Int}() #Dictionary for times went from s via a to s'
    rho = Dict{Tuple{S, A}, Float64}()  #cumulative reward for a at s
    s = RL.observe(gwe)
    for s in states(gwe)
        for a in actions(gwe)
            for s' in states(gwe)
                N[(s,a,s')] = 0
                rho[(s,a)] = 0
            end
        end
    end
    for k in 1:steps
        a = evalPolicy!(policy,s,gwe)
        r = RL.act!(gwe,a)
        N[(s,a,RL.observe(gwe))] += 1
        rho[(s,a)] += r
        tot = sum(values(N));
        for s in states(gwe)
            for a in actions(gwe)
                for s' in states(gwe)
                    R[(s,a)] += rho[(s,a)]/tot
                    T[(s,a,s')] += N[(s,a,s')]/tot
                end
            end
        end
        if(k%evals = 0)
            solve!(policy,gwe,T,R,gamma)
        end
        s = RL.observe(gwe)
    end

end