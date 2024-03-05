using DMUStudent.HW4: HW4, GridWorldEnv, RL
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using StaticArrays: SA, SVector
using Statistics: mean
using BenchmarkTools: @btime


# HW4.evaluate("collin.hudson@colorado.edu",time=true)
gwe = GridWorldEnv()

function evalPolicy(pi,s,gwe)
    if !haskey(pi, s)
        pi[s] = rand(actions(env))
    end
    return pi[s]
end

function TMLMBRL(gwe)
    gamma = 0.9
    pi = Dict{Tuple{S}, SVector}()
    T = Dict{Tuple{S, A, S}, Float64}() #estimation of transition probability from s via a to s'
    R = Dict{Tuple{S, A}, Float64}()  #estimation of reward function for a at s
    N = Dict{Tuple{S, A, S}, Int}() #Dictionary for times went from s via a to s'
    rho = Dict{Tuple{S, A}, Float64}()  #cumulative reward for a at s
    s = RL.observe(gwe)
    for k in 1:100
        a = evalPolicy(pi,s,gwe)
        r = RL.act!(gwe,a)
        N[(s,a,RL.observe(gwe))] += 1
        rho[(s,a,)] += r
        tot = sum(values(N));
        for s in states(gwe)
            for a in actions(gwe)
                for s' in states(gwe)
                    if !haskey(N, (s,a,s'))
                        T[(s,a,s')] += N[(s,a,s')]/tot
                    end
                    if !haskey(rho, (s,a))
                        R[(s,a)] += rho[(s,a)]/tot
                    end
                end
            end
        end
        if(k%10 = 0)
            solve!(pi,T,R,gamma)
        end
        s = RL.observe(gwe)
    end

end