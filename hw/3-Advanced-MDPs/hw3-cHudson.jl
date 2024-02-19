using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW3 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

Please make sure to update DMUStudent to gain access to the HW3 module.

=#

############
# Question 2
############

mrand = HW3.DenseGridWorld(seed=3)
m = HW3.DenseGridWorld()
function rollout(mdp, policy_function, s0, max_steps=100)
    # fill this in with code from the assignment document
    r_total = 0.0
    t = 0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(m,s)
        s, r = @gen(:sp,:r)(mdp, s, a)
        r_total += discount(m)^t*r
        t += 1
    end
    return r_total # replace this with the reward
end

function heuristic_policy(m, s)
    # closeness of x and y coords to a multiple of 20 
    xclose = modf(s[1]/20)[1]
    yclose = modf(s[2]/20)[1]
    if xclose > yclose
        if xclose >= 0.5 || s[1] < 20
            return :right
        else
            return :left
        end
    else
        if yclose >= 0.5 || s[2] < 20
            return :up
        else
            return :down
        end
    end
end

function randPolicy(m,s)
    return rand(actions(m))
end

maxRuns = 500
# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
resultsRand = [rollout(mrand, randPolicy, rand(initialstate(m))) for _ in 1:maxRuns]
@show meanRand = sum(resultsRand)/maxRuns
@show SEMRand = sqrt(sum(abs2,(resultsRand .- meanRand))/maxRuns^2)
results = [rollout(mrand, heuristic_policy, rand(initialstate(m))) for _ in 1:maxRuns]
@show meanRes = sum(results)/maxRuns
@show SEMRes = sqrt(sum(abs2,(results .- meanRes))/maxRuns^2)


############
# Question 3
############
#=
m = DenseGridWorld()

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

# This is an example state - it is a StaticArrays.SVector{2, Int}
s = SA[19,19]
@show typeof(s)
@assert s isa statetype(m)

# here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
q[(SA[1,1], :right)] = 0.0
q[(SA[2,1], :right)] = 0.0
n[(SA[1,1], :right)] = 1
n[(SA[2,1], :right)] = 0
t[(SA[1,1], :right, SA[2,1])] = 1

inbrowser(visualize_tree(q, n, t, SA[1,1]), "google-chrome")

############
# Question 4
############

# A starting point for the MCTS select_action function which can be used for Questions 4 and 5
function select_action(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()


    for _ in 1:1000
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        break # replace this with mcts iterations to fill n and q
    end

    # select a good action based on q and/or n

    return rand(actions(m)) # this dummy function returns a random action, but you should return your selected action
end

@btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

############
# Question 5
############

HW3.evaluate(select_action, "your.gradescope.email@colorado.edu")

# If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl

########
# Extras
########

# With a typical consumer operating system like Windows, OSX, or Linux, it is nearly impossible to ensure that your function *always* returns within 50ms. Do not worry if you get a few warnings about time exceeded.

# You may wish to call select_action once or twice before submitting it to evaluate to make sure that all parts of the function are precompiled.

# Instead of submitting a select_action function, you can alternatively submit a POMDPs.Solver object that will get 50ms of time to run solve(solver, m) to produce a POMDPs.Policy object that will be used for planning for each grid world.
=#