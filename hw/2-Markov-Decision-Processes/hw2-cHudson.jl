# ASEN-5264 Homework 2 Collin Hudson 2/6/2024
using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states
using LinearAlgebra

############
# Question 3
############

function value_iteration(m,discount,sprc)
    # It is good to put performance-critical code in a function: https://docs.julialang.org/en/v1/manual/performance-tips/

    V = rand(length(states(m)))
    Vprime = rand(length(states(m)))
    R = reward_vectors(m)
    T = transition_matrices(m,sparse=sprc)
    # put your value iteration code here
    ep = 1e-7
    temp = Array{Float64, 2}(undef,length(states(m)),length(actions(m)))
    a = actions(m)
    while norm(V - Vprime,2) > ep
        V[:] = Vprime
        for j in axes(temp,2)
            temp[:,j] = R[a[j]] + discount*T[a[j]]*Vprime
        end
        Vprime[:] = maximum(temp, dims=2)
    end
    return V
end
V = value_iteration(grid_world,0.95,false)
# You can use the following commented code to display the value. If you are in an environment with multimedia capability (e.g. Jupyter, Pluto, VSCode, Juno), you can display the environment with the following commented code. From the REPL, you can use the ElectronDisplay package.
    # display(render(grid_world, color=reshape(V[1:100],(10,10))))
display(render(grid_world, color=V))

############
# Question 4
############

# You can create an mdp object representing the problem with the following:
m = UnresponsiveACASMDP(7)

# transition_matrices and reward_vectors work the same as for grid_world, however this problem is much larger, so you will have to exploit the structure of the problem. In particular, you may find the docstring of transition_matrices helpful:
display(@doc(transition_matrices))

V = value_iteration(m,0.99,true)

@show HW2.evaluate(V)

HW2.evaluate(V, "collin.hudson@colorado.edu")


########
# Extras
########

# The comments below are not needed for the homework, but may be helpful for interpreting the problems or getting a high score on the leaderboard.

# Both UnresponsiveACASMDP and grid_world implement the POMDPs.jl interface. You can find complete documentation here: https://juliapomdp.github.io/POMDPs.jl/stable/api/#Model-Functions

# To convert from physical states to indices in the transition function, use the stateindex function
# IMPORTANT NOTE: YOU ONLY NEED TO USE STATE INDICES FOR THIS ASSIGNMENT, using the states may help you make faster specialized code for the ACAS problem, but it is not required
# using POMDPs: states, stateindex

# s = first(states(m))
# @show si = stateindex(m, s)

# # To convert from a state index to a physical state in the ACAS MDP, use convert_s:
# using POMDPs: convert_s

# @show s = convert_s(ACASState, si, m)

# # To visualize a state in the ACAS MDP, use
# render(m, (s=s,))
