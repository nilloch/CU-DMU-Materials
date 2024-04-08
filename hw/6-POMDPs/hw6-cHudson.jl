using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using LinearAlgebra
# Collin Hudson 4/7/2024 Homework 6
##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
    
end

# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# this function may be helpful to get the belief as a vector in stateindex order
beliefvec(b::DiscreteBelief) = b.b 
function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    bp_vec = zeros(length(states(up.m)))
    idxSp = 1
    for sp in ordered_states(up.m)
        sumPred = 0
        idxS = 1
        for s in ordered_states(up.m)
            sumPred += T(up.m, s, a, sp)*b.b[idxS]
            idxS += 1
        end
        bp_vec[idxSp] = Z(m::POMDP, a, sp, o)*sumPred
        idxSp += 1
    end
    # Normalize belief vector
    bp_vec = bp_vec./sum(bp_vec)
    return DiscreteBelief(up.m, bp_vec)
end
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    idx = 1
    for s in ordered_states(up.m)
        b_vec[idx] = pdf(distribution, s)
        idx += 1
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    # choose action based on alpha vectors
    return p.alpha_actions(argmax(idx->(p.alphas(idx)'*b.b)))
end

#------
# QMDP
#------

function value_iteration(m,discount,sprc)
    V = rand(length(states(m)))
    Vprime = rand(length(states(m)))
    R = reward_vectors(m)
    T = transition_matrices(m,sparse=sprc)
    # put your value iteration code here
    ep = 1e-7
    temp = Array{Float64, 2}(undef,length(states(m)),length(actions(m)))
    a = ordered_actions(m)
    while norm(V - Vprime,2) > ep
        V[:] = Vprime
        for j in axes(temp,2)
            temp[:,j] = R[a[j]] + discount*T[a[j]]*Vprime
        end
        Vprime[:] = maximum(temp, dims=2)
    end
    for key in keys(R)
        R[key] = R[key] + discount*V
    end
    return R
end

function qmdp_solve(m, discount=discount(m))

    # Fill in Value Iteration to compute the Q-values
    Q = value_iteration(m,discount,true)

    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    for a in ordered_actions(m)
        push!(acts,a)
        # @show typeof(Q[a])
        # @show typeof(alphas)
        push!(alphas,Q[a])
        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

@show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
@show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)

###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(

    # Fill in your actual code from last homework here

    states = [:healthy, :in_situ, :invasive, :death],
    actions = [:wait, :test, :treat],
    observations = [true, false],
    transition = (s, a) -> Deterministic(s),
    observation = (a, sp) -> Deterministic(false),
    reward = (s, a) -> 0.0,
    discount = 0.99,
    initialstate = Deterministic(:death),
    isterminal = s->s==:death,
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

                               # Fill in your heuristic policy here
                               # Use pdf(b, s) to get the probability of a state

                               return :wait
                           end
                          )

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79

#####################
# Problem 3: LaserTag
#####################

m = LaserTagPOMDP()

qmdp_p = qmdp_solve(m)
up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
@show HW6.evaluate((qmdp_p, up), n_episodes=100)

# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
using BasicPOMCP
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
    solver = POMCPSolver(tree_queries=10,
                         c=1.0,
                         default_action=first(actions(m)),
                         estimate_value=FORollout(FunctionPolicy(s->rand(actions(m)))))
    return solve(solver, m)
end
pomcp_p = pomcp_solve(m)

@show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((qmdp_p, up), "REPLACE_WITH_YOUR_EMAIL@colorado.edu")

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly

makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, qmdp_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("lasertag.png"), displayable_object)
