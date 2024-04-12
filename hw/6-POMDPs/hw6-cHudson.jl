using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using POMDPPolicies: alphavectors
using LinearAlgebra
using Plots: plot, plot!
using BasicPOMCP
using DiscreteValueIteration
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
Z(m::POMDP, a, sp, o) = pdf(observation(up.m, a, sp), o)
T(m::POMDP, s, a, sp) = pdf(transition(up.m, s, a), sp)
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
    bp_vec ./= sum(bp_vec)
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
    return p.alpha_actions[argmax(idx->(p.alphas[idx]'*b.b),eachindex(p.alphas))]
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
    #Calculate Q (by replacing R)
    for key in keys(R)
        R[key] = R[key] + discount*T[key]*V
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
        push!(alphas,Q[a])
    end
    return HW6AlphaVectorPolicy(alphas, acts)
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

# @show mean(simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:5000)
# @show mean(simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000)
maxRuns = 5000
resultsQMDP = [simulate(RolloutSimulator(max_steps=500), m, qmdp_p, up) for _ in 1:maxRuns]
@show meanQMDP = sum(resultsQMDP)/maxRuns
@show SEMQMDP = sqrt(sum(abs2,(resultsQMDP .- meanQMDP))/maxRuns^2)
resultsSARSOP = [simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:maxRuns]
@show meanSARSOP = sum(resultsSARSOP)/maxRuns
@show SEMSARSOP = sqrt(sum(abs2,(resultsSARSOP .- meanSARSOP))/maxRuns^2)
# PLOT ALPHA VECTORS!!!!!!!
qmdpAlphas = plot(title="QMDP Alpha Vectors")
for j in eachindex(qmdp_p.alphas)
    plot!(qmdpAlphas,[0,1],qmdp_p.alphas[j],label=string("action: ",qmdp_p.alpha_actions[j]))
end
sarsopAlphas = plot(title="SARSOP Alpha Vectors")
for j in eachindex(alphavectors(sarsop_p))
    plot!(sarsopAlphas,[0,1],alphavectors(sarsop_p)[j],label=string("action: ",j))
end
display(qmdpAlphas)
display(sarsopAlphas)

###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(

    states = [:h, :isc, :ic, :d], #healthy, in-situ-cancer, invasive-cancer, death
    actions = [:wait, :test, :treat],
    observations = [:pos, :neg],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :h
            return SparseCat([:h, :isc], [0.98, 0.02])
        elseif s == :isc && a == :treat
            return SparseCat([:h, :isc], [0.60, 0.40])
        elseif s == :isc && a != :treat
            return SparseCat([:isc, :ic], [0.90, 0.10])
        elseif s == :ic && a == :treat
            return SparseCat([:h, :d], [0.20, 0.80])
        elseif s == :ic && a != :treat
            return SparseCat([:ic, :d], [0.40, 0.60])
        else
            return SparseCat([s], [1])
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (a, sp)
        if a == :test
            if sp == :h
                return SparseCat([:pos, :neg], [0.05, 0.95])
            elseif sp == :isc
                return SparseCat([:pos, :neg], [0.80, 0.20])
            elseif sp == :ic
                return SparseCat([:pos], [1])
            else
                return SparseCat([:neg], [1])
            end
        elseif a == :treat && (sp == :isc || sp == :ic)
            return SparseCat([:pos], [1])
        else
            return SparseCat([:neg], [1])
        end
    end,

    reward = function (s, a)
        if s == :d
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        else
            return 0.0
        end
    end,

    initialstate = SparseCat([:h], [1]),

    discount = 0.99
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)

    # Fill in your heuristic policy here
    # Use pdf(b, s) to get the probability of a state
    # @show pdf(b,:isc)
    if pdf(b,:isc) > 0.1
        return :treat
    elseif pdf(b,:ic) > 0.9
        if rand() < 0.5
            return :treat
        else
            return :wait
        end
    else
        if rand() < 0.8
            return :test
        else
            return :wait
        end
    end
end
)
# @show mean(simulate(RolloutSimulator(max_steps=1000), cancer, qmdp_p, up) for _ in 1:1000)     # Should be approximately 66
# @show mean(simulate(RolloutSimulator(max_steps=1000), cancer, heuristic, up) for _ in 1:1000)
# @show mean(simulate(RolloutSimulator(max_steps=1000), cancer, sarsop_p, up) for _ in 1:1000)   # Should be approximately 79
maxRuns = 5000
maxSteps = 1000
resultsQMDP = [simulate(RolloutSimulator(max_steps=maxSteps), cancer, qmdp_p, up) for _ in 1:maxRuns]
@show meanQMDP = sum(resultsQMDP)/maxRuns
@show SEMQMDP = sqrt(sum(abs2,(resultsQMDP .- meanQMDP))/maxRuns^2)
resultsHeuristic = [simulate(RolloutSimulator(max_steps=maxSteps), cancer, heuristic, up) for _ in 1:maxRuns]
@show meanHeuristic = sum(resultsHeuristic)/maxRuns
@show SEMHeuristic = sqrt(sum(abs2,(resultsHeuristic .- meanHeuristic))/maxRuns^2)
resultsSARSOP = [simulate(RolloutSimulator(max_steps=maxSteps), cancer, sarsop_p, up) for _ in 1:maxRuns]
@show meanSARSOP = sum(resultsSARSOP)/maxRuns
@show SEMSARSOP = sqrt(sum(abs2,(resultsSARSOP .- meanSARSOP))/maxRuns^2)
#####################
# Problem 3: LaserTag
#####################

m = LaserTagPOMDP()

qmdp_p = qmdp_solve(m)
up = DiscreteUpdater(m) # you may want to replace this with your updater to test it

# Use this version with only 100 episodes to check how well you are doing quickly
# @show HW6.evaluate((qmdp_p, up), n_episodes=100)

# A good approach to try is POMCP, implemented in the BasicPOMCP.jl package:
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
    solver = POMCPSolver(tree_queries=100,
        c=10.0,
        default_action=rand(actions(m)),
        estimate_value=FORollout(ValueIterationSolver()))
    return solve(solver, m)
end
pomcp_p = pomcp_solve(m)

@show HW6.evaluate((pomcp_p, up), n_episodes=100)

# When you get ready to submit, use this version with the full 1000 episodes
# HW6.evaluate((pomcp_p, up), "collin.hudson@colorado.edu")


#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
# using POMDPGifs
# import Cairo, Fontconfig # needed to display properly

# makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# # You can render a single frame like this
# using POMDPTools: stepthrough, render
# using Compose: draw, PNG

# history = []
# for step in stepthrough(m, qmdp_p, up, max_steps=10)
#     push!(history, step)
# end
# displayable_object = render(m, last(history))
# # display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
# draw(PNG("lasertag.png"), displayable_object)
