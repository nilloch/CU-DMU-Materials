using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs
# Collin Hudson 3/18/2024 ASEN 5264 Homework 5
############
# Question 1
############

dragon = QuickPOMDP(
    states = [:h, :isc, :ic, :d], #healthy, in-situ-cancer, invasive-cancer, death
    actions = [:wait, :test, :treat],
    observations = [:pos, :neg],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition = function (s, a)
        if s == :h
            return SparseCat([:h, :isc], [0.98, 0.02])
        elseif s == :isc && a == :treat
            SparseCat([:h, :isc], [0.60, 0.40])
        elseif s == :isc && a != :treat
            SparseCat([:isc, :ic], [0.90, 0.10])
        elseif s == :ic && a == :treat
            SparseCat([:h, :d], [0.20, 0.80])
        elseif s == :ic && a != :treat
            SparseCat([:ic, :d], [0.40, 0.60])
        else
            return SparseCat([s], [1])
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation = function (s, a, sp)
        if a == :test
            if sp == :h
                return SparseCat([:pos, :neg], [0.05, 0.95])
            elseif sp == :isc
                return SparseCat([:pos, :neg], [0.80, 0.20])
            elseif sp == :ic
                return SparseCat([:pos], [1])
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
        elseif :a == wait
            return 1.0
        elseif :a == test
            return 0.8
        elseif :a == treat
            return 0.1
        end
    end,

    initialstate = :h,

    discount = 0.99
)

# evaluate with a random policy
policy = FunctionPolicy(o->return :wait)
sim = RolloutSimulator(max_steps=100)
# @show @time mean(POMDPs.simulate(sim, tiger, policy) for _ in 1:10_000)

############
# Question 2
############

# The notebook at https://github.com/zsunberg/CU-DMU-Materials/blob/master/notebooks/110-Neural-Networks.ipynb can serve as a starting point for this problem.

############
# Question 3
############

using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
# using VegaLite
# using ElectronDisplay # not needed if you're using a notebook or something that can display graphs
# using DataFrames: DataFrame

# The following are some basic components needed for DQN

# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )

function dqn(env)
    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 128, relu),
              Dense(128, length(actions(env))))

    # We can create 1 tuple of experience like this
    s = observe(env)
    a_ind = 1 # action index - the index, rather than the actual action itself, will be needed in the loss function
    r = act!(env, actions(env)[a_ind])
    sp = observe(env)
    done = terminated(env)

    experience_tuple = (s, a_ind, r, sp, done)

    # this container should work well for the experience buffer:
    buffer = [experience_tuple]
    # you will need to push more experience into it and randomly select data for training

    # create your loss function for Q training here
    function loss(Q, s, a_ind, r, sp, done)
        return (r-Q(s)[a_2])^ind # this is not correct! you need to replace it with the true Q-learning loss function
        # make sure to take care of cases when the problem has terminated correctly
    end

    # select some data from the buffer
    data = rand(buffer, 10)

    # do your training like this (you may have to adjust some things, and you will have to do this many times):
    Flux.Optimise.train!(loss, Q, data, Flux.setup(ADAM(0.0005), Q))

    # Make sure to evaluate, print, and plot often! You will want to save your best policy.
    
    return Q
end

Q = dqn(env)

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100) # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
display(render(env))

# The following code allows you to render the value function
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")


# function render_value(value)
#     xs = -3.0:0.1:3.0
#     vs = -0.3:0.01:0.3
# 
#     data = DataFrame(
#                      x = vec([x for x in xs, v in vs]),
#                      v = vec([v for x in xs, v in vs]),
#                      val = vec([value([x, v]) for x in xs, v in vs])
#     )
# 
#     data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
# end
# 
# display(render_value(s->maximum(Q(s))))