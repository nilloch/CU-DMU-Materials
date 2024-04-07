using DMUStudent.HW5: HW5, mc
using Plots: scatter, scatter!, plot, plot!
using QuickPOMDPs: QuickPOMDP
using StaticArrays: SA, SVector
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
using Flux
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

    # evaluate with a random policy
    function justWait(s)
        return :wait
    end
    policy = FunctionPolicy(o->justWait(o))
    sim = RolloutSimulator(max_steps=100)
    @show @time mean(POMDPs.simulate(sim, dragon, policy) for _ in 1:10_000)

    ############
    # Question 2
    ############

    # The notebook at https://github.com/zsunberg/CU-DMU-Materials/blob/master/notebooks/110-Neural-Networks.ipynb can serve as a starting point for this problem.
    n = 100
    dx = convert.(Float32,rand(n))
    dy = convert.(Float32,(1 .- dx).*sin.(20 .* log.(dx .+ 0.2)) + 0.1*randn(n));
    # display(scatter(dx, dy))
    sz = 128
    # m = Chain(Dense(1=>sz,sigmoid), Dense(sz=>sz,sigmoid), Dense(sz=>1))
    m = Chain(Dense(1=>100,sigmoid_fast), Dense(100=>100,sigmoid_fast), Dense(100=>100,sigmoid_fast), Dense(100=>1))
    loss(x, y) = Flux.mse(m(x), y)
    # loss(x, y) = sum((m(x)-y).^2)
    data = [(SVector(dx[i]), SVector(dy[i])) for i in 1:length(dx)]
    ploss = plot(label="loss")
    is = Vector{Int}()
    ys = Vector{Float64}()
    for i in 1:2000
        Flux.train!(loss, Flux.params(m), data, RMSProp(0.01))
        push!(ys,mean(loss.(SVector.(dx),SVector.(dy))))
        push!(is,i)
    end
    scatter!(ploss,is,ys,label="")
    p = plot(sort(dx), x->((1 - x)*sin.(20 * log(x + 0.2))), label="(1-x)*sin(20*log(x + 0.2))")
    plot!(p, sort(dx), first.(m.(SVector.(sort(dx)))), label="NN approximation")
    scatter!(p, dx, dy, label="data")
    display(p)
    display(ploss)

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
    #Training Parameters
    Q = Chain(Dense(2, 128, relu),
              Dense(128, length(actions(env))))
    Qp = deepcopy(Q)
    Qbest = deepcopy(Q)
    opt = Flux.setup(ADAM(0.0005), Q)
    bestEval = 0
    steps = 100
    intensity = 10
    episodes = 1000
    gamma = 0.99
    eps = 0.5
    #Make buffer
    reset!(env) 
    s = observe(env)
    a_ind = rand(eachindex(actions(env)))
    r = act!(env, actions(env)[a_ind])
    sp = observe(env)
    done = terminated(env)
    experience_tuple = (s, a_ind, r, sp, done)
    buffer = [experience_tuple]
    # lossMean = Vector{Float64}()
    rewardTot = Vector{Float64}()
    rewardMean = Vector{Float64}()
    # create your loss function for Q training here
    epSteps = 0
    for k in 1:episodes
        # if(!isempty(rewardTot) && last(rewardTot) > 20)
        #     deleteat!(buffer,(length(buffer) - epSteps):length(buffer))
        # end
        if k%10 == 0
            @show k
            Qp = deepcopy(Q)
            # deleteat!(buffer,1:200)
        end
        function loss(Q, s, a_ind, r, sp, done)
            if !done
                return (r + gamma*maximum(Qp(sp)) - Q(s)[a_ind])^2
            else
                return (r - Q(s)[a_ind])^2
                # return 0
            end
        end
        j = 1
        reset!(env)
        while j < steps && !done
            s = observe(env)
            if rand() <  max(eps/10, eps*(1-((k*steps + j)/100000)))
                a_ind = rand(eachindex(actions(env)))
            else
                a_ind = argmax(a->Q(s)[a],eachindex(actions(env)))
            end
            r = gamma^j*act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)
            experience_tuple = (s, a_ind, r, sp, done)
            push!(buffer,experience_tuple)
            j += 1
        end
        batchSize = intensity*steps
        data = rand(buffer, batchSize)
        Flux.Optimise.train!(loss, Q, data, opt)
        # tot = 0
        # totr = 0
        # for exp in data
        #     tot += loss(Q,exp[1],exp[2],exp[3],exp[4],exp[5])
        #     # totr += exp[3]
        # end
        # push!(lossMean,tot/batchSize)
        reset!(env)
        if k%10 == 0
            evalVal = HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))],n_episodes=250)[1]
            @show evalVal
            @show bestEval
            push!(rewardTot,evalVal)
            if evalVal > bestEval
                if evalVal >= 40
                    HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))],"collin.hudson@colorado.edu")
                end
                @show "copying Q->Qbest"
                Qbest = deepcopy(Q)
                bestEval = evalVal
            end
        end
        epSteps = j;
    end
    # ploss3 = plot(label="P3 loss per episode")
    # plot!(ploss3, 1:episodes, lossMean,label="mean loss per episode")
    # display(ploss3)
    rew = plot(label="rewards")
    # avrew = plot(label="avrew")
    # nPts = 1
    # for idx in eachindex(rewardTot)
    #     if idx%10 == 0
    #         push!(rewardMean,mean(rewardTot[idx-9:idx]))
    #         nPts += 1
    #     end
    # end
    # plot!(rew, 1:nPts, rewardMean, label="eval score per episode")
    plot!(rew, rewardTot, label="eval score per episode")
    display(rew)
    # plot!(avrew, rewardMean, label="average reward")
    # display(avrew)

    return Qbest
end

Q = dqn(env)

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))],"collin.hudson@colorado.edu") # you will need to remove the n_episodes=100 keyword argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------
#=
# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
display(render(env))

# The following code allows you to render the value function
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")

=#
# function render_value(value)
#     xs = -3.0:0.1:3.0
#     vs = -0.3:0.01:0.3

#     data = DataFrame(
#                      x = vec([x for x in xs, v in vs]),
#                      v = vec([v for x in xs, v in vs]),
#                      val = vec([value([x, v]) for x in xs, v in vs])
#     )

#     data |> @vlplot(:rect, "x:o", "v:o", color=:val, width="container", height="container")
# end

# display(render_value(s->maximum(Q(s))))
