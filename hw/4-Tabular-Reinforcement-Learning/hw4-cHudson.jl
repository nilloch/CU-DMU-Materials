using DMUStudent.HW4: HW4, DenseGridWorld
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using StaticArrays: SA
using Statistics: mean
using BenchmarkTools: @btime


# HW4.evaluate("collin.hudson@colorado.edu",time=true)
