# ASEN-5264 Homework 1 Collin Hudson 1/20/2024
import DMUStudent.HW1

function f(a, bs)
    #=
    INPUTS:
        a: a matrix (size not specified)
        bs: a non-empty vector of vectors
    RETURNS:
        vector of the elementwise maximum of the resulting vectors from the multiplication of a and the vectors in bs.
    METHODOLOGY:
        Assume a has size MxP, and bs contains N vectors of length P.
        1. Construct meta-matrix with size PxN from all arrays in bs using stack()
        2. Multiply stack(bs) by a to create an MxN matrix
        3. Get a column vector with the largest element of each row using maximum(matrix, dims=2)
        4. Convert to vector with vec() and return.
    =#
    return vec(maximum(a*stack(bs), dims=2))
end

# You can can test it yourself with inputs like this
a = [1.0 2.0; 3.0 4.0]
@show a
bs = [[1.0, 2.0], [3.0, 4.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "collin.hudson@colorado.edu")
