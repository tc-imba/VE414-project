#=
density:
- Julia version: 1.1.1
- Author: liu
- Date: 2019-08-04
=#

using CSV
# using Distributions
# using LinearAlgebra
using Random
# using Base.Threads
# using Distributed
# using SharedArrays
# addprocs(convert(Int, Sys.CPU_THREADS / 2))

# 2D-Tree Node of object T
mutable struct Node{V<:Real, T}
    x::V
    y::V
    left
    right
    object::T
end

function Node(x::Real, y::Real, object)
    Node(x, y, nothing, nothing, object)
end

mutable struct Tree
    root
end

function Tree()
    Tree(nothing)
end

function compare_node(a::Node, b::Node, div_x::Bool)
    div_x ? a.x < b.x : a.y < b.y
end

function tree_insert(tree::Tree, node::Node, root::Node, div_x::Bool)
    if node == nothing
        return
    end
    if compare_node(node, root, div_x)
        if root.left == nothing
            root.left = node
        else
            tree_insert(tree, node, root.left, !div_x)
        end
    else
        if root.right == nothing
            root.right = node
        else
            tree_insert(tree, node, root.right, !div_x)
        end
    end
end

function tree_insert(tree::Tree, x::Real, y::Real, object)
    node = Node(x, y, object)
    if tree.root == nothing
        tree.root = node
    else
        tree_insert(tree, node, tree.root, true)
    end
end

function tree_range_search(tree::Tree, x::Real, y::Real, r::Real,
    node, div_x::Bool, result::Array)
    if node == nothing
        return result
    end
    dist = (x - node.x) ^ 2 + (y - node.y) ^ 2
    if r ^ 2 >= dist
        push!(result, node.object)
    end
    delta = div_x ? x - node.x : y - node.y
    node1 = delta < 0 ? node.left : node.right;
    tree_range_search(tree, x, y, r, node1, !div_x, result);
    if delta < r
        node2 = delta < 0 ? node.right : node.left;
        tree_range_search(tree, x, y, r, node2, !div_x, result);
    end
    result
end

function tree_range_search(tree::Tree, x::Real, y::Real, r::Real)
    tree_range_search(tree, x, y, r, tree.root, true, [])
end

# tree = Tree()
# tree_insert(tree, 0, 0, 1)
# tree_insert(tree, 1, 1, 2)
# tree_insert(tree, -1, -1, 3)
# tree_insert(tree, 2, 2, 4)
#
# res = tree_range_search(tree, 0, 0, 3)

mutable struct Intersection
    a
    b
    area::Float64
    expectation::Float64
end

mutable struct Point
    i::Int
    x::Float64
    y::Float64
    close::Int
    far::Int
    close_neighbors::Dict{Int,Intersection}
    far_neighbors::Dict{Int,Intersection}
end

function Point(i::Int, x::Float64, y::Float64, close::Int, far::Int)
    Point(i, x, y, close, far, Dict{Int,Point}(), Dict{Int,Point}())
end

mutable struct Forest
    points::Array{Point}
    tree::Tree
end

function Forest()
    Forest([], Tree())
end

function Intersection(a::Point, ra::Float64, na::Int, b::Point, rb::Float64, nb::Int)
    d = sqrt((a.x-b.x)^2+(a.y-b.y)^2)
    # println([a.x, a.y, b.x, b.y, d])
    sa = pi * ra ^ 2
    sb = pi * rb ^ 2
    sc = sqrt((-d+ra-rb)*(-d-ra+rb)*(-d+ra+rb)*(d+ra+rb))
    x = convert(Int, min(na, nb))
    if x == 0
        return Intersection(a, b, sc, 0.0)
    end
    p = Array{Float64,1}(undef, x + 1)
    pa = sc / sa
    pb = sc / sb
    # println([d, pa, pb, sa, sb, sc])
    for i=0:x
        p[i+1] = binomial(na,i)*pa^i*(1-pa)^(na-i)*binomial(nb,i)*pb^i*(1-pb)^(nb-i)
    end
    result = 0
    for i=1:x
        result += i * p[i+1]
    end
    result /= sum(p)
    # println(p, result)
    Intersection(a, b, sc, result)
end


function range_search(forest::Forest, x::Float64, y::Float64, radius::Float64)
    tree_range_search(forest.tree, x, y, radius)
end

function forest_init()
    forest = Forest()

    start = time_ns()
    println("Read CSV File.")
    csvFile = CSV.File("data_proj_414.csv")
    for (i, row) in enumerate(csvFile)
        point = Point(i, row.X, row.Y, row.Close, row.Far)
        push!(forest.points, point)
        # println("a=$(row.X), b=$(row.Y)")
    end
    println((time_ns() - start) / 1e9)

    start = time_ns()
    println("Shuffle the Points and Build 2D-Tree.")
    for point in shuffle(forest.points)
        tree_insert(forest.tree, point.x, point.y, point)
    end
    println((time_ns() - start) / 1e9)

    # forest.x_index = sort(forest.points, by = point -> point.x)
    # forest.y_index = sort(forest.points, by = point -> point.y)

    start = time_ns()
    println("Calculate Neighbors and Density.")
    for pa in forest.points
        close_neighbors = range_search(forest, pa.x, pa.y, 1.0)
        for pb in close_neighbors
            if !haskey(pa.close_neighbors, pb.i) && pa.i != pb.i
                pa.close_neighbors[pb.i] = pb.close_neighbors[pa.i] =
                    Intersection(pa, 1.0, pa.close, pb, 1.0, pb.close)
            end
        end
    end
    println((time_ns() - start) / 1e9)

    return forest
end

mutable struct SamplePoint
    x::Float64
    y::Float64
end

function forest_sample(forest::Forest, skip_width::Float64)
    result = []
    grid_size = convert(Int, 107 / skip_width)

    for i=1:grid_size
        for j=1:grid_size
            pos_x = i * 107.0 / grid_size - skip_width / 2
            pos_y = j * 107.0 / grid_size - skip_width / 2
            set = range_search(forest, pos_x, pos_y, 1.0)
            num = length(set)
            density = 0
            if num == 1
                point = set[1]
                density = point.close / pi
            elseif num > 1
                for pa in set
                    for pb in set
                        if pa.i != pb.i && haskey(pa.close_neighbors, pb.i)
                            intersection = pa.close_neighbors[pb.i]
                            if intersection.area > 0
                                density += intersection.expectation / intersection.area
                            end
                        end
                    end
                end
                density /= (num * (num - 1))
            end
            if density > 0
                # println(i," ",j," ",density)
                prob = density * skip_width ^ 2
                base_tayes = floor(prob)
                rand_tayes = (prob - base_tayes > rand(1)[1]) ? 1 : 0
                for k=1:convert(Int, base_tayes+rand_tayes)
                    sample_point = SamplePoint(pos_x, pos_y)
                    push!(result, sample_point)
                end
            end
        end
    end
    result
end


forest = forest_init()

start = time_ns()
println("Sample the Tayes points.")
samples = forest_sample(forest, 0.5)
println((time_ns() - start) / 1e9)

@show num=length(samples)

open("samples.csv", "w") do f
    for sample_point in samples
        write(f, "$(sample_point.x), $(sample_point.y)\n")
    end
end
println("Results written to samples.csv.")
