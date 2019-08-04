#=
density:
- Julia version: 1.1.1
- Author: liu
- Date: 2019-08-04
=#

using CSV
using Distributions
using LinearAlgebra
using Random

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

struct Point
    x::Float64
    y::Float64
    close::Int
    far::Int
    close_neighbors::Array
    far_neighbors::Array
end

function Point(x::Float64, y::Float64, close::Int, far::Int)
    Point(x, y, close, far, [], [])
end

mutable struct Forest
    points::Vector{Point}
    tree::Tree
end

function Forest()
    Forest([], Tree())
end


function range_search(forest::Forest, x::Float64, y::Float64, radius::Float64)
    tree_range_search(forest.tree, x, y, radius)
    # point_start = Point(x - radius, y - radius, 0.0, 0.0)
    # point_end = Point(x + radius, y + radius, 0.0, 0.0)
    # x_start = searchsortedfirst(forest.x_index, point_start, by = point -> point.x)
    # x_end = searchsortedlast(forest.x_index, point_end, by = point -> point.x)
    # y_start = searchsortedfirst(forest.y_index, point_start, by = point -> point.y)
    # y_end = searchsortedlast(forest.y_index, point_end, by = point -> point.y)
    # # println(x_start)
    # # println(x_end)
    # # println(y_start)
    # # println(y_end)
    # x_set = Set{Point}(forest.x_index[x_start:x_end])
    # y_set = Set{Point}(forest.y_index[y_start:y_end])
    # intersect(x_set, y_set)
end

function initForest()
    forest = Forest()

    start = time_ns()
    println("Read CSV File.")
    csvFile = CSV.File("data_proj_414.csv")
    for row in csvFile
        point = Point(row.X, row.Y, row.Close, row.Far)
        push!(forest.points, point)
        # println("a=$(row.X), b=$(row.Y)")
    end
    println((time_ns() - start) / 1e9)

    start = time_ns()
    println("Shuffle the Points and Build 2D-Tree.")
    shuffle!(forest.points)
    for point in forest.points
        tree_insert(forest.tree, point.x, point.y, point)
    end
    println((time_ns() - start) / 1e9)

    # forest.x_index = sort(forest.points, by = point -> point.x)
    # forest.y_index = sort(forest.points, by = point -> point.y)

    start = time_ns()
    println("Calculate Neighbors and Density.")
    for point in forest.points
        point.close_neighbors = range_search(forest, point.x, point.y, 1.0)
        
    end
    println((time_ns() - start) / 1e9)

    forest
end


forest = initForest()

set = range_search(forest, 49.7029704654651, 50.4066946371814, 1.0)
points = Point[]
for point in set
    if point.close > 0
        push!(points, point)
    end
end

# init(forest)
