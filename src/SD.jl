using Base.Iterators: product, filter
using LinearAlgebra
using FastGaussQuadrature

"""
    A semidiscretization implements the f in 
    ẋ(t) = f(x) and serves as the input for time steppers
"""
abstract type AbstractSemidiscretization end

abstract type AbstractSemidiscretizationCell end

abstract type AbstractSemidiscretizationBasis end

abstract type AbstractSemidiscretizationQuadrature end

abstract type AbstractSemidiscretizationElliptic end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_Δs(sd::AbstractSemidiscretization)
    return sd.Δx, sd.Δt
end


"""
    jth 1-D basis function
"""
function φ_j(p,j,x_k,x) 

    n = p + 1

    f = 1.0
    for k in 1:n
        if k != j
            f *= ( x - x_k[k] ) / ( x_k[j] - x_k[k] )
        end
    end

    return f
end

"""
    derivative of jth 1-D basis function
"""
function Dφ_j(p,j,x_k,x) 
    
    f = 0.0
    for l in 1:p+1
        if l != j
            g = 1.0
            for k in 1:p+1
                if !(k in (l,j))
                    g *= ( x - x_k[k]  ) / ( x_k[j] - x_k[k] )
                end
            end
            g *= 1 / ( x_k[j] - x_k[l] )
            f +=  g
        end
    end

    return f
end

"
    Compute weight and evaluation points for Clenshaw Curtis quadrature
"
function clenshaw_curtis(nodes)

    if mod(nodes,2) == 0
        msg = "Clenshaw-Curtis cannot take even number of nodes."
        error(msg...)
    end

    N = nodes - 1

    x = [ cos(  k * pi / N  ) for k in N:-1:0  ]

    N_2 = Int(ceil(N/2))
    D = zeros(N_2+1,N_2+1)
    d = zeros(N_2+1)
    for n in 0:N_2 
        for k in 0:N_2
            D[n+1,k+1] = 2 * cos( 2 * n * k * pi / N ) / N
        end

        d[n+1] = 2 / ( 1 - 4*n^2 )
    end
    d[1] = 1
    D[1,:] .*= 0.5

    w = D*d

    w = append!( w , w[end-1:-1:1] )

    return x,w 
end

"
    Compute weight and evaluation points for trapezoid rule
"
function trapezoid(nodes)
    N = nodes - 1
    x =  collect(0:1:N) ./ N
    w = ones(nodes)
    w[1] /= 2
    w[end] /= 2
    return x,w 
end

"""
    Compute weight for Fisher-Rao inner product
"""
function quadrature_points(Δx,nodes;type="gausslegendre")

    if type == "gausslegendre"
        z,w = gausslegendre(nodes)
        x = ( Δx / 2 ) .* ( z .+ 1 )
        w *= Δx / 2
    elseif type == "clenshaw_curtis"
        z,w = clenshaw_curtis(nodes)
        x = ( Δx / 2 ) .* ( z .+ 1 )
        w *= Δx / 2
    elseif type == "trapezoid"
        x,w = trapezoid(nodes)
    end

    return x,w 
end

"""
    Select surface node indices 
"""

function surface_nodes(d,p)

    if d == 1
        dim_type = Tuple{UnitRange{Int64}}
    elseif d == 2
        dim_type = Tuple{UnitRange{Int64}, UnitRange{Int64}}
    elseif d == 3
        dim_type = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}
    end

    nodes_surf = Base.Iterators.ProductIterator{dim_type}[]

    for j in 1:d
        p_mod = vcat(p[1:j-1],[0],p[j+1:d])
        surf_idx = [ 1:c+1 for c in p_mod ]
        push!( nodes_surf , product(surf_idx...) )
        surf_idx[j] = p[j]+1:p[j]+1
        push!( nodes_surf , product(surf_idx...) )
    end

    data_type = Base.Iterators.ProductIterator{dim_type}
    
    return nodes_surf
end


"""
    Generate matrix containing left and right value in each node 
    given vector of values at interfaces
"""
function dg_interface_value(q)
    q = copy([ circshift(q,0) circshift(q,-1) ])
    return reshape(q',:,1) 
end

function dg_interface_value(p,q)
    m = Int( size(q,1) / p )
    s = zeros(p+1,m)
    s[1:p,:] = reshape(q,(p,m))
    s[p+1,:] = circshift(s[1,:],-1)
    return reshape(s,:,1) 
end

"""
    Computes log(a/b) / (a - b), 
    taylor approximation is used when a and b are sufficiently close 
    to avoid numerical error 
"""
function log_ab(a,b)
    y = b / a - 1
    if abs(y) > 1e-9
        return log(a/b)/(a-b)
    else
        return ( 1 - ( y / 2 ) + ( ( y ^ 2 ) / 3 ) - ( ( y ^ 3 ) / 4 ) + ( ( y ^ 4 ) / 5 ) - ( ( y ^ 5 ) / 6 ) ) / b
    end
end

"""

"""
function M_ab(a,b)

    x = a / b - 1
    if abs(x) > 1e-5
        Mab = - ( b^2 + 2*a*b*log(1+x) - a^2 )  /  ( (a - b)^3 ) 
        return Mab
    else
        return ( (1/b) - ( 2*a / (3*b^2) ) )
    end

end

"""

"""
function z0(a,b,h)

    y = b / a - 1

    if abs(y) > 1e-5
        return h * log( y + 1 ) / ( a * y )
    else
        return ( h / a ) *  ( 1 - ( y / 2 ) + ( y^2 / 3 ) - ( y^3 / 4 ) + ( y^4 / 5 ) - ( y^5 / 6 ) + ( y^6 / 7 ) )
    end

end

"""

"""
function z1(a,b,h)

    y = b / a - 1

    if abs(y) > 1e-3
        return h * ( y - log( y + 1 ) ) / ( a * y^2 )
    else
        return ( h / a ) *  ( ( 1 / 2 ) - ( y / 3 ) + ( y^2 / 4 ) - ( y^3 / 5 ) + ( y^4 / 6 ) - ( y^5 / 7 ) + ( y^6 / 8 ) )
    end

end

"""

"""
function z2(a,b,h)

    y = b / a - 1

    if abs(y) > 1e-2
        return h * ( y^2 - 2*y + 2*log( y + 1 ) ) / ( 2 * a * y^3 )
    else
        return ( h / a ) *  ( ( 1 / 3 ) - ( y / 4 ) + ( y^2 / 5 ) - ( y^3 / 6 ) + ( y^4 / 7 ) - ( y^5 / 8 ) + ( y^6 / 9 ) )
    end

end

"""

"""
function compute_conservation(sd::AbstractSemidiscretization,q)
    return reshape( sum( q , dims = (1,2) ) ./ 2 , (:) )
end



function initialize_spacial_array(d,m,p,Δx)
    x_cell = product( [ ( 0.0 : p[i] ) * ( Δx[i] / p[i] ) for i in 1:d ]... )  
    x_macro = product([ ( 0.0 : m[i]-1 ) * Δx[i] for i in 1:d ]...)
    x_combo = map( cell -> map( nodes -> nodes .+ cell , x_cell ) , x_macro )

    cell_size = size(x_cell)
    dims = size(x_cell) .* size(x_macro)

    xs = Array{Tuple{Vararg{Float64,d}}}(undef,dims...)

    for (idx,cell) in pairs(x_combo)
        end_idx = Tuple(idx) .* cell_size
        cell_range = [ j-i+1:j for (j,i) in zip(end_idx,cell_size) ]
        xs[cell_range...] = cell
    end
    
    return xs
end


function tensor_index_to_linear_index(index,dim,d)
    
    linear_index = index[1]

    for i in 2:d
        linear_index += ( index[i] - 1 ) * dim[i-1]
    end
    return linear_index
    
end

function linear_index_to_tensor_index(index,dims,d)
    multidim_index = Vector{Int}(undef, d)
    for i in d:-1:1
        multidim_index[i] = (index - 1) % dims[i] + 1
        index = div(index - 1, dims[i]) + 1
    end
    return multidim_index
end