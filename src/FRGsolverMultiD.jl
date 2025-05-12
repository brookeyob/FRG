#Fisher-Rao Discontinuous Galerkin for Transport Equation
"""
    A semidiscretization specific to Transport equation
"""
abstract type AbstractTransportSemidiscretization<:AbstractSemidiscretization end

abstract type AbstractTransportSemidiscretizationBasis<:AbstractSemidiscretizationBasis end

abstract type AbstractTransportSemidiscretizationQuadrature<:AbstractSemidiscretizationQuadrature end

abstract type AbstractTransportSemidiscretizationCell<:AbstractSemidiscretizationCell end


"""
    Fisher-Rao DG semidiscretization for Transport equation
"""
struct FRGSemidiscretizationDGMultiD{dim_type}<:AbstractTransportSemidiscretization
    # Order of polynomial function
    p::Vector{Int64}
    # Dimension polynomial function
    d::Int64
    # Spatial discretization size
    Δx::Vector{Float64}
    # Temporal discretization size
    Δt::Float64
    # Number of nodes
    m::Vector{Int64}
    # Storage for the numerical fluxes computed 
    f::Vector{Vector{Vector{Float64}}}
    # Flux type
    f_type::String
    # Include Fisher-Rao Weighting
    FR_weight::Bool
    # Number of points in dimension
    n_k::Vector{Int64}
    # Node points in cell
    x_k::Vector{Vector{Float64}}
    # Number of quadrature points
    n_q::Vector{Int64}
    # quadrature points
    x_q::Vector{Vector{Float64}}
    # quadrature weights
    w::Vector{Vector{Float64}}
    # Index of cells
    cell_idx::Base.Iterators.ProductIterator{dim_type} 
    # Number of node points in cell
    num_nodes::Int64
    # Cell node points index
    nodes::Base.Iterators.ProductIterator{dim_type} 
    # Cell surface node points index
    nodes_surf::Vector{Base.Iterators.ProductIterator{dim_type}} 
    # Cell quadrature node points index
    nodes_q::Base.Iterators.ProductIterator{dim_type} 
    # Cell surface quadrature node points index
    nodes_q_surf::Vector{Base.Iterators.ProductIterator{dim_type}} 
    # Surface index
    s::Vector{Int64}
end

"""
    Basis function information
"""
struct FRGSemidiscretizationMultiDBasis{dim_type}<:AbstractTransportSemidiscretizationBasis
    # Order of polynomial function
    p::Vector{Int64} # [ d ]
    # Dimension polynomial function
    d::Int64
    # Spatial discretization size
    Δx::Vector{Float64}
    # Number of points in dimension
    n_k::Vector{Int64} # [ d ]
    # Node points in cell
    x_k::Vector{Vector{Float64}} # [ d x n ]
    # Number of node points in cell
    num_nodes::Int64
    # Cell node points index
    nodes::Base.Iterators.ProductIterator{dim_type} # [ n^d ]
    # Cell surface node points index
    nodes_surf::Vector{Base.Iterators.ProductIterator{dim_type}} # [ 2d x n_q^(d-1) ]
    # Surface index
    s::Vector{Int64} # [ 2d ]
end

"""
    quadrature information
"""
struct FRGSemidiscretizationMultiDQuadrature{dim_type}<:AbstractTransportSemidiscretizationQuadrature
    # Dimension polynomial function
    d::Int64
    # Cell quadrature node points index
    nodes_q::Base.Iterators.ProductIterator{dim_type} # [ n_q^d ]
    # Cell surface quadrature node points index
    nodes_q_surf::Vector{Base.Iterators.ProductIterator{dim_type}} # [ 2d x n_q^(d-1) ]
    # Number of quadrature points
    n_q::Vector{Int64} # [ d ]
    # quadrature points
    x_q::Vector{Vector{Float64}} # [ d x n_q ]
    # quadrature weights
    w::Vector{Vector{Float64}} # [ d x n_q ]
    # Surface index
    s::Vector{Int64} # [ 2d ]
end

"""
    Fisher-Rao DG semidiscretization cell for Transport equation
"""
struct FRGSemidiscretizationMultiDCell{dim_type}<:AbstractTransportSemidiscretizationCell
    # Mass Matrix
    M::Matrix{Float64} # [ n^d , n^d ]
    # Stiffness Matrix
    K::Matrix{Float64} # [ n^d , n^d ]
    # Cell Flux
    f::Vector{Float64} # [ n^d ]

    # Temporarily store Mass Matrix integrand
    M_scratch::Matrix{Float64} # [ n^d , n^d ]
    # Temporarily store Stiffness Matrix integrand
    K_scratch::Matrix{Float64} # [ n^d , n^d ]
    # Temporarily store flux vector integrand
    F_scratch::Vector{Float64} # [ n^d , n^d ]


    # Cell node density
    r::Vector{Float64} # [ n ]
    # Cell node velocity
    u::Vector{Vector{Float64}} # [ n x d ]
    # Cell numerical flux
    f_hat::Vector{Vector{Float64}} # [ 2d x n ]

    # Include Fisher-Rao weighting
    FR_weight::Bool
end

"""
    jth 1-D basis function
"""
function φ_j(   basis::FRGSemidiscretizationMultiDBasis,
                i,
                j,
                x   ) 
                
    # Extract the node points and number of points in the i-th dimension
    x_k = basis.x_k[i]
    n = basis.n_k[i]

    # Filter out the j-th node to compute the Lagrange polynomial
    iter = Iterators.filter(k -> k != j, 1:n)

    f = 1.0

    # Compute the Lagrange polynomial for the j-th basis function
    for k in iter
        f *= ( x - x_k[k] ) / ( x_k[j] - x_k[k] )
    end

    return f
end

"""
    derivative of jth 1-D basis function
"""
function Dφ_j(  basis::FRGSemidiscretizationMultiDBasis,
                i,
                j, 
                x   ) 

    # Extract the node points and number of points in the i-th dimension
    x_k = basis.x_k[i]
    n = basis.n_k[i]

    # Filter out the j-th node for the outer summation
    iter_outer = Iterators.filter( l -> l != j , 1:n )

    f = 0.0
 
    # Compute the derivative of the Lagrange polynomial
    for l in iter_outer
        # Filter out the l-th and j-th nodes for the inner product
        iter_inner = Iterators.filter( k -> k ∉ (l,j) , 1:n )
        g = 1.0
        
        for k in iter_inner
            g *= ( x - x_k[k]  ) / ( x_k[j] - x_k[k] )
        end
        
        g *= 1 / ( x_k[j] - x_k[l] )
        f +=  g
    end

    return f
end

"""
    jth d-D basis function
"""
function L_j(   basis::FRGSemidiscretizationMultiDBasis,
                j_idx,
                x   ) 

    d = basis.d

    f = 1.0

    # Compute the product of 1-D basis functions for each dimension
    for i in 1:d
        f *= φ_j(basis,i,j_idx[i],x[i])
    end

    return f
end

"""
    derivative of jth basis function wrt x_l
"""
function DxL_j( basis::FRGSemidiscretizationMultiDBasis,
                j_idx,
                l,
                x   )
    # Extract the dimension
    d = basis.d

    f = 1.0
    # Compute the derivative of the d-D basis function with respect to x_l
    for i in 1:d
        if i != l
            f *= φ_j(basis,i,j_idx[i],x[i])
        elseif i == l
            f *= Dφ_j(basis,i,j_idx[i],x[i])
        end
    end
    
    return f
end

"""
    product of ith and jth basis function
"""
function LL_ij( basis::FRGSemidiscretizationMultiDBasis,
                ij,
                x   )
    # Compute the product of the i-th and j-th d-D basis functions
    i,j = ij
    return L_j(basis,i,x) * L_j(basis,j,x) 
end

"""
    inner product between f, and ith & jth basis function
"""
function LLf_ij(    basis::FRGSemidiscretizationMultiDBasis,
                    ij,
                    f_j,
                    b,
                    x   )
    # Compute the inner product between f and the i-th and j-th basis functions
    s = basis.s[b]
    return sign(s) * f_j * LL_ij(basis,ij,x)
end

"""
    product of ith, jth & kth basis function
"""
function LLL_ijk(   basis::FRGSemidiscretizationMultiDBasis,
                    ijk,
                    x   )
    # Compute the product of the i-th, j-th, and k-th d-D basis functions
    i,j,k = ijk
    return L_j(basis,i,x) * L_j(basis,j,x) * L_j(basis,k,x) 
end

"""
    inner product between u, and ith, jth & kth basis function
"""
function LLLu_ijk(  basis::FRGSemidiscretizationMultiDBasis,
                    ijk,
                    u_k,
                    b,
                    x   )
    s = basis.s[b]
    return sign(s) * u_k[abs(s)] * LLL_ijk(basis,ijk,x)
end

"""
    product of ith, jth basis function & derivative of kth basis function
"""
function LLDxL_ijk( basis::FRGSemidiscretizationMultiDBasis,
                    ijk,
                    l,
                    x   )
    # Compute the product of the i-th, j-th basis functions and the derivative of the k-th basis function
    i,j,k = ijk
    return L_j(basis,i,x) * L_j(basis,j,x) * DxL_j(basis,k,l,x)
end

"""
    inner product between u, and ith, jth basis function & derivative of kth basis function
"""
function LLDxLu_ijk(    basis::FRGSemidiscretizationMultiDBasis,
                        ijk,
                        u_l,
                        x   )
    # Compute the inner product between u and the i-th, j-th basis functions and the derivative of the k-th basis function
    f = 0.0
    d = basis.d
    for l in 1:d
        f += u_l[l] * LLDxL_ijk(basis,ijk,l,x)
    end
    return f
end

"""
    Compute weight for Fisher-Rao inner product
"""
function compute_FR_weight( cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis,
                            x   )
    # Compute the Fisher-Rao weight based on the cell density
    r = cell.r
    nodes = basis.nodes
    FR_weight = cell.FR_weight

    if FR_weight
        g = 0.0
        for (i,i_idx) in enumerate(nodes)
            g += r[i] * L_j(basis,i_idx,x)
        end
        return 1.0 / g
    else
        return 1.0
    end
end


"""
    Integrand of Fisher-Rao DG mass matrix for Transport equation
"""
function FRG_compute_M_integrand!(  cell::FRGSemidiscretizationMultiDCell,
                                    basis::FRGSemidiscretizationMultiDBasis,
                                    x   )

    # Compute the Fisher-Rao weight at the given point x
    FR_weight = compute_FR_weight(cell,basis,x)

    # Extract the basis function nodes and initialize the scratch matrix
    nodes = basis.nodes
    M_scratch = cell.M_scratch
    M_scratch .*= 0.0

    # Loop over all pairs of basis function nodes
    for (k,ij) in enumerate(product(nodes,nodes))
        (i,j) = ij
        # Compute the integrand for the mass matrix and store it in the scratch matrix
        M_scratch[k] = LL_ij(basis,(j,i),x) * FR_weight
    end

end

"""
    Fisher-Rao DG mass matrix for Transport equation
"""
function FRG_compute_M!(    cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis,
                            quadrature::FRGSemidiscretizationMultiDQuadrature   )

    # Extract the mass matrix and scratch matrix from the cell
    M = cell.M
    M_scratch = cell.M_scratch

    # Reset the mass matrix to zero
    M .*= 0.0

    # Extract quadrature weights, points, and node indices
    w = quadrature.w
    x_q = quadrature.x_q
    nodes_q = quadrature.nodes_q

    # Loop over all quadrature nodes
    for i in nodes_q
        # Compute the physical coordinates of the quadrature point
        x = [ y[j] for (j,y) in zip(i,x_q) ]

        # Compute the integrand for the mass matrix at the quadrature point
        FRG_compute_M_integrand!(cell,basis,x)

        # Compute the weight for the quadrature point
        weight = prod( ( y[j] for (j,y) in zip(i,w) ) )

        # Accumulate the weighted integrand into the mass matrix
        M .+= weight * M_scratch
    end

end

"""
    Volume integrand for Fisher-Rao DG stiffness vector for Transport equation
"""
function FRG_compute_K_volume_integrand!(   cell::FRGSemidiscretizationMultiDCell,
                                            basis::FRGSemidiscretizationMultiDBasis,
                                            x   )
    # Extract the velocity and basis function nodes
    u = cell.u
    nodes = basis.nodes

    # Compute the Fisher-Rao weight at the given point x
    FR_weight = compute_FR_weight(cell,basis,x)

    # Reset the scratch matrix for the stiffness vector
    K_scratch = cell.K_scratch
    K_scratch .*= 0.0

    # Loop over all pairs of basis function nodes
    for (l,ij) in enumerate(product(nodes,nodes))
        i,j = ij
        for (n,k) in enumerate(nodes)
            # Compute the volume integrand for the stiffness vector
            K_scratch[l] += LLDxLu_ijk(basis,(i,k,j),u[n],x)
            K_scratch[l] += LLDxLu_ijk(basis,(i,j,k),u[n],x)
        end
    end

    # Scale the stiffness vector integrand by the Fisher-Rao weight
    K_scratch .*= FR_weight

end

"""
    Surface integrand for Fisher-Rao DG stiffness vector for Transport equation
"""
function FRG_compute_K_surface_integrand!(  cell::FRGSemidiscretizationMultiDCell,
                                            basis::FRGSemidiscretizationMultiDBasis,
                                            b,
                                            x   )
    # Extract the velocity and basis function nodes
    u = cell.u
    nodes = basis.nodes

    # Compute the Fisher-Rao weight at the given point x
    FR_weight = compute_FR_weight(cell,basis,x)

    # Reset the scratch matrix for the stiffness vector
    K_scratch = cell.K_scratch
    K_scratch .*= 0.0

    # Extract the surface index
    s = basis.s

    # Loop over all pairs of basis function nodes
    for (l,ij) in enumerate(product(nodes,nodes))
        i,j = ij
        for (n,k) in enumerate(nodes)
            # Compute the surface integrand for the stiffness vector
            K_scratch[l] += LLLu_ijk(basis,(j,i,k),u[n],b,x)
        end
    end

    # Scale the stiffness vector integrand by the Fisher-Rao weight
    K_scratch .*= FR_weight

end

"""
    Fisher-Rao DG stiffness vector for Transport equation
"""
function FRG_compute_K!(    cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis, 
                            quadrature::FRGSemidiscretizationMultiDQuadrature   )

    # Extract the stiffness matrix and reset it to zero
    K = cell.K
    K .*= 0.0

    # Extract the scratch matrix, quadrature points, and weights
    K_scratch = cell.K_scratch
    x_k = basis.x_k
    s = basis.s
    nodes_q = quadrature.nodes_q
    nodes_q_surf = quadrature.nodes_q_surf
    w = quadrature.w
    x_q = quadrature.x_q

    # Compute the volume integral
    for i in nodes_q
        # Compute the coordinates of the quadrature point
        x = [ y[j] for (j,y) in zip(i,x_q) ]
        # Compute the volume integrand for the stiffness vector
        FRG_compute_K_volume_integrand!(cell,basis,x) 
        # Compute the weight for the quadrature point
        w_i = prod( ( y[j] for (j,y) in zip(i,w) ) )
        # Accumulate the weighted integrand into the stiffness matrix
        K .+= w_i * K_scratch
    end

    # Compute the surface integral
    for (b,surf) in enumerate(nodes_q_surf)
        for i in surf
            # Compute the physical coordinates of the quadrature point
            x = [ y[j] for (j,y) in zip(i,x_q) ]
            # Adjust the coordinate for the surface
            x[abs(s[b])] = s[b] < 0 ? 0 : x_k[abs(s[b])][end]
            # Compute the surface integrand for the stiffness vector
            FRG_compute_K_surface_integrand!(cell,basis,b,x)
            # Compute the weight for the quadrature point
            w_i = prod( ( y[j] for (l,(j,y)) in enumerate(zip(i,w)) if l != abs(s[b]) ) , init=1 )
            # Subtract the weighted integrand from the stiffness matrix
            K .-= w_i * K_scratch
        end
    end
    
end

"""
    Surface integrand for Fisher-Rao DG numerical flux vector for Transport equation
"""
function FRG_compute_F_surface_integrand!(  cell::FRGSemidiscretizationMultiDCell,
                                            basis::FRGSemidiscretizationMultiDBasis, 
                                            b,
                                            x   )

    # Extract basis function nodes and surface nodes
    nodes = basis.nodes
    nodes_surf = basis.nodes_surf
    s = basis.s

    # Extract velocity, numerical flux, and scratch storage for flux
    u = cell.u
    f_hat = cell.f_hat
    F = cell.F_scratch
    F .*= 0.0

    # Loop over all basis function nodes and surface nodes
    for (l,j) in enumerate(nodes)
        for (n,k) in enumerate(nodes_surf[b])   
            # Compute the surface integrand for the numerical flux vector
            F[l] += LLf_ij(basis,(j,k),f_hat[b][n],b,x) 
        end
    end

    # Compute the Fisher-Rao weight at the given point x
    FR_weight = compute_FR_weight(cell,basis,x)

    # Scale the flux integrand by the Fisher-Rao weight
    F .*= FR_weight

end

"""
    Fisher-Rao DG numerical flux vector for Transport equation
"""
function FRG_compute_F!(    cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis, 
                            quadrature::FRGSemidiscretizationMultiDQuadrature   )

    # Extract surface indices, quadrature points, and weights
    s = basis.s
    x_k = basis.x_k
    x_q = quadrature.x_q
    w = quadrature.w
    nodes_q_surf = quadrature.nodes_q_surf
    F_scratch = cell.F_scratch
    F = cell.f
    F .*= 0.0

    # Loop over all surface quadrature nodes
    for (b,surf) in enumerate(nodes_q_surf)
        for i in surf
            # Compute the coordinates of the quadrature point
            x = [ y[j] for (j,y) in zip(i,x_q) ]
            # Adjust the coordinate for the surface
            x[abs(s[b])] = s[b] < 0 ? 0 : x_k[abs(s[b])][end]
            # Compute the surface integrand for the numerical flux vector
            FRG_compute_F_surface_integrand!(cell,basis,b,x)
            # Compute the weight for the quadrature point
            w_i = prod( ( y[j] for (l,(j,y)) in enumerate(zip(i,w)) if l != abs(s[b]) ) , init=1.0 )
            # Accumulate the weighted integrand into the flux vector
            F .+= w_i * F_scratch
        end
    end

end

"""
    Construct struct with DG semidiscretization for Transport equations
"""
function FRGSemidiscretizationDGMultiD( p,
                                        d,
                                        Δx,
                                        Δt,
                                        m,
                                        n_q;
                                        f_type="upwind",
                                        FR_weight=FR_weight,
                                        quad_type="gausslegendre") 

    # Generate basis function nodes and surface nodes
    nodes = product([ 1:c+1 for c in p ]...)
    nodes_surf = surface_nodes(d,p)
    nodes_q = product([ 1:c for c in n_q ]...)
    nodes_q_surf = surface_nodes(d,n_q.-1)
    cell_idx = product([ 1:c for c in m ]...)
    num_cell = prod(m)
    num_surf = 2*d

    # Initialize flux storage
    f = [ [ zeros(Float64,length(nodes_surf[i])) for i in 1:2*d  ] for k in cell_idx ]
    f = reshape(f,(:))
    w = Vector{Float64}[]
    n_k = p .+ 1
    x_k = Vector{Float64}[]
    x_q = Vector{Float64}[]
    s = Int64[]
    num_nodes = length(nodes)

    # Compute quadrature points and weights for each dimension
    for i in 1:d
        xi_q,wi = quadrature_points(Δx[i],n_q[i],type=quad_type)
        xi_k = collect(0 : p[i]) * ( Δx[i] / p[i] )
        push!(w,wi)
        push!(x_q,xi_q)
        push!(x_k,xi_k)
        push!(s,-i) ; push!(s,i)
    end

    # Determine the type of dimension for the semidiscretization
    if d == 1
        type_dim = Tuple{UnitRange{Int64}}
    elseif d == 2
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}}
    elseif d == 3
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}
    end
    
    # Return the semidiscretization struct
    return FRGSemidiscretizationDGMultiD{type_dim}(     p,
                                                        d,
                                                        Δx,
                                                        Δt,
                                                        m,
                                                        f,
                                                        f_type,
                                                        FR_weight,
                                                        n_k,
                                                        x_k,
                                                        n_q,
                                                        x_q,
                                                        w,
                                                        cell_idx,
                                                        num_nodes,
                                                        nodes,
                                                        nodes_surf,
                                                        nodes_q,
                                                        nodes_q_surf,
                                                        s               ) 
end

"""
    Construct struct with basis function information
"""
function FRGSemidiscretizationMultiDBasis(sd::FRGSemidiscretizationDGMultiD)

    # Extract parameters from the semidiscretization object
    p = sd.p
    d = sd.d
    Δx = sd.Δx
    n_k = sd.n_k
    x_k = sd.x_k
    num_nodes = sd.num_nodes
    nodes = sd.nodes
    nodes_surf  = sd.nodes_surf
    s = sd.s

    # Determine the type of dimension for the basis
    if d == 1
        type_dim = Tuple{UnitRange{Int64}} 
    elseif d == 2
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}} 
    elseif d == 3
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}} 
    end

    # Return the basis struct
    return FRGSemidiscretizationMultiDBasis{type_dim}(  p,
                                                        d,
                                                        Δx,
                                                        n_k,
                                                        x_k,
                                                        num_nodes,
                                                        nodes,
                                                        nodes_surf,
                                                        s               ) 
end

"""
    Construct struct with quadrature information based on semidiscretization
"""
function FRGSemidiscretizationMultiDQuadrature(sd::FRGSemidiscretizationDGMultiD)

    # Extract parameters from the semidiscretization object
    d = sd.d
    nodes_q = sd.nodes_q
    nodes_q_surf = sd.nodes_q_surf
    n_q  = sd.n_q
    x_q = sd.x_q
    w = sd.w
    s = sd.s

    # Determine the type of dimension for the quadrature
    if d == 1
        type_dim = Tuple{UnitRange{Int64}} 
    elseif d == 2
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}} 
    elseif d == 3
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}} 
    end

    # Return the quadrature struct
    return FRGSemidiscretizationMultiDQuadrature{type_dim}(     d,
                                                                nodes_q,
                                                                nodes_q_surf,
                                                                n_q,
                                                                x_q,
                                                                w,
                                                                s               ) 
end

"""
    Construct struct with quadrature information
"""
function FRGSemidiscretizationMultiDQuadrature( d,
                                                L,
                                                n_q;
                                                quad_type="gausslegendre"   )

    # Generate quadrature nodes and surface nodes
    nodes_q = product([ 1:c for c in n_q ]...)
    nodes_q_surf = surface_nodes(d,n_q.-1)
    w = Vector{Float64}[]
    x_q = Vector{Float64}[]
    s = Int64[]
    # Compute quadrature points and weights for each dimension
    for i in 1:d
        xi_q,wi = quadrature_points(L[i],n_q[i],type=quad_type)
        push!(w,wi)
        push!(x_q,xi_q)
        push!(s,-i) ; push!(s,i)
    end

    # Determine the type of dimension for the quadrature
    if d == 1
        type_dim = Tuple{UnitRange{Int64}} 
    elseif d == 2
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}} 
    elseif d == 3
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}} 
    end

    # Return the quadrature struct
    return FRGSemidiscretizationMultiDQuadrature{type_dim}(     d,
                                                                nodes_q,
                                                                nodes_q_surf,
                                                                n_q,
                                                                x_q,
                                                                w,
                                                                s               ) 
end

"""
    Function that generates a function to construct 
    DG cell for Transport equations
"""
function cell_generator_generator(sd::FRGSemidiscretizationDGMultiD)
    
    # Extract the dimension, Fisher-Rao weighting flag, and number of nodes
    d = sd.d
    FR_weight = sd.FR_weight
    num_nodes = sd.num_nodes

    # Determine the type of dimension for the cell
    if d == 1
        type_dim = Tuple{UnitRange{Int64}} 
    elseif d == 2
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}} 
    elseif d == 3
        type_dim = Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}} 
    end

    # Define the cell generator function
    function cell_generator(r_vec, u_vec, f_hat_vec)
        # Initialize density, velocity, and numerical flux vectors
        r = r_vec
        u = u_vec
        f_hat = f_hat_vec        

        # Initialize matrices and vectors for mass, stiffness, and flux
        M = zeros(num_nodes, num_nodes)
        K = zeros(num_nodes, num_nodes)
        f = zeros(num_nodes)
        
        # Initialize scratch storage for intermediate computations
        M_scratch = zeros(num_nodes, num_nodes)
        K_scratch = zeros(num_nodes, num_nodes)
        F_scratch = zeros(num_nodes)

        # Return the constructed cell object
        return FRGSemidiscretizationMultiDCell{type_dim}(   M,
                                                            K,
                                                            f,
                                                            M_scratch,
                                                            K_scratch,
                                                            F_scratch,
                                                            r,
                                                            u,
                                                            f_hat,
                                                            FR_weight         )
    end

    return cell_generator

end

"""
    Generate an empty cell for Transport equations
"""
function empty_cell(sd::FRGSemidiscretizationDGMultiD)

    # Generate a cell generator function
    cell_generator = cell_generator_generator(sd)

    # Initialize density, velocity, and numerical flux vectors
    r = zeros(sd.num_nodes)
    u = [ zeros(sd.d) for i in 1:sd.num_nodes ]
    f_hat = sd.f[1] .* 0

    return cell_generator(r, u, f_hat)

end

"""
    Initialize spacial array from semidiscretization
"""
function initialize_spacial_array(sd::AbstractTransportSemidiscretization)

    # Extract parameters from the semidiscretization object
    p = sd.p
    d = sd.d
    m = sd.m
    Δx = sd.Δx

    # Initialize the spatial array
    xs = initialize_spacial_array(d, m, p, Δx)  

    return xs
    
end

"""
    Compute density flux for Transport Equation
"""
function compute_density_flux(sd::AbstractTransportSemidiscretization,
                              ρ, 
                              u)
    # Compute the density flux as the product of density and velocity
    return ρ .* u
end

"""
    Compute upwind flux
"""
function upwind_flux!(sd::AbstractTransportSemidiscretization,
                      ρ_flux,
                      u)

    # Extract parameters from the semidiscretization object
    f = sd.f 
    m = sd.m
    p = sd.p
    s = sd.s
    d = sd.d
    n_k = sd.n_k
    cell_idx = sd.cell_idx
    nodes_surf = sd.nodes_surf
    dim = n_k .* m

    # Loop over all cells in the domain
    for (k, c) in enumerate(cell_idx)
        # Adjust the indices for the current cell
        idx_adj = (c .- 1) .* n_k
        for (i, b) in enumerate(s)
            # Loop over all surface nodes for the current dimension
            for (j, j_idx) in enumerate(nodes_surf[i])
                # Adjust the node indices for the current cell
                j_idx = j_idx .+ idx_adj
                # Convert the tensor index to a linear index
                j_adj = tensor_index_to_linear_index(j_idx, dim, d)
                if sign(b) == 1
                    # Handle the case for positive surface index
                    if sign(u[j_adj][abs(b)]) > 0
                        # Upwind flux: use the current cell's flux
                        f[k][i][j] = ρ_flux[j_adj][abs(b)]  
                    else  
                        # Downwind flux: adjust the index to the neighboring cell
                        j_idx[abs(b)] = mod(j_idx[abs(b)] + 1, m[abs(b)] * n_k[abs(b)]) 
                        j_adj = tensor_index_to_linear_index(j_idx, dim, d)
                        f[k][i][j] = ρ_flux[j_adj][abs(b)]
                    end
                else
                    # Handle the case for negative surface index
                    if sign(u[j_adj][abs(b)]) > 0
                        # Upwind flux: adjust the index to the neighboring cell
                        j_idx[abs(b)] = mod(j_idx[abs(b)] - 2, m[abs(b)] * n_k[abs(b)]) + 1
                        j_adj = tensor_index_to_linear_index(j_idx, dim, d)
                        f[k][i][j] = ρ_flux[j_adj][abs(b)]  
                    else  
                        # Downwind flux: use the current cell's flux
                        f[k][i][j] = ρ_flux[j_adj][abs(b)]
                    end
                end
            end
        end
    end

end


"""
    Compute numerical flux for Transport equation
"""
function compute_numerical_fluxes!(sd::FRGSemidiscretizationDGMultiD,
                                   ρ,
                                   u)

    # Extract the flux type
    f_type = sd.f_type

    # Compute the density flux
    ρ_flux = compute_density_flux(sd, ρ, u)

    # Compute the numerical flux based on the specified flux type
    if f_type == "upwind"
        upwind_flux!(sd, ρ_flux, u)
    else
        msg = "invalid flux type specified"
        error(msg...)
    end

end

"""
    Compute cell update 
"""
function FRG_cell_update!(  cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis, 
                            quadrature::FRGSemidiscretizationMultiDQuadrature   )
    # Compute the mass matrix for the cell
    FRG_compute_M!(cell, basis, quadrature)
    # Compute the stiffness matrix for the cell
    FRG_compute_K!(cell, basis, quadrature)
    # Compute the numerical flux vector for the cell
    FRG_compute_F!(cell, basis, quadrature)

    # Return the cell update by solving the linear system
    return - cell.M \ (cell.f + cell.K * cell.r)  
end

"""
    Return cell generator parameters
"""
function cell_parameters(   sd::FRGSemidiscretizationDGMultiD,
                            c_idx,
                            k,
                            ρ,
                            u;
                            dim = sd.n_k .* sd.m  )
    # Extract the dimension, number of nodes, and basis function nodes
    d = sd.d
    n_k = sd.n_k
    nodes = sd.nodes
    num_nodes = sd.num_nodes

    # Initialize density and velocity vectors for the cell
    r_k = zeros(num_nodes)
    u_k = [zeros(d) for i in 1:num_nodes]

    # Compute the index adjustment for the current cell
    idx_adj = (c_idx .- 1) .* n_k

    # Map the global density and velocity to the local cell
    for (i, i_idx) in enumerate(nodes) 
        idx_domain = i_idx .+ idx_adj
        i_adj = tensor_index_to_linear_index(idx_domain, dim, d)
        r_k[i] = ρ[i_adj]
        u_k[i] = u[i_adj]
    end

    # Extract the numerical flux for the cell
    f_k = sd.f[k] 

    return r_k, u_k, f_k
end

"""
    Update cell density with values from the global domain
"""
function update_cell_density!(  cell::FRGSemidiscretizationMultiDCell,
                                sd::FRGSemidiscretizationDGMultiD,
                                c_idx,
                                ρ;
                                dim = sd.n_k .* sd.m   )
    # Extract the dimension, number of nodes, and basis function nodes
    d = sd.d
    n_k = sd.n_k
    nodes = sd.nodes
    num_nodes = sd.num_nodes

    # Extract the density vector for the cell
    r_k = cell.r

    # Compute the index adjustment for the current cell
    idx_adj = (c_idx .- 1) .* n_k

    # Map the global density to the local cell
    for (i, i_idx) in enumerate(nodes) 
        idx_domain = i_idx .+ idx_adj
        i_adj = tensor_index_to_linear_index(idx_domain, dim, d)
        r_k[i] = ρ[i_adj]
    end
end


"""
    Compute time update for density 
"""
function FRG_update!(   sd::FRGSemidiscretizationDGMultiD,
                        ∂tρ,
                        ρ,
                        u   )

    # Extract parameters from the semidiscretization object
    p = sd.p
    d = sd.d
    m = sd.m
    nodes = sd.nodes
    nodes_surf = sd.nodes_surf
    cell_idx = sd.cell_idx
    num_nodes = sd.num_nodes
    n_k = sd.n_k
    x_k = sd.x_k
    s = sd.s
    dim = n_k .* m

    # Initialize basis and quadrature objects
    basis = FRGSemidiscretizationMultiDBasis(sd)
    quadrature = FRGSemidiscretizationMultiDQuadrature(sd)
    
    # Compute numerical fluxes for the current density and velocity
    compute_numerical_fluxes!(sd,ρ,u)

    # Access the flux storage
    f = sd.f 

    # Generate a cell generator function
    cell_generator = cell_generator_generator(sd)

    # Loop over all cells in the domain
    for (k,c) in enumerate(cell_idx) 

        # Adjust the indices for the current cell
        idx_adj = ( c .- 1 ) .* n_k

        # Extract cell-specific parameters
        r_k, u_k, f_k = cell_parameters(sd, c, k, ρ, u)

        # Generate a cell object for the current cell
        cell = cell_generator(r_k, u_k, f_k)

        # Compute the time update for the current cell
        update = FRG_cell_update!(cell, basis, quadrature)

        # Map the cell-local updates to the global domain
        for (i, i_idx) in enumerate(nodes) 
            idx = i_idx .+ idx_adj
            i_adj = tensor_index_to_linear_index(idx, dim, d)
            ∂tρ[i_adj] = update[i]
        end

    end  
end

"""
    Runge–Kutta 4th order time step for transport equation
"""
function rk4_step!( sd::FRGSemidiscretizationDGMultiD,
                    update_ρ,
                    k1_ρ,
                    k2_ρ,
                    k3_ρ,
                    k4_ρ,
                    Δt,
                    ρ,
                    u   )

    # compute k1
    FRG_update!( sd, k1_ρ, ρ, u )

    # compute k2
    ρ2 = ρ .+ Δt * k1_ρ / 2
    FRG_update!( sd, k2_ρ, ρ2, u )

    # compute k3
    ρ3 = ρ .+ Δt * k2_ρ / 2
    FRG_update!( sd, k3_ρ, ρ3, u )

    # compute k4
    ρ4 = ρ .+ Δt * k3_ρ 

    # compute update term
    FRG_update!( sd, k4_ρ, ρ4, u )

    update_ρ .= Δt * ( k1_ρ .+ 2 * k2_ρ .+ 2 * k3_ρ .+ k4_ρ ) / 6

end

"""
    Runge–Kutta 4th order solver for transport equation
"""
function rk4(   sd::FRGSemidiscretizationDGMultiD, 
                T, 
                ρ0, 
                u0, 
                Δt_record; 
                verbose=false, 
                stype=Float64, 
                catch_error=true, 
                adaptive_time_step=false,
                DG_limiter=false )
    
    # number of nodes
    m = sd.m 
    # basis function polynomial order
    p = sd.p
    #dimension
    d = sd.d

    # construct matrix with left and right value at each node
    ρ = ρ0
    u = u0

    # initialize the time
    Δt = get_Δs(sd)[2]
    t = Δt
    reduced_time_step = false

    # initialize update vectors 
    update_ρ = copy(ρ)
    diff = copy(ρ)
    k1_ρ = zeros( size(update_ρ) )
    k2_ρ = zeros( size(update_ρ) )
    k3_ρ = zeros( size(update_ρ) )
    k4_ρ = zeros( size(update_ρ) )

    # setting up arrays for the output
    ρ_out = Array{Float64, d}[]
    push!(ρ_out, ρ0)
    t_out = zeros(stype, 1) 

    while t < T 
        if catch_error
            try       
                rk4_step!(  sd,
                            update_ρ,
                            k1_ρ,
                            k2_ρ,
                            k3_ρ,
                            k4_ρ,
                            Δt,
                            ρ,
                            u   )

            catch
                break
            end
        else

            rk4_step!(  sd,
                        update_ρ,
                        k1_ρ,
                        k2_ρ,
                        k3_ρ,
                        k4_ρ,
                        Δt,
                        ρ,
                        u   )

        end

        # check if updated value is negative
        diff = ρ - update_ρ

        # time stepper 
        if adaptive_time_step
            if minimum(diff) > 0.0 && Δt < sd.Δt && !reduced_time_step
                reduced_time_step = false
                Δt = 2 * Δt
                continue
            elseif minimum(diff) <= 0.0 && Δt > sd.Δt / 1024
                reduced_time_step = true
                Δt = Δt / 2
                continue
            else
                reduced_time_step = false
                ρ += update_ρ
                t += Δt
            end
        else
            ρ += update_ρ
            t += Δt
        end
        
        # modify negative cells to ensure they are positive
        # while ensuring mass conservation is maintained
        if DG_limiter == true
            alternative_positivity_preservation!(sd,ρ)
        end

        # Write to output only if we have reached a new multiple of Δt_record
        if div( t, Δt_record ) > div( t_out[end], Δt_record )   
            push!(t_out, t)
            push!(ρ_out, ρ )

            if verbose
                println("Current time snapshot is $(t)")
            end
        end

    end

    return ρ_out, t_out

end

"""
    Runge–Kutta 3rd order time step for transport equation
"""
function rk3_step!( sd::FRGSemidiscretizationDGMultiD,
                    update_ρ,
                    k1_ρ,
                    k2_ρ,
                    k3_ρ,
                    Δt,
                    ρ,
                    u   )

    # compute k1
    FRG_update!( sd, k1_ρ, ρ, u )

    # compute k2
    ρ2 = ρ  .+  Δt * k1_ρ
    FRG_update!( sd, k2_ρ, ρ2, u )

    # compute k3
    ρ3 =  3/4 * ρ  .+ 1/4 * ( ρ2 .+ Δt * k2_ρ )
    FRG_update!( sd, k3_ρ, ρ3, u )

    # compute time step
    update_ρ .= Δt * ( k1_ρ .+ k2_ρ .+ 4 * k3_ρ  ) / 6

end

"""
    Runge–Kutta 3rd order solver for transport equation
"""
function rk3(   sd::FRGSemidiscretizationDGMultiD, 
                T, 
                ρ0, 
                u0, 
                Δt_record; 
                verbose=false, 
                stype=Float64,
                DG_limiter=false,  
                catch_error=true    )
    
    # number of nodes
    m = sd.m 
    # basis function polynomial order
    p = sd.p
    #dimension 
    d = sd.d

    # construct matrix with left and right value at each node
    ρ = ρ0
    u = u0

    # initialize the time
    Δt = get_Δs(sd)[2]
    t = Δt

    # initialize update vectors 
    update_ρ = copy(ρ)
    k1_ρ = zeros( size(update_ρ) )
    k2_ρ = zeros( size(update_ρ) )
    k3_ρ = zeros( size(update_ρ) )

    # setting up arrays for the output
    ρ_out = Array{Float64, d}[]
    push!(ρ_out, ρ0)
    t_out = zeros(stype, 1) 

    while t < T 
        if catch_error
            try       
                rk3_step!(  sd,
                            update_ρ,
                            k1_ρ,
                            k2_ρ,
                            k3_ρ,
                            Δt,
                            ρ,
                            u   )

            catch
                break
            end
        else

            rk3_step!(  sd,
                        update_ρ,
                        k1_ρ,
                        k2_ρ,
                        k3_ρ,
                        Δt,
                        ρ,
                        u   )
        end


        # time stepper 
        
        ρ += update_ρ
        t += Δt

        # modify negative cells to ensure they are positive
        if DG_limiter
            alternative_positivity_preservation!(sd,ρ)
        end
        
        # Write to output only if we have reached a new multiple of Δt_record
        if div( t, Δt_record ) > div( t_out[end], Δt_record )   
            push!(t_out, t)
            push!(ρ_out, ρ )

            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end

    if T >  t_out[end] 
        push!(t_out, t)
        push!(ρ_out, ρ )

        if verbose
            println("Current time snapshot is $(t)")
        end
    end

    return ρ_out, t_out

end
"""
    compute density at point x
"""
function density_integrand!(    cell::FRGSemidiscretizationMultiDCell,
                                basis::FRGSemidiscretizationMultiDBasis,
                                x   )
    # Compute the density integrand at a given point x
    nodes = basis.nodes
    F_scratch = cell.F_scratch

    for (k,j) in enumerate(nodes)
        F_scratch[k] = L_j(basis,j,x)
    end
end

"""
    compute mass in cell
"""
function compute_cell_mass( cell::FRGSemidiscretizationMultiDCell,
                            basis::FRGSemidiscretizationMultiDBasis,
                            quadrature::FRGSemidiscretizationMultiDQuadrature   )
    # Compute the total mass in a single cell using quadrature integration

    # Extract the cell density and flux storage
    r = cell.r
    F = cell.f
    F_scratch = cell.F_scratch

    # Reset the flux storage to zero
    F .*= 0.0

    # Extract quadrature weights, points, and node indices
    w = quadrature.w
    x_q = quadrature.x_q
    nodes_q = quadrature.nodes_q

    # Loop over all quadrature nodes
    for i in nodes_q
        # Compute the physical coordinates of the quadrature point
        x = [ y[j] for (j,y) in zip(i,x_q) ]

        # Compute the density integrand at the quadrature point
        density_integrand!(cell, basis, x)

        # Compute the weight for the quadrature point
        weight = prod( ( y[j] for (j,y) in zip(i,w) ) )

        # Accumulate the weighted integrand into the flux storage
        F .+= weight * F_scratch
    end

    # Compute the total mass by summing the product of density and flux
    return sum( F .* r )
end

"""
    compute mass in whole domain
"""
function compute_domain_mass(   sd::FRGSemidiscretizationDGMultiD,
                                ρ   )  
    # Compute the total mass in the entire domain by summing over all cells

    # Initialize the basis and quadrature objects
    basis = FRGSemidiscretizationMultiDBasis(sd)
    quadrature = FRGSemidiscretizationMultiDQuadrature(sd)

    # Extract the cell indices and initialize an empty cell object
    cell_idx = sd.cell_idx
    cell = empty_cell(sd)

    # Initialize the total mass to zero
    mass = 0.0
    
    # Loop over all cells in the domain
    for c in cell_idx 
        # Update the cell density for the current cell
        update_cell_density!(cell,sd,c,ρ)

        # Compute the mass of the current cell and add it to the total mass
        mass += compute_cell_mass(cell,basis,quadrature)
    end

    return mass
end

"""
    compute mass at each time point
"""
function compute_mass(sd::FRGSemidiscretizationDGMultiD,ρ_out)

    # Initialize an empty vector to store the mass at each time point
    mass_vec = Float64[]  

    for ρ in ρ_out

        # Compute the total mass in the domain for the current density ρ
        mass = compute_domain_mass(sd,ρ)

        # Append the computed mass to the mass vector
        push!( mass_vec , mass )

    end
    return mass_vec  
end

"""
    compute basis function value at point x
"""
function basis_function_value!( basis::FRGSemidiscretizationMultiDBasis,
                                L_scratch,
                                x   )
    nodes = basis.nodes
    for (k,j) in enumerate(nodes)
        L_scratch[k] = L_j(basis,j,x)
    end
end

"""
    Evaluate density at x given solution 
""" 
function evaluate_density(  basis::FRGSemidiscretizationMultiDBasis, 
                            ρ,
                            x   )
    # Evaluate the density at a given point x using the solution ρ
    Δx = basis.Δx
    p = basis.p
    nodes = basis.nodes
    n = size(ρ)

    # Identify cell for x
    macro_idx = Int.( div.( x , Δx ) )
    cell_idx_adj = macro_idx .* ( p .+ 1 )

    # Normalize x to cell
    x_cell = x .- macro_idx .* Δx

    # Initialize the density value at the given point x
    ρ_x = 0.0

    # Loop over all basis function nodes
    for j_idx in nodes
        # Adjust the node indices to account for the cell offset
        j_idx_mod = j_idx .+ cell_idx_adj
        # Wrap the indices around the domain using modular arithmetic
        j_idx_mod = mod1.(j_idx_mod, n)
        # Accumulate the contribution of the current basis function to the density
        ρ_x += ρ[j_idx_mod...] * L_j(basis, j_idx, x_cell)
    end

    return ρ_x
end

"""
    Evaluate density at point in given cell 
""" 
function evaluate_density_in_cell(  basis::FRGSemidiscretizationMultiDBasis,
                                    cell::FRGSemidiscretizationMultiDCell, 
                                    x   )
    # Evaluate the density at a given point x within a specific cell
    nodes = basis.nodes
    r = cell.r

    # Evaluate basis function at normalized x  
    ρ_x = 0.0

    # Loop through each basis function node and accumulate the contribution
    for (j,j_idx) in enumerate(nodes)
        ρ_x += r[j] * L_j(basis,j_idx,x)
    end

    return ρ_x
end

"""
    Evaluate velocity at x given solution 
""" 
function evaluate_velocity( basis::FRGSemidiscretizationMultiDBasis, 
                            u,
                            x   )

    Δx = basis.Δx
    p = basis.p
    n = size(u)
    d = basis.d
    nodes = basis.nodes

    # Identify cell for x
    macro_idx = Int.( div.( x , Δx ) )
    cell_idx_adj = macro_idx .* ( p .+ 1 )

    # Normalize x to cell
    x_cell = x .-  macro_idx .* Δx

    # Evaluate basis function at normalized x  
    u_x = zeros(d)

    for j_idx in nodes
        j_idx_mod = j_idx .+ cell_idx_adj
        j_idx_mod = mod1.( j_idx_mod , n )
        u_x .+= u[j_idx_mod...] * L_j(basis,j_idx,x_cell)
    end

    return u_x
end
"""
    evaluate density at x points given solution
"""
function evaluate_density_points(   sd::FRGSemidiscretizationDGMultiD,
                                    x_points,
                                    ρ  )
    # Create a basis object from the semidiscretization
    basis = FRGSemidiscretizationMultiDBasis(sd)

    # Initialize the output array for density values at given points
    ρ_points = zeros(size(x_points))

    # Loop through each point and evaluate the density
    for (i,x) in enumerate(x_points)
        ρ_points[i] = evaluate_density(basis,ρ,x)
    end

    return ρ_points
end

"""
    update cell density with positive values
"""
function positive_density_cell_update!( cell::FRGSemidiscretizationMultiDCell,
                                        basis::FRGSemidiscretizationMultiDBasis,
                                        quadrature::FRGSemidiscretizationMultiDQuadrature,   
                                        evaluated_cell_density_points                       )
    # Extract cell volume
    Δx = basis.Δx
    nodes = basis.nodes
    vol = prod(Δx)
    
    # Compute minimum density, total mass, and mean density
    min_density = minimum(evaluated_cell_density_points)
    mass = compute_cell_mass(cell, basis, quadrature)
    mean_density = mass / vol
    weight = mean_density / (mean_density - min_density)

    # Update cell density to ensure positivity
    ρ = cell.r

    if mean_density > 0.0
        # Adjust density values to ensure positivity while preserving mass
        for (i, _) in enumerate(nodes) 
            ρ[i] = weight * ρ[i] + (1 - weight) * mean_density + 1e-16
        end
    else
        # If mean density is non-positive, set a small positive value
        for (i, _) in enumerate(nodes) 
            ρ[i] = 1e-15
        end 
    end
end

"""
    alternative positivity preservation
"""
function alternative_positivity_preservation!(  sd::FRGSemidiscretizationDGMultiD,
                                                ρ   )
    # Extract parameters from the semidiscretization
    n_k = sd.n_k
    x_q = sd.x_q
    d = sd.d
    m = sd.m
    Δx = sd.Δx
    nodes = sd.nodes
    dim = n_k .* m

    # Create basis, quadrature, and empty cell objects
    basis = FRGSemidiscretizationMultiDBasis(sd)
    quadrature = FRGSemidiscretizationMultiDQuadrature(sd)
    cell = empty_cell(sd)

    # Generate quadrature points within a cell
    x_q_cell = product( [ x_q[i] for i in 1:d ]... )
    density_quadrature_points = zeros(size(x_q_cell))

    # Identify cells with negative quadrature points
    negative_cells_idx = cells_with_negative_quadrature_points(sd,ρ)

    # Loop through each negative cell and update its density
    for c_idx in negative_cells_idx
        update_cell_density!(cell,sd,c_idx,ρ)

        # Evaluate density at quadrature points within the cell
        for (i,quadrature_node) in enumerate(x_q_cell)
            density_quadrature_points[i] = evaluate_density_in_cell(    basis,
                                                                        cell,
                                                                        quadrature_node       )
        end 

        # Update cell density to ensure positivity
        positive_density_cell_update!(  cell,
                                        basis,
                                        quadrature,   
                                        density_quadrature_points )

        # Map the updated cell density back to the global domain
        idx_adj = ( c_idx .- 1 ) .* n_k

        for (i,i_idx) in enumerate(nodes) 
            idx_domain =  i_idx .+ idx_adj
            i_adj = tensor_index_to_linear_index(idx_domain,dim,d)
            ρ[i_adj] = cell.r[i]
        end
    end
end

"""
    find cells with negative nodes
"""
function cells_with_negative_nodes( sd::FRGSemidiscretizationDGMultiD,
                                    ρ   )
    # Extract parameters from the semidiscretization
    n_k = sd.n_k
    d = sd.d
    m = sd.m
    dim = n_k .* m

    # Initialize a vector to store indices of cells with negative nodes
    negative_cell = Vector{Int64}[]

    # Identify negative nodes in the global density array
    iter = filter( x -> x[2] < 0 , enumerate(ρ) )

    # Map the negative nodes to their respective cells
    for (i, _) in iter 
        cell_idx = ( linear_index_to_tensor_index(i,dim,d) .- 1 ) .÷ n_k .+ 1
        push!(negative_cell,cell_idx)
    end

    return negative_cell
end

"""
    find cells with negative quadrature points
"""
function cells_with_negative_quadrature_points(    sd::FRGSemidiscretizationDGMultiD,
                                                    ρ   )
    # Extract parameters from the semidiscretization
    d = sd.d
    x_q = sd.x_q
    m = sd.m
    n_k = sd.n_k
    cell_idx = sd.cell_idx
    num_nodes = sd.num_nodes

    # Create basis and empty cell objects
    basis = FRGSemidiscretizationMultiDBasis(sd)
    cell = empty_cell(sd)

    # Generate quadrature points within a cell
    x_q_cell = product( [ x_q[i] for i in 1:d ]... )
    negative_cell = Vector{Int64}[]

    # Loop through each cell and check for negative quadrature points
    for c_idx in cell_idx 
        # Update the cell density for the current cell
        update_cell_density!( cell, sd, c_idx, ρ )

        # Loop through each quadrature point in the cell
        for quadrature_node in x_q_cell
            # Evaluate the density at the current quadrature point
            ρ_x =  evaluate_density_in_cell( basis, cell, quadrature_node ) 

            # If the density is negative, add the cell index to the list and break
            if ρ_x < 0.0
                push!(negative_cell, collect(c_idx))
                break
            end
        end 
    end

    return negative_cell
end

"""
    Initialize quadrature points for the entire domain
"""
function initialize_quadrature_points(sd::FRGSemidiscretizationDGMultiD)
    # Extract parameters from the semidiscretization
    d = sd.d
    x_q = sd.x_q
    m = sd.m
    Δx = sd.Δx

    # Generate quadrature points within a cell
    x_q_cell = product( [ x_q[i] for i in 1:d ]... )

    # Generate macro-level cell coordinates
    x_macro = product([ ( 0.0 : m[i]-1 ) * Δx[i] for i in 1:d ]...)

    # Combine macro-level and cell-level coordinates
    x_combo = map( cell -> map( nodes -> nodes .+ cell , x_q_cell ) , x_macro )

    return x_combo
end


"""
    generate file name for solution output
"""
function generate_saved_example_file_name(  sd::FRGSemidiscretizationDGMultiD,
                                            type,
                                            DG_limiter,
                                            T)

    FR_weight = sd.FR_weight
    m = sd.m
    p = sd.p
    n_q = sd.n_q
    d = sd.d

    if FR_weight
        file_name = "DFRG_"*string(type)*"_"*string(d)*"D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
    elseif DG_limiter
        file_name = "DG+limiter_"*string(type)*"_"*string(d)*"D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
    else
        file_name = "DG_"*string(type)*"_"*string(d)*"D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
    end

    return file_name*".ser"

end