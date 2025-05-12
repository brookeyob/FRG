using Serialization
using CSV
using DataFrames

"""
    compute L1 error in domain
"""
function compute_L1_error(  quadrature::FRGSemidiscretizationMultiDQuadrature,
                            σ,
                            ρ   )

    w = quadrature.w
    nodes_q = quadrature.nodes_q

    # Initialize error solution
    error_soln = 0.0

    # Compute L1 error using quadrature weights and absolute difference
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        error_soln += weight * abs.(ρ[i] - σ[i])
    end

    return error_soln
end

"""
    compute L1 norm in domain
"""
function compute_L1_norm(   quadrature::FRGSemidiscretizationMultiDQuadrature,
                            ρ   )

    w = quadrature.w
    nodes_q = quadrature.nodes_q

    # Initialize norm value
    norm_val = 0.0

    # Compute L1 norm using quadrature weights and absolute values
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        norm_val += weight * abs.(ρ[i])
    end

    return norm_val
end

"""
    compute L2 error in domain
"""
function compute_L2_error(  quadrature::FRGSemidiscretizationMultiDQuadrature,
                            σ,
                            ρ   )

    w = quadrature.w
    nodes_q = quadrature.nodes_q

    # Initialize error solution
    error_soln = 0.0

    # Compute L2 error using quadrature weights and squared differences
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        error_soln += weight * (ρ[i] - σ[i])^2
    end

    return error_soln
end

"""
    compute L2 norm in domain
"""
function compute_L2_norm(  quadrature::FRGSemidiscretizationMultiDQuadrature,
                            ρ   )

    w = quadrature.w
    nodes_q = quadrature.nodes_q

    # Initialize norm value
    norm_val = 0.0

    # Compute L2 norm using quadrature weights and squared values
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        norm_val += weight * (ρ[i])^2
    end

    return norm_val
end

"""
    compute KL error in domain
"""
function compute_KL_error(  quadrature::FRGSemidiscretizationMultiDQuadrature,
                            σ,
                            ρ     )
    
    w = quadrature.w
    nodes_q = quadrature.nodes_q

    # Compute masses for normalization
    mass_ρ = 0.0
    mass_σ = 0.0
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        mass_ρ += weight * ρ[i]
        mass_σ += weight * σ[i]
    end

    # Normalize densities
    ρ = ρ / mass_ρ
    σ = σ / mass_σ

    # Initialize error solution
    error_soln = 0.0

    # Compute KL divergence using quadrature weights and log ratios
    for (i, i_idx) in enumerate(nodes_q)
        weight = prod((y[j] for (j, y) in zip(i_idx, w)))
        ratio = ρ[i] / σ[i]
        if ratio < 0.0
            # Handle small negative values for numerical stability
            if -1e15 < ρ[i] < 0.0
                ρ[i] = 1e-15
                ratio = ρ[i] / σ[i]
            else
                error_soln = Inf
                break
            end
        end
        error_soln += weight * ρ[i] * log(ratio)
    end

    return error_soln
end

"""
    select error type
"""
function select_error_type(error_type)
    # Check the error type and assign the corresponding error computation function
    if error_type == "L1"
        compute_error_type = compute_L1_error
    elseif error_type == "L2"
        compute_error_type = compute_L2_error
    elseif error_type == "KL"
        compute_error_type = compute_KL_error
    else
        # Raise an error if an invalid error type is specified
        msg = "invalid error type specified"
        error(msg...)
    end
    return compute_error_type
end

"""
    Select the appropriate norm computation function based on the norm type.
"""
function select_norm_type(norm_type)
    # Check the norm type and assign the corresponding norm computation function
    if norm_type == "L1"
        compute_norm_type = compute_L1_norm
    elseif norm_type == "L2"
        compute_norm_type = compute_L2_norm
    elseif norm_type == "KL"
        # For KL divergence, use the L1 norm as the norm computation function
        compute_norm_type = compute_L1_norm
    else
        # Raise an error if an invalid norm type is specified
        msg = "invalid error type specified"
        error(msg...)
    end
    return compute_norm_type
end

"""
    Normalize the density by dividing it by the total mass in the domain.
"""
function normalize_density( sd::FRGSemidiscretizationDGMultiD,
                            ρ   )
    # Compute the total mass of the density in the domain
    mass = compute_domain_mass(sd,ρ)
    # Normalize the density
    return ρ / mass
end

"""
    compute error over whole domain
"""
function compute_domain_error(  sd::FRGSemidiscretizationDGMultiD,
                                sd_ref::FRGSemidiscretizationDGMultiD,
                                ρ,
                                ρ_ref;
                                error_type="L2",
                                relative_error=false,
                                quad_type="gausslegendre" )

    # Select the appropriate error and norm computation functions
    compute_error_type = select_error_type(error_type)
    compute_norm_type = select_norm_type(error_type)
    d = sd.d
    n_q = sd.n_q
    soln_error = 0.0
    norm_val = 0.0

    # Special handling for KL divergence
    if error_type == "KL"
        mass = compute_domain_mass(sd,ρ) 
        mass_ref = compute_domain_mass(sd_ref,ρ_ref)
        ρ = normalize_density(sd,ρ)
        ρ_ref = normalize_density(sd_ref,ρ_ref)
        alternative_positivity_preservation!(sd,ρ)
    end
    
    # Generate unique coordinates for error computation
    dim_coordinate = Vector{Float64}[]
    for i in 1:sd.d
        coordinate_sd = range(0.0, sd.m[i] * sd.Δx[i], length = sd.m[i] + 1)
        coordinate_sd_ref = range(0.0, sd_ref.m[i] * sd_ref.Δx[i], length = sd_ref.m[i] + 1)
        coordinate = sort(unique(vcat(coordinate_sd, coordinate_sd_ref)))
        push!( dim_coordinate , coordinate )
    end
    
    # Compute error over each cell in the domain
    error_cell_coordinates = product(dim_coordinate...) |> collect
    dim = size(error_cell_coordinates)

    start_coordinate = error_cell_coordinates[[ 1:c for c in dim.-1 ]...] 
    end_coordinate = error_cell_coordinates[[ 2:c+1 for c in dim.-1 ]...]
    cell_size = [ j.-i for (i,j) in zip(start_coordinate,end_coordinate) ]

    for (x_i,Δx) in zip(start_coordinate,cell_size)

        # Initialize quadrature for the current cell
        quadrature = FRGSemidiscretizationMultiDQuadrature( d,
                                                            Δx,
                                                            n_q,
                                                            quad_type=quad_type  )
    
        x_q = quadrature.x_q
        x_q_points = product(x_q...) |> collect
        x_q_points_vec = [ x_q_point .+ x_i for x_q_point in x_q_points  ]
    
        # Evaluate densities at quadrature points
        ρ_points = evaluate_density_points( sd,
                                            x_q_points_vec,
                                            ρ           )
    
        ρ_points_ref = evaluate_density_points( sd_ref,
                                                x_q_points_vec,
                                                ρ_ref       )
    
        # Add error and norm values
        soln_error += compute_error_type(   quadrature, 
                                            ρ_points,
                                            ρ_points_ref   )
        
        norm_val += compute_norm_type(  quadrature, 
                                        ρ_points_ref )
    end

    # Post-process error and norm values
    if error_type == "L2"
        soln_error = sqrt(soln_error)
        norm_val = sqrt(norm_val)
    elseif error_type == "KL" && !relative_error
        soln_error *= mass_ref
    end

    # Compute relative error if required
    if relative_error && error_type != "KL"
        soln_error /= norm_val
    end
    
    return soln_error

end

"""
    compute entropy over whole domain
"""
function compute_domain_entropy(    sd::FRGSemidiscretizationDGMultiD,
                                    ρ;
                                    error_type="KL",
                                    quad_type="gausslegendre"   )

    # Select the appropriate error computation function based on the error type
    compute_error_type = select_error_type(error_type)
    d = sd.d
    n_q = sd.n_q
    soln_error = 0.0

    # Special handling for KL divergence
    if error_type == "KL"
        mass = compute_domain_mass(sd,ρ) 
        mass_ref = 1.0
        ρ = normalize_density(sd,ρ)
    end
    
    # Generate unique coordinates for entropy computation
    dim_coordinate = Vector{Float64}[]
    for i in 1:sd.d
        coordinate = range(0.0, sd.m[i] * sd.Δx[i], length = sd.m[i] + 1)
        push!(dim_coordinate, coordinate)
    end
    
    # Compute entropy over each cell in the domain
    error_cell_coordinates = product(dim_coordinate...) |> collect
    dim = size(error_cell_coordinates)

    start_coordinate = error_cell_coordinates[[1:c for c in dim .- 1]...] 
    end_coordinate = error_cell_coordinates[[2:c+1 for c in dim .- 1]...]
    cell_size = [j .- i for (i, j) in zip(start_coordinate, end_coordinate)]

    for (x_i, Δx) in zip(start_coordinate, cell_size)

        # Initialize quadrature for the current cell
        quadrature = FRGSemidiscretizationMultiDQuadrature( d,
                                                            Δx,
                                                            n_q,
                                                            quad_type=quad_type  )
    
        x_q = quadrature.x_q
        x_q_points = product(x_q...) |> collect
    
        # Evaluate density at quadrature points
        ρ_points = evaluate_density_points(     sd,
                                                x_q_points,
                                                ρ           )
    
        # Reference density is uniform (all ones)
        ρ_points_ref = ones(size(ρ_points)) 
    
        # Add entropy contribution from the current cell
        soln_error += compute_error_type(   quadrature, 
                                            ρ_points, 
                                            ρ_points_ref   )
    
    end

    # Post-process entropy value
    if error_type == "L2"
        soln_error = sqrt(soln_error)
    elseif error_type == "KL"
        soln_error *= mass
    end

    return soln_error
end

"""
    Compute the error at each time point for a given density and reference density.
"""
function compute_error( sd::FRGSemidiscretizationDGMultiD,
                        sd_ref::FRGSemidiscretizationDGMultiD,
                        ρ_out,
                        ρ_out_ref;
                        error_type="L2",
                        relative_error=false,
                        quad_type="gausslegendre" )

    error_vec = Float64[]

    for (ρ, ρ_ref) in zip(ρ_out, ρ_out_ref)

        # Compute the error for the current time point
        soln_error = compute_domain_error(  sd,
                                            sd_ref,
                                            ρ,
                                            ρ_ref,
                                            error_type=error_type,
                                            relative_error=relative_error,
                                            quad_type=quad_type   )
        push!(error_vec, soln_error)
    end
    return error_vec
end

"""
    compute entropy at each time point
"""
function compute_entropy(   sd::FRGSemidiscretizationDGMultiD,
                            ρ_out;
                            error_type="KL",
                            quad_type="gausslegendre" )

    # Initialize a vector to store entropy values for each time point
    entropy_vec = Float64[]
    
    # Loop through density profile at each time point
    for ρ in ρ_out

        # Compute the entropy for the current time point
        soln_entropy = compute_domain_entropy(  sd,
                                                ρ,
                                                error_type=error_type,
                                                quad_type=quad_type   )

        push!( entropy_vec , soln_entropy )
    end
    
    return entropy_vec
end

"""
    open experiment file
"""
function open_experiment_file(filename)

    sd, ρ_out , t_out = open(filename*".ser", "r") do file
        deserialize(file)
    end

    return sd, ρ_out, t_out
end

function open_experiment_file_type( type;
                                    d=1,
                                    m = [ 16 ],
                                    m_ref = repeat([ 1024 ],d),
                                    p = [ 1 ],
                                    n_q = [ 5 ],
                                    FR_weight=true,
                                    DG_limiter=false,
                                    is_exact=false,
                                    T=1.0   )

    if d == 1
        if is_exact
            file_name = "Exact_"*string(m[1])*"_"*type*"_1D_"*string(m_ref[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        elseif FR_weight
            file_name = "DFRG_"*type*"_1D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        elseif DG_limiter
            file_name = "DG+limiter_"*type*"_1D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        else
            file_name = "DG_"*type*"_1D_"*string(m[1])*"_p_"*string(p[1])*"_q_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        end
    else d == 2
        if is_exact
            file_name = "Exact_"*string(m[1])*"_"*string(m[2])*"_"*type*"_2D_"*string(m_ref[1])*"_"*string(m_ref[2])*"_p_"*string(p[1])*"_"*string(p[2])*"_q_"*string(n_q[1])*"_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        elseif FR_weight
            file_name = "DFRG_"*type*"_2D_"*string(m[1])*"_"*string(m[2])*"_p_"*string(p[1])*"_"*string(p[2])*"_q_"*string(n_q[1])*"_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        elseif DG_limiter
            file_name = "DG+limiter_"*type*"_2D_"*string(m[1])*"_"*string(m[2])*"_p_"*string(p[1])*"_"*string(p[2])*"_q_"*string(n_q[1])*"_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        else
            file_name = "DG_"*type*"_2D_"*string(m[1])*"_"*string(m[2])*"_p_"*string(p[1])*"_"*string(p[2])*"_q_"*string(n_q[1])*"_"*string(n_q[1])*"_T_"*replace(string(T),"." => "-")
        end
    end

    directory_name = dirname(@__DIR__)
    file_path = joinpath(directory_name, "saved examples", string(d)*"D"*"_"*type , file_name*".ser")  


    sd, xs, ρ_out, u0, t_out = open(file_path, "r") do file
        deserialize(file)
    end

    return sd, ρ_out, t_out
end

"""
    compute error at each time point 
"""
function compute_error_over_time(   sd::FRGSemidiscretizationDGMultiD,
                                    sd_ref::FRGSemidiscretizationDGMultiD,
                                    ρ_out,
                                    ρ_out_ref,
                                    t_out,
                                    t_out_ref;
                                    relative_error=false,
                                    quad_type="gausslegendre"  )
                                            
    # Compute L1 error over time
    error_L1 = compute_error(   sd,
                                sd_ref,
                                ρ_out,
                                ρ_out_ref;
                                error_type="L1",
                                relative_error=relative_error,
                                quad_type=quad_type )

    # Compute L2 error over time
    error_L2 = compute_error(   sd,
                                sd_ref,
                                ρ_out,
                                ρ_out_ref;
                                error_type="L2",
                                relative_error=relative_error,
                                quad_type=quad_type )

    # Compute KL divergence (ρ || σ) over time
    error_KL_1 = compute_error( sd,
                                sd_ref,
                                ρ_out,
                                ρ_out_ref;
                                error_type="KL",
                                relative_error=relative_error,
                                quad_type=quad_type )

    # Compute KL divergence (σ || ρ) over time
    error_KL_2 = compute_error( sd_ref,
                                sd,
                                ρ_out_ref,
                                ρ_out;
                                error_type="KL",
                                relative_error=relative_error,
                                quad_type=quad_type )

    # Combine all error metrics into a single array
    error_combo = [ error_L1 error_L2 error_KL_1 error_KL_2  ]
    time_combo = [ t_out, t_out_ref ]

    return error_combo, time_combo

end

"""
    compute entropy at each time point 
"""
function compute_entropy_over_time(     sd::FRGSemidiscretizationDGMultiD,
                                        ρ_out,
                                        t_out;
                                        quad_type="gausslegendre"  )
                                            
    # Compute L1 entropy over time
    error_L1 = compute_entropy(     sd,
                                    ρ_out,
                                    error_type="L1",
                                    quad_type=quad_type )

    # Compute L2 entropy over time
    error_L2 = compute_entropy(     sd,
                                    ρ_out,
                                    error_type="L2",
                                    quad_type=quad_type )

    # Compute KL entropy over time
    error_KL = compute_entropy(     sd,
                                    ρ_out,
                                    error_type="KL",
                                    quad_type=quad_type )

    # Combine all entropy metrics into a single array
    error_combo = [ error_L1 error_L2 error_KL  ]

    return error_combo, t_out

end

"""
    compute error at each time point given experiment type
"""
function compute_error_over_time_type(  type;
                                        d = 1,
                                        m = [10],
                                        p = [1],
                                        n_q = [ 5 ],
                                        FR_weight = true,
                                        DG_limiter = false,
                                        m_ref = [10],
                                        p_ref = [1],
                                        n_q_ref = [ 5 ],
                                        FR_weight_ref = true,
                                        n_q_error = [ 1001 ],
                                        T = 1.0,
                                        n_t = 1,
                                        L = [1.0],
                                        is_exact = false,
                                        relative_error = false,
                                        quad_type = "gausslegendre"                   )

    # Load experiment data for the given type and discretization
    sd, ρ_out , t_out = open_experiment_file_type(type,d=d,m=m,p=p,n_q=n_q,FR_weight=FR_weight,DG_limiter=DG_limiter,T=T)
    sd_ref, ρ_out_ref , t_out_ref = open_experiment_file_type(type,d=d,m=m,m_ref=m_ref,p=p_ref,n_q=n_q_ref,FR_weight=FR_weight_ref,is_exact=is_exact,T=T)


    error_combo, time_combo = compute_error_over_time(  sd,
                                                        sd_ref,
                                                        ρ_out[n_t:n_t:end],
                                                        ρ_out_ref[n_t:n_t:end],
                                                        t_out[n_t:n_t:end],
                                                        t_out_ref[n_t:n_t:end],
                                                        relative_error=relative_error,
                                                        quad_type=quad_type   )


    return error_combo, time_combo

end

"""
    compute entropy at each time point given experiment type
"""
function compute_entropy_over_time_type(    type;
                                            d = 1,
                                            m = [10],
                                            p = [1],
                                            n_q = [ 5 ],
                                            FR_weight = true,
                                            DG_limiter = false,
                                            n_q_error = [ 1001 ],
                                            T = 1.0,
                                            n_t = 1,
                                            L = [1.0],
                                            quad_type="gausslegendre"                   )

    # Load experiment data for the given type and discretization
    sd, ρ_out , t_out = open_experiment_file_type(type,d=d,m=m,p=p,n_q=n_q,FR_weight=FR_weight,DG_limiter=DG_limiter,T=T)

    # Compute entropy over time for the given density
    entropy_combo, t_out = compute_entropy_over_time(   sd,
                                                        ρ_out[n_t:n_t:end],
                                                        t_out[n_t:n_t:end],
                                                        quad_type=quad_type   )

    return entropy_combo, t_out  

end

"""
    Compute error at each time point given experiment type 
    for a set of discretization sizes and methods.
"""
function compute_errors_given_disc_method(  type;
                                            d = 1,
                                            p = [ 1 ],
                                            n_q = [ 5 ],
                                            p_ref = [ 1 ],
                                            m_ref = [ 1024 ],
                                            n_q_ref = [ 5 ],
                                            FR_weight_ref = true, 
                                            n_q_error = [ 5 ], 
                                            T = 1.0, 
                                            n_t = 1, 
                                            m_vec = [8, 16, 32, 64, 128, 256, 512, 1024],
                                            is_exact = false,
                                            relative_error = false,
                                            quad_type = "gausslegendre",
                                            methods = [ "DG", "DG+limiter", "DFRG" ] )

    # Load reference experiment data for the given type and discretization
    sd, _, t_out = open_experiment_file_type(type, d=d, p=p_ref, m=repeat([m_vec[end]], d), m_ref=m_ref, n_q=n_q_ref, is_exact=is_exact, T=T)

    # Initialize array to store errors for each method and discretization size
    error_time_norm_method_disc = Array{Float64}(undef, size(t_out[n_t:n_t:end], 1), 4 , size(methods,1) , size(m_vec, 1))

    # Define settings for each method
    settings = Dict("DG" => [false,false], "DG+limiter" => [false,true], "DFRG" => [true,false])

    # Loop through discretization sizes
    for (m_idx, m) in enumerate(m_vec)

        m = repeat([m], d)

        # Loop through methods
        for (method_idx, method) in enumerate(methods)

            FR_weight, DG_limiter = settings[method]

            # Compute errors over time for the current method and discretization size
            error_combo, _ = compute_error_over_time_type(  type,
                                                            d=d,
                                                            m=m,
                                                            p=p,
                                                            n_q=n_q,
                                                            n_q_ref=n_q_ref,
                                                            FR_weight=FR_weight,
                                                            DG_limiter=DG_limiter,
                                                            m_ref=m_ref,
                                                            p_ref=p_ref,
                                                            FR_weight_ref=FR_weight_ref,
                                                            n_q_error=n_q_error,
                                                            is_exact=is_exact,
                                                            relative_error=relative_error,
                                                            n_t=n_t,
                                                            quad_type = quad_type,
                                                            T=T                         )

            # Store computed errors in the array
            error_time_norm_method_disc[:,:,method_idx,m_idx] = error_combo

        end

    end

    return error_time_norm_method_disc , t_out[n_t:n_t:end]
end

"""
    compute error at each time point given experiment type 
    for a set of discretization sizes and methods
"""
function compute_entropy_given_disc_method( type;
                                            d = 1,
                                            p = [ 1 ],
                                            n_q = [ 5 ],
                                            m_ref = [ 1024 ],
                                            n_q_error = [ 5 ], 
                                            T = 1.0, 
                                            n_t = 1, 
                                            m_vec = [8, 16, 32, 64, 128, 256, 512, 1024], 
                                            methods = [ "DG", "DG+limiter", "DFRG" ] )

    # Load reference experiment data for the given type and discretization
    sd, _, t_out = open_experiment_file_type(type, d=d, m=m_ref, p=p, n_q=n_q, T=T)

    # Initialize array to store entropy values for each method and discretization size
    entropy_time_norm_method_disc = Array{Float64}(undef, size(t_out[n_t:n_t:end], 1), 3 , size(methods,1) , size(m_vec, 1))

    # Define settings for each method
    settings = Dict("DG" => [false,false], "DG+limiter" => [false,true], "DFRG" => [true,false])

    # Loop through discretization sizes
    for (m_idx, m) in enumerate(m_vec)

        # Repeat the discretization size for all dimensions
        m = repeat([m], d)

        # Loop through methods
        for (method_idx, method) in enumerate(methods)

            # Get the settings for the current method
            FR_weight, DG_limiter = settings[method]

            # Compute entropy over time for the current method and discretization size
            entropy_combo, _ = compute_entropy_over_time_type(  type,
                                                                d=d,
                                                                m=m,
                                                                p=p,
                                                                FR_weight=FR_weight,
                                                                DG_limiter=DG_limiter,
                                                                n_q_error=n_q_error,
                                                                n_t=n_t,
                                                                T=T )

            # Store computed entropy values in the array
            entropy_time_norm_method_disc[:,:,method_idx,m_idx] = entropy_combo

        end

    end

    return entropy_time_norm_method_disc , t_out[n_t:n_t:end]

end

"""
    convert errors Array to a DataFrame
"""
function convert_errors_to_dataframe(   error_time_norm_method_disc, 
                                        norm_types, 
                                        methods, 
                                        m_vec   )

    # Convert discretization sizes to strings for use as column identifiers
    discretization_sizes = string.(m_vec)
    
    # Initialize an empty DataFrame to store the error data
    error_time_norm_method_disc_df = DataFrame()

    # Loop through each norm type, method, and discretization size
    for (norm_idx, norm) in enumerate(norm_types)
        for (method_idx, method) in enumerate(methods)
            for (m_idx, m) in enumerate(discretization_sizes)
                
                # Create a column name based on norm type, method, and discretization size
                column_name = Symbol(norm, "_", method, "_", m)
                
                # Add the corresponding error data to the DataFrame
                error_time_norm_method_disc_df[!, column_name] = error_time_norm_method_disc[:, norm_idx, method_idx, m_idx]

            end
        end
    end

    return error_time_norm_method_disc_df
end

"""
    get headers from a CSV file
"""
function get_csv_headers(file_path)
    open(file_path) do file
        return CSV.read(file, DataFrame) |> names
    end
end

"""
    update a CSV file with new data
"""
function update_csv_with_errors(    filename, 
                                    error_time_norm_method_disc_df::DataFrame,
                                    time::Vector{Float64}   )
    folder_name = "error results csv"
    file_path = joinpath(folder_name, filename)

    # Create the folder if it doesn't exist
    if !isdir(folder_name)
        mkdir(folder_name)
    end

    # Check if the file exists
    if isfile(file_path)
        # Get existing headers
        existing_headers = get_csv_headers(file_path)
        
        # Get new headers from error_time_norm_method_disc_df
        new_headers = names(error_time_norm_method_disc_df)
        
        # Find headers that are not in the existing file
        missing_headers = setdiff(new_headers, existing_headers)
        
        if !isempty(missing_headers)
            # Add missing headers to the existing file
            existing_data = CSV.read(file_path, DataFrame)
            for header in missing_headers
                existing_data[!, header] = error_time_norm_method_disc_df[!, header]
            end 
            CSV.write(file_path, existing_data)
        end
    else
        # Add time column to the DataFrame
        error_time_norm_method_disc_df[!, :time] = time
        # Create the file with headers if it doesn't exist
        CSV.write(file_path, error_time_norm_method_disc_df)
    end
end

function update_csv_with_errors(    filename, 
                                    error_time_norm_method_disc::Array{Float64, 4},
                                    time::Vector{Float64}; 
                                    norm_type = ["L1", "L2", "KL"],
                                    methods = ["DG", "DG+limiter", "DFRG"],
                                    m_vec = [8, 16, 32, 64, 128, 256, 512, 1024]   )
    folder_name = "error results csv"
    file_path = joinpath(folder_name, filename)

    # Create the folder if it doesn't exist
    if !isdir(folder_name)
        mkdir(folder_name)
    end

    error_time_norm_method_disc_df = convert_errors_to_dataframe(   error_time_norm_method_disc, 
                                                                    norm_type, 
                                                                    methods, 
                                                                    m_vec                   )
    # Check if the file exists
    if isfile(file_path)
        # Get existing headers
        existing_headers = get_csv_headers(file_path)
        
        # Get new headers from error_time_norm_method_disc_df
        new_headers = names(error_time_norm_method_disc_df)
        
        # Find headers that are not in the existing file
        missing_headers = setdiff(new_headers, existing_headers)
        
        if !isempty(missing_headers)
            # Add missing headers to the existing file
            existing_data = CSV.read(file_path, DataFrame)
            for header in missing_headers
                existing_data[!, header] = error_time_norm_method_disc_df[!, header]
            end 
            CSV.write(file_path, existing_data)
        end
    else
        # Add time column to the DataFrame
        insertcols!(error_time_norm_method_disc_df, 1, :time => time)

        # Create the file with headers if it doesn't exist
        CSV.write(file_path, error_time_norm_method_disc_df)
    end
end

"""
    load serialized file
"""
function load_serialized_file(file_name)
    if isfile(file_name)
        sd, ρ_out , t_out = open(file_name, "r") do file
            deserialize(file)
        end;
        return sd, ρ_out , t_out
    else
        println("File not found: $file_name")
        return nothing
    end
end

"""
    save serialized file
"""
function generate_error_csv_filename( type,
                            d, 
                            p, 
                            T, 
                            n_t; 
                            FR_weight_ref =true, 
                            is_exact = true)
    return "error_time_norm_method_disc_" * type * "_d_" * string(d) * "_p_" * string(p[1]) * "_T_" * replace(string(T), "." => "-") * "_n_t_" * string(n_t) * "_FR_weight_ref_" * string(FR_weight_ref) * "_is_exact_" * string(is_exact) * ".csv"
end

