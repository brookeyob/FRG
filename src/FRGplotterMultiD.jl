using Statistics: mean
"""
    Plot DG solution for transport equation 
""" 
# Function to plot 2D FRG solution with density and velocity
function plot_2D_FRG_solution(  sd::FRGSemidiscretizationDGMultiD, 
                                ρT, 
                                u0, 
                                xs, 
                                ts; 
                                x_lim=nothing, 
                                y_lim=nothing,
                                levels=15   )

    # Extract parameters from the semidiscretization object
    d = sd.d
    m = sd.m
    p = sd.p
    Δx = sd.Δx 
    L = Δx .* m

    # Extract x and y coordinates from the grid points
    x_vector = [ z[1] for z in xs[:,1] ]
    y_vector = [ z[2] for z in xs[1,:] ]

    # Extract velocity components and compute their magnitude
    ux = [ z[1] for z in u0 ]
    uy = [ z[2] for z in u0 ]
    strength = vec(sqrt.(ux .^ 2 .+ uy .^ 2))

    u_mag = zeros(size(u0))
    for (i,u_i) in enumerate(u0)
        u_mag[i] = norm(u_i)
    end

    # Create a figure and axes for density and velocity
    fig = Figure()
    ax_ρ = Axis(fig[1, 1], title="Density", ylabel="y", xlabel="x",xgridvisible = false,ygridvisible = false)
    ax_u = Axis(fig[1, 3], title="Velocity", ylabel="y", xlabel="x", backgroundcolor = "white",xgridvisible = false,ygridvisible = false)

    # Plot velocity vectors as arrows
    co_u = arrows!(x_vector, y_vector, ux, uy, arrowsize = 10, lengthscale = 0.1,arrowcolor = strength, linecolor = strength)
    Colorbar(fig[1, 4], co_u)

    # Set axis limits if provided
    if x_lim != nothing 
        xlims!( ax_ρ , x_lim[1] , x_lim[2] )
        xlims!( ax_u , x_lim[1] , x_lim[2] )
    end
    if y_lim != nothing 
        ylims!( ax_ρ , y_lim[1] , y_lim[2] )
        ylims!( ax_u , y_lim[1] , y_lim[2] )
    end

    # Determine density limits for contour plot
    ρ_values = reshape(reduce(vcat,ρT),:)
    min = minimum(ρ_values)
    max = maximum(ρ_values)
    if abs(min - max) <= 1e-12
        limit = (min-0.1,max+0.1)
    else
        diff = max - min
        limit = (min-diff*0.1,max+diff*0.1)
    end

    # Observable for time slider
    time = Observable(1)
    nT = length(ts)

    # Create a contour plot for density
    ρ_cont = @lift(ρT[$time])
    co_ρ = contourf!(ax_ρ,x_vector,y_vector,ρ_cont,levels = range(limit[1], limit[2], length = levels))
    Colorbar(fig[1, 2], co_ρ)

    # Define a function to format time for the slider
    idx_to_time(idx) = string(round( ts[idx]; digits =6 )," s")

    # Add a slider for time navigation
    slg = SliderGrid( fig[2, 1:4], (; range = 1:1:nT, label = "Time",format = idx_to_time) )
    on(slg.sliders[1].value) do v
        time[] = v
    end

    return fig
end

function plot_2D_surface_FRG_solution(  sd::FRGSemidiscretizationDGMultiD, 
                                        ρT, 
                                        u0, 
                                        xs, 
                                        ts   )

    # Extract parameters from the semidiscretization object
    d = sd.d
    m = sd.m
    p = sd.p
    Δx = sd.Δx 
    L = Δx .* m

    # Extract x and y coordinates from the grid points
    x_vector = [ x[1] for x in xs ]
    y_vector = [ x[2] for x in xs ]

    # Create a figure and 3D axis for density
    fig = Figure()
    ax_ρ = Axis3(fig[1, 1], title="Density", ylabel="y", xlabel="x",xgridvisible = false,ygridvisible = false)

    # Determine density limits for surface plot
    ρ_values = reshape(reduce(vcat,ρT),:)
    min = minimum(ρ_values)
    max = maximum(ρ_values)
    if abs(min - max) <= 1e-12
        limit = (min-0.1,max+0.1)
    else
        diff = max - min
        limit = (min-diff*0.1,max+diff*0.1)
    end

    time = Observable(1)
    nT = length(ts)

    # Create a surface plot for density
    ρ_cont = @lift(ρT[$time])
    co_ρ = surface!(ax_ρ,x_vector,y_vector,ρ_cont)
    Colorbar(fig[1, 2], co_ρ)

    # Define a function to format time for the slider
    idx_to_time(idx) = string(round( ts[idx]; digits=6 )," s")

    # Add a slider for time navigation
    slg = SliderGrid( fig[2, 1:2], (; range = 1:1:nT, label = "Time",format = idx_to_time) )
    on(slg.sliders[1].value) do v
        time[] = v
    end

    return fig
end

function plot_1D_FRG_solution(  sd::FRGSemidiscretizationDGMultiD, 
                                ρT, 
                                u0, 
                                xs, 
                                ts; 
                                x_lim=nothing, 
                                y_lim=nothing   )

    # Extract parameters from the semidiscretization object
    d = sd.d
    m = sd.m
    p = sd.p
    Δx = sd.Δx 
    L = Δx .* m

    # Extract x coordinates from the grid points
    x_vector = [ z[1] for z in xs[:,1] ]

    # Compute velocity magnitude
    u_mag = zeros(size(u0))
    for (i,u_i) in enumerate(u0)
        u_mag[i] = u_i[1]
    end

    # Create a figure and axes for density and velocity
    fig = Figure()
    ax_ρ = Axis(fig[1, 1], title="Density", xlabel="x")
    ax_u = Axis(fig[1, 2], title="Velocity", xlabel="x") 

    # Set axis limits if provided
    if x_lim != nothing 
        xlims!( ax_ρ , x_lim[1] , x_lim[2] )
        xlims!( ax_u , x_lim[1] , x_lim[2] )
    end
    if y_lim != nothing 
        ylims!( ax_ρ , y_lim[1] , y_lim[2] )
        ylims!( ax_u , y_lim[1] , y_lim[2] )
    else
        update_y_limits!( ax_ρ, reshape(reduce(vcat,ρT),:) )
        update_y_limits!( ax_u, reshape(u_mag[:,1],:) )
    end

    # Plot velocity and density lines
    lines!(ax_u,x_vector,u_mag[:,1])
    time = Observable(1)
    nT = length(ts)
    ρ_cont = @lift(ρT[$time][:,1])
    lines!(ax_ρ,x_vector,ρ_cont)

    # Define a function to format time for the slider
    idx_to_time(idx) = string(round( ts[idx]; digits = 6 )," s")

    # Add a slider for time navigation
    slg = SliderGrid( fig[2, 1:2], (; range = 1:1:nT, label = "Time",format = idx_to_time) )
    on(slg.sliders[1].value) do v
        time[] = v
    end

    return fig
end
# Function to plot 1D FRG solution with density and velocity, including a reference solution
function plot_1D_FRG_solution(  sd::FRGSemidiscretizationDGMultiD,
                                ρT, 
                                ρT_ref, 
                                u0, 
                                xs, 
                                ts; 
                                x_lim=nothing, 
                                y_lim=nothing   )

    # Extract parameters from the semidiscretization object
    d = sd.d
    m = sd.m
    p = sd.p
    Δx = sd.Δx 
    L = Δx .* m

    # Extract x coordinates from the grid points
    x_vector = [ z[1] for z in xs[:,1] ]

    u_mag = zeros(size(u0))
    for (i,u_i) in enumerate(u0)
        u_mag[i] = u_i[1]
    end

    # Create a new figure for plotting
    fig = Figure()

    # Create axes for density and velocity plots
    ax_ρ = Axis(fig[1, 1], title="Density",  xlabel="x")
    ax_u = Axis(fig[1, 3], title="Velocity",  xlabel="x") 

    # Set limits for x-axis and y-axis if provided
    if x_lim != nothing 
        xlims!( ax_ρ , x_lim[1] , x_lim[2] )
        xlims!( ax_u , x_lim[1] , x_lim[2] )
    end
    if y_lim != nothing 
        ylims!( ax_ρ , y_lim[1] , y_lim[2] )
        ylims!( ax_u , y_lim[1] , y_lim[2] )
    else
        # Automatically update y-axis limits based on data
        update_y_limits!( ax_ρ, reshape(reduce(vcat,ρT),:) )
        update_y_limits!( ax_u, reshape(u_mag[:,1],:) )
    end

    # Plot velocity magnitude as a line
    lines!(ax_u,x_vector,u_mag[:,1])

    # Create an observable for time slider
    time = Observable(1)

    # Number of time steps
    nT = length(ts)

    # Create observables for density and reference density at the current time
    ρ_cont = @lift(ρT[$time][:,1])
    ρ_ref_cont = @lift(ρT_ref[$time][:,1])

    # Plot density and reference density as lines
    ρ_cont_line = lines!(ax_ρ,x_vector,ρ_cont,label="Test")
    ρ_ref_cont_line = lines!(ax_ρ,x_vector,ρ_ref_cont,label="Reference")

    # Add a legend for the density plots
    Legend( fig[1, 2],
            [ρ_cont_line, ρ_ref_cont_line],
            ["Test", "Reference"])

    # Define a function to format time for the slider
    idx_to_time(idx) = string(round( ts[idx]; digits = 6 )," s")

    # Add a slider for time navigation
    slg = SliderGrid( fig[2, 1:3], (; range = 1:1:nT, label = "Time",format = idx_to_time) )

    # Update the observable when the slider value changes
    on(slg.sliders[1].value) do v
        time[] = v
    end

    return fig
end

function plot_1D_FRG_solution(  sd::FRGSemidiscretizationDGMultiD,
                                ρT_vec::Vector{Vector{Array{Float64}}}, 
                                u0, 
                                xs, 
                                ts; 
                                labels=["Test"],
                                x_lim=nothing, 
                                y_lim=nothing   )

    # Extract parameters from the semidiscretization object
    d = sd.d
    m = sd.m
    p = sd.p
    Δx = sd.Δx 
    L = Δx .* m

    # Extract x coordinates from the grid points
    x_vector = [ z[1] for z in xs[:,1] ]

    # Compute velocity magnitude
    u_mag = zeros(size(u0))
    for (i,u_i) in enumerate(u0)
        u_mag[i] = u_i[1]
    end

    # Create a figure and axes for density and velocity
    fig = Figure()

    ax_ρ = Axis(fig[1, 1], title="Density", xlabel="x")
    ax_u = Axis(fig[1, 3], title="Velocity",  xlabel="x") 

    # Set limits for x-axis and y-axis if provided
    if x_lim != nothing 
        xlims!( ax_ρ , x_lim[1] , x_lim[2] )
        xlims!( ax_u , x_lim[1] , x_lim[2] )
    end
    if y_lim != nothing 
        ylims!( ax_ρ , y_lim[1] , y_lim[2] )
        ylims!( ax_u , y_lim[1] , y_lim[2] )
    else
        # Automatically update y-axis limits based on data
        update_y_limits!( ax_ρ, reshape(reduce(vcat,ρT_vec[1]),:) )
        update_y_limits!( ax_u, reshape(u_mag[:,1],:) )
    end

    # Plot velocity magnitude as a line
    lines!(ax_u,x_vector,u_mag[:,1])

    # Create an observable for time slider
    time = Observable(1)

    # Number of time steps
    nT = length(ts)

    # Initialize containers for density observables and plot lines
    ρT_cont_vec = []
    ρT_line_vec = []

    # Loop through each density vector and create observables and plot lines
    for (i,ρT) in enumerate(ρT_vec)

        ρ_cont = @lift(ρT[$time][:,1])
        ρ_cont_line = lines!(ax_ρ,x_vector,ρ_cont,label=labels[i])
        push!(ρT_cont_vec, ρ_cont)
        push!(ρT_line_vec, ρ_cont_line)

    end

    # Add a legend for the density plots
    Legend( fig[1, 2],
            ρT_line_vec,
            labels)

    # Define a function to format time for the slider
    idx_to_time(idx) = string(round( ts[idx]; digits = 6 )," s")

    # Add a slider for time navigation
    slg = SliderGrid( fig[2, 1:3], (; range = 1:1:nT, label = "Time",format = idx_to_time) )

    # Update the observable when the slider value changes
    on(slg.sliders[1].value) do v
        time[] = v
    end

    return fig
end

"""
    Evaluate plot points 
""" 
function evaluate_plot_points(   sd::FRGSemidiscretizationDGMultiD,
                                x_plot,
                                ρ,
                                u,
                                t_out   )

    # Extract the dimensionality of the problem
    d = sd.d

    # Create a basis object for the semidiscretization
    basis = FRGSemidiscretizationMultiDBasis(sd)

    # Initialize arrays to store density and velocity at plot points
    ρ_plot = Array{Float64}[]
    u_plot = [ zeros(d) for i in x_plot ]

    # Evaluate velocity at each plot point
    for (i, x) in enumerate(x_plot)
        u_plot[i] = evaluate_velocity(basis, u, x)
    end

    # Evaluate density at each plot point for each output time
    for (j, t) in enumerate(t_out)
        ρ_plot_t = zeros(size(x_plot))
        for (i, x) in enumerate(x_plot)
            ρ_plot_t[i] = evaluate_density(basis, ρ[j], x)
        end
        push!(ρ_plot, ρ_plot_t)
    end

    return ρ_plot, u_plot
end

# Function to evaluate density at plot points for given times
function evaluate_plot_points(  sd::FRGSemidiscretizationDGMultiD,
                                x_plot,
                                ρ,
                                t_out   )

    # Create a basis object for the semidiscretization
    basis = FRGSemidiscretizationMultiDBasis(sd)

    # Initialize an array to store density at plot points
    ρ_plot = Array{Float64}[]

    # Evaluate density at each plot point for each output time
    for (j, t) in enumerate(t_out)
        ρ_plot_t = zeros(size(x_plot))
        for (i, x) in enumerate(x_plot)
            ρ_plot_t[i] = evaluate_density(basis, ρ[j], x)
        end
        push!(ρ_plot, ρ_plot_t)
    end

    return ρ_plot
end

"""
    Method color map
"""
function method_custom_colormap()
    colors = ["#FBDC7F", "#D98C21", "#B8420F"]
    return cgrad(colors, categorical=true)
end

"""
    Plot error vs discretization
"""
function plot_log_error_vs_disc(    error_time_norm_method_disc, 
                                    m_vec, 
                                    methods; 
                                    type = "bump"   )

    # Create a figure for plotting
    fig = Figure()

    ncolor = size(methods, 1)
    method_colormap = method_custom_colormap()
    method_marker = [:xcross, :utriangle, :rect]

    # Create axes for different error types
    ax_L1 = Axis(fig[1, 1], title="L1 error", ylabel="L1 error", xlabel="cells", xscale=log2, yscale=log10, xgridvisible=false, ygridvisible=false)
    ax_L2 = Axis(fig[1, 2], title="L2 error", ylabel="error", xlabel="cells", xscale=log2, yscale=log10, xgridvisible=false, ygridvisible=false)
    ax_KL = Axis(fig[1, 3], title="KL error", ylabel="KL error", xlabel="cells", xscale=log2, yscale=log10, xgridvisible=false, ygridvisible=false)

    # Loop through each method and plot errors
    for (method_idx, method) in enumerate(methods)
        L1_error = mean(error_time_norm_method_disc[:, 1, method_idx, :], dims=1)[:]
        L2_error = mean(error_time_norm_method_disc[:, 2, method_idx, :], dims=1)[:]
        KL_error = mean(abs.(error_time_norm_method_disc[:, 3, method_idx, :]), dims=1)[:] 

        # Plot lines and scatter points for each error type
        lines!(ax_L1, m_vec, L1_error, label=method, color=method_colormap[method_idx])
        scatter!(ax_L1, m_vec, L1_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        lines!(ax_L2, m_vec, L2_error, label=method, color=method_colormap[method_idx])
        scatter!(ax_L2, m_vec, L2_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        if method != "DG+limiter"
            lines!(ax_KL, m_vec, KL_error, label=method, color=method_colormap[method_idx])
            scatter!(ax_KL, m_vec, KL_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        end
    end

    # Add a legend to the figure
    Legend(fig[1, 4], ax_L2)
    resize!(fig, 750, 250)

    return fig
end



"""
    Plot error vs cells without scale
"""
function plot_error_vs_disc(    error_time_norm_method_disc, 
                                m_vec, 
                                methods; 
                                type = "bump"   )

    # Create a figure for plotting
    fig = Figure()

    ncolor = size(methods, 1)
    method_colormap = method_custom_colormap()
    method_marker = [:xcross, :utriangle, :rect]

    # Create axes for different error types
    ax_L1 = Axis(fig[1, 1], title="L1 error", ylabel="L1 error", xlabel="cells", xscale=log2, xgridvisible=false, ygridvisible=false)
    ax_L2 = Axis(fig[2, 1], title="L2 error", ylabel="L2 error", xlabel="cells", xscale=log2, xgridvisible=false, ygridvisible=false)
    ax_KL = Axis(fig[1, 2], title="KL error", ylabel="KL error", xlabel="cells", xscale=log2, xgridvisible=false, ygridvisible=false)

    # Loop through each method and plot errors
    for (method_idx, method) in enumerate(methods)
        L1_error = mean(error_time_norm_method_disc[:, 1, method_idx, :], dims=1)[:]
        L2_error = mean(error_time_norm_method_disc[:, 2, method_idx, :], dims=1)[:]
        KL_error = mean(abs.(error_time_norm_method_disc[:, 3, method_idx, :]), dims=1)[:] 

        # Plot lines for each error type
        lines!(ax_L1, m_vec, L1_error, label=method, color=method_colormap[method_idx])
        scatter!(ax_L1, m_vec, L1_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        lines!(ax_L2, m_vec, L2_error, label=method, color=method_colormap[method_idx])
        scatter!(ax_L2, m_vec, L2_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        if method != "DG+limiter"
            lines!(ax_KL, m_vec, KL_error, label=method, color=method_colormap[method_idx])
            scatter!(ax_KL, m_vec, KL_error, color=method_colormap[method_idx], marker=method_marker[method_idx])
        end
    end

    

    # Add a legend to the figure
    Legend(fig[1, 4], ax_L2)
    resize!(fig, 750, 250)

    return fig
end

"""
    Plot errors for a given cells
"""
function plot_log_errors_given_disc(    error_time_norm_method_disc, 
                                        m_vec, 
                                        m_idx, 
                                        methods,
                                        time    )

    # Create a figure for plotting
    fig = Figure()
    Label(fig[1, 1:3], "Errors for m=$(m_vec[m_idx])", fontsize = 24, halign = :center)
    ncolor = size(methods, 1)
    method_colormap = method_custom_colormap()
    method_marker = [:xcross, :utriangle, :rect]

    # Create axes for different error types
    ax_L1 = Axis(fig[2, 1], title="L1 error", ylabel="L1 error", xlabel="time", yscale=log10, xgridvisible = false, ygridvisible = false)
    ax_L2 = Axis(fig[2, 2], title="L2 error", ylabel="L2 error", xlabel="time", yscale=log10, xgridvisible = false, ygridvisible = false)
    ax_KL = Axis(fig[2, 3], title="KL error", ylabel="KL error", xlabel="time", yscale=log10, xgridvisible = false, ygridvisible = false)

    # Loop through each method and plot errors
    for (method_idx, method) in enumerate(methods)
        L1_error = error_time_norm_method_disc[:, 1, method_idx, m_idx]
        L2_error = error_time_norm_method_disc[:, 2, method_idx, m_idx]
        KL_error = abs.(error_time_norm_method_disc[:, 3, method_idx, m_idx])

        # Plot lines for each error type
        lines!(ax_L1, time, L1_error, label=method, color=method_colormap[method_idx])
        lines!(ax_L2, time, L2_error, label=method, color=method_colormap[method_idx])
        if method != "DG+limiter"
            lines!(ax_KL, time, KL_error, label=method, color=method_colormap[method_idx])
        end
    end

    # Add a legend to the figure
    Legend(fig[2, 4], ax_L2)
    resize!(fig, 750, 250)

    return fig
    
end

"""
    Plot errors for a given cells without scale
"""
function plot_errors_given_disc(    error_time_norm_method_disc, 
                                    m_vec, 
                                    m_idx, 
                                    methods,
                                    time    )

    # Create a figure for plotting
    fig = Figure()
    Label(fig[1, 1:3], "Errors for m=$(m_vec[m_idx])", fontsize = 24, halign = :center)
    ncolor = size(methods, 1)
    method_colormap = method_custom_colormap()
    method_marker = [:xcross, :utriangle, :rect]

    # Create axes for different error types
    ax_L1 = Axis(fig[2, 1], title="L1 error", ylabel="L1 error", xlabel="time", xgridvisible = false, ygridvisible = false)
    ax_L2 = Axis(fig[2, 2], title="L2 error", ylabel="L2 error", xlabel="time", xgridvisible = false, ygridvisible = false)
    ax_KL = Axis(fig[2, 3], title="KL error", ylabel="KL error", xlabel="time", xgridvisible = false, ygridvisible = false)

    # Loop through each method and plot errors
    for (method_idx, method) in enumerate(methods)
        L1_error = error_time_norm_method_disc[:, 1, method_idx, m_idx]
        L2_error = error_time_norm_method_disc[:, 2, method_idx, m_idx]
        KL_error = abs.(error_time_norm_method_disc[:, 3, method_idx, m_idx])

        # Plot lines for each error type
        lines!(ax_L1, time, L1_error, label=method, color=method_colormap[method_idx])
        lines!(ax_L2, time, L2_error, label=method, color=method_colormap[method_idx])
        if method != "DG+limiter"
            lines!(ax_KL, time, KL_error, label=method, color=method_colormap[method_idx])
        end
    end

    # Add a legend to the figure
    Legend(fig[2, 4], ax_L2)
    resize!(fig, 750, 250)

    return fig
end
