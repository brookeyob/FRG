using GLMakie

"""
    Update y-axis limit for plot
"""
function update_y_limits!(ax, values)
    max = maximum(values)  # Find the maximum value in the array
    min = minimum(values)  # Find the minimum value in the array
    diff = 0.1 * abs(max - min)  # Calculate a buffer based on 10% of the range
    if abs(diff) <= 1e-14  # Handle cases where the range is very small or zero
        diff = 0.1 * abs(min)  # Use 10% of the minimum value as a fallback
        if diff == 0  # If both min and diff are zero, use 10% of the maximum value
            diff = 0.1 * abs(max)
        end
    end
    ylims!(ax, min - diff, max + diff)  # Update the y-axis limits with the buffer
end

"""
    Plot conservation
"""
function plot_conservation(sd::AbstractSemidiscretization, q)
    q_t = compute_conservation(sd, q)  # Compute conservation values over time

    nt = size(q_t, 1)  # Number of time steps
    Δt = sd.Δt  # Time step size
    ts = collect(1:nt) * Δt  # Generate time points for the x-axis

    fig = Figure()  # Create a new figure
    ax = Axis(fig[1, 1], title="Total", ylabel="total", xlabel="x")  # Create an axis
    lines!(ax, ts, q_t .- q_t[1])  # Plot the deviation from the initial value

    return fig  # Return the figure
end

"""
    Initialize plot points
"""
function initialize_plot_array(n_plot, Δx_plot, L)
    return product([0:Δ:l for (l, Δ) in zip(L, Δx_plot)]...)  # Generate a Cartesian product of points
end
