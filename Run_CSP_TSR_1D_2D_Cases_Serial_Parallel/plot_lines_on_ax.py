import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def ensemble_average(x, y, delta_x):
    """
    Computes the ensemble average of y-values in bins of width delta_x along the x-axis.

    Parameters:
    x       : 1D NumPy array of x-values (assumed to be in the range [0,1]).
    y       : 1D NumPy array of corresponding y-values.
    delta_x : Bin width for partitioning x.

    Returns:
    x_centers : 1D array of bin center points.
    y_means   : 1D array of mean y-values in each bin.
    """
    # Define bin edges from 0 to 1 with step size delta_x
    bins = np.arange(0, x.max() + delta_x, delta_x)
    
    # Compute the bin centers (midpoints)
    x_centers = (bins[:-1] + bins[1:]) / 2

    # Compute mean y-values in each bin
    y_means = np.array([y[(x >= bins[i]) & (x < bins[i+1])].mean() for i in range(len(bins) - 1)])
    y_stds = np.array([y[(x >= bins[i]) & (x < bins[i+1])].std() for i in range(len(bins) - 1)])

    return x_centers, y_means,y_stds

def plot_lines_on_ax(
    ax,
    fig,
    x_list,
    y_list,
    labels=None,
    xlabel='X-axis',
    ylabel='Y-axis',
    title=None,
    show_legend=True,
    log_x=False,
    log_y=False,
    grid=True,
    line_styles=None,
    markers=None,
    save_fig_path=None,
    use_scatter=False,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None
):
    """
    Plot multiple variables on the given matplotlib Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes, target Axes to plot on
    - x: array-like, common x-axis values
    - y_list: list of array-like, each being a Y-series
    - labels: list of labels for each line
    - xlabel: str
    - ylabel: str
    - title: str
    - show_legend: bool
    - log_x: bool
    - log_y: bool
    - grid: bool
    - line_styles: list of line styles
    - markers: list of marker styles
    -save_fig_path: str complete path to save figure. 
    """
    for idx, y in enumerate(y_list):
        if labels is not None:
            label = labels[idx]
        else:
            label = None
        # label = labels[idx] if labels[idx] and idx < len(labels) else None
        style = line_styles[idx] if line_styles and idx < len(line_styles) else '-'
        marker = markers[idx] if markers and idx < len(markers) else ''
        if use_scatter:
            ax.scatter(x_list[idx], y,label=label,s=5)#, marker=marker, label=label)
            if plot_vline:
                ax.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
        else:
            ax.plot(x_list[idx], y, linestyle=style, marker=marker, label=label)
            if plot_vline:
                ax.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
            
    # if plot_line_over_scatter_x != None:
    #     ax.plot(plot_line_over_scatter_x, plot_line_over_scatter_y, linestyle=style, marker=marker, label=label)
            

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True)
    if show_legend and labels is not None:
        ax.legend()
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print("Figure saved at: %s" % save_fig_path)
    #plt.show()


def plot_lines_on_ax_errbar(
    ax,
    fig,
    x_list,
    y_list,
    error_list,
    compute_ensbld_aver_std = True,
    labels=None,
    xlabel='X-axis',
    ylabel='Y-axis',
    title=None,
    show_legend=True,
    log_x=False,
    log_y=False,
    grid=True,
    line_styles=None,
    markers=None,
    save_fig_path=None,
    use_scatter=False,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None
):
    """
    Plot multiple variables on the given matplotlib Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes, target Axes to plot on
    - x: array-like, common x-axis values
    - y_list: list of array-like, each being a Y-series
    - labels: list of labels for each line
    - xlabel: str
    - ylabel: str
    - title: str
    - show_legend: bool
    - log_x: bool
    - log_y: bool
    - grid: bool
    - line_styles: list of line styles
    - markers: list of marker styles
    -save_fig_path: str complete path to save figure. 
    """
    
    for idx, y in enumerate(y_list):
        label = labels[idx] if labels and idx < len(labels) else None
        style = line_styles[idx] if line_styles and idx < len(line_styles) else '-'
        marker = markers[idx] if markers and idx < len(markers) else ''
        if use_scatter:
            ax.scatter(x_list[idx], y,label=label)#, marker=marker, label=label)
            if plot_vline:
                ax.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
        else:
            ax.errorbar(x_list[idx], y, error_list[idx], linestyle=style, marker=marker, label=label)
            if plot_vline:
                ax.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
            
    # if plot_line_over_scatter_x != None:
    #     ax.plot(plot_line_over_scatter_x, plot_line_over_scatter_y, linestyle=style, marker=marker, label=label)
            

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True)
    if show_legend and labels:
        ax.legend()
    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print("Figure saved at: %s" % save_fig_path)
    plt.show()


def plot_dual_y_axes(
    x,
    y1,
    y2,
    fig=None,
    ax1=None,
    label1='Left Y-axis',
    label2='Right Y-axis',
    x_label='X-axis',
    title=None,
    label1_legend=None,
    label2_legend=None,
    color1='tab:blue',
    color2='tab:red',
    style1='-',
    style2='--',
    marker1='',
    marker2='',
    log_x=False,
    log_y1=False,
    log_y2=False,
    grid=True,
    show_legend=True,
    save_fig_path=None,
    use_scatter=False,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None,
    intensity_ax1 = None,
    intensity_ax2 = None
):
    """
    Plots two y-variables on a shared x-axis with left and right y-axes.

    Parameters:
    - x: array-like, shared x-axis
    - y1: array-like, left y-axis variable
    - y2: array-like, right y-axis variable
    - fig: matplotlib Figure (optional)
    - ax1: matplotlib Axes (optional)
    - label1: str, label for left y-axis
    - label2: str, label for right y-axis
    - x_label: str, label for x-axis
    - title: str, plot title
    - label1_legend: str, legend label for y1
    - label2_legend: str, legend label for y2
    - style1/style2: line styles (ignored in scatter)
    - marker1/marker2: marker styles
    - color1/color2: line colors
    - log_x/log_y1/log_y2: bool, log scale on axes
    - grid: bool, show grid
    - show_legend: bool
    - save_fig_path: str, path to save figure
    - use_scatter: bool, if True use scatter instead of line plot
    """

    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()

    # Plot first variable (left y-axis)
    if use_scatter:
        if intensity_ax1 is not None:
            line1 = ax1.scatter(x, y1, c=intensity_ax1, cmap='jet')#, marker=marker1, label=label1_legend)
            cbar = fig.colorbar(line1, ax=ax1)
            cbar.set_label("phi normalized")
            if plot_vline:
                ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
        else:
            line1 = ax1.scatter(x, y1, color=color1)#, marker=marker1, label=label1_legend)
            if plot_vline:
                ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
    else:
        line1, = ax1.plot(x, y1, linestyle=style1, marker=marker1,
                          color=color1, label=label1_legend)
        if plot_vline:
            ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
    
    # Plot second variable (right y-axis)
    if use_scatter:
        if intensity_ax2 is not None:
            line2 = ax2.scatter(x, y2, c=intensity_ax2, cmap='jet')#, marker=marker2, label=label2_legend)
            cbar = fig.colorbar(line2, ax=ax2)
            cbar.set_label("phi normalized")
        else:
            line2 = ax2.scatter(x, y2, color=color2)
    else:
        line2, = ax2.plot(x, y2, linestyle=style2, marker=marker2,
                          color=color2, label=label2_legend)
    
    if plot_line_over_scatter_x is not None and plot_line_over_scatter_y is not None:
        ax1.plot(plot_line_over_scatter_x, plot_line_over_scatter_y,color=color1)#, linestyle=style, marker=marker, label=label)
    
    # Axis labels and formatting
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(label1, color=color1)
    ax2.set_ylabel(label2, color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Log scales
    if log_x:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    if log_y1:
        ax1.set_yscale('log')
    if log_y2:
        ax2.set_yscale('log')

    if title:
        ax1.set_title(title)
    if grid:
        ax1.grid(True)

    # Legend
    if show_legend:
        # For scatter, wrap into list to be iterable
        lines = [line1, line2]
        labels = [label1_legend, label2_legend]
        ax1.legend(lines, labels, loc='best')

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print("Figure saved at: %s" % save_fig_path)

    #plt.show()
    
    
def plot_single_y_axes(
    x,
    y1,
    fig=None,
    ax1=None,
    label1='Left Y-axis',
    x_label='X-axis',
    title=None,
    label1_legend=None,
    color1='tab:blue',
    style1='-',
    marker1='',
    log_x=False,
    log_y1=False,
    grid=True,
    show_legend=True,
    save_fig_path=None,
    use_scatter=False,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None,
    intensity_ax1 = None,
):
    """
    Plots single y-variables on a shared x-axis with left and right y-axes.

    Parameters:
    - x: array-like, shared x-axis
    - y1: array-like, left y-axis variable
    - fig: matplotlib Figure (optional)
    - ax1: matplotlib Axes (optional)
    - label1: str, label for left y-axis
    - x_label: str, label for x-axis
    - title: str, plot title
    - label1_legend: str, legend label for y1
    - style1/style2: line styles (ignored in scatter)
    - marker1/marker2: marker styles
    - color1/color2: line colors
    - log_x/log_y1/log_y2: bool, log scale on axes
    - grid: bool, show grid
    - show_legend: bool
    - save_fig_path: str, path to save figure
    - use_scatter: bool, if True use scatter instead of line plot
    """
    
    if  intensity_ax1 is not None:
        print(intensity_ax1.min(),intensity_ax1.max())
        norm = mcolors.TwoSlopeNorm(vmin=intensity_ax1.min(), vcenter=1.0, vmax=intensity_ax1.max())
    
    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots(figsize=(8, 6))


    # Plot first variable (left y-axis)
    if use_scatter:
        if intensity_ax1 is not None:
            line1 = ax1.scatter(x, y1, c=intensity_ax1, cmap='jet',norm=norm)#, marker=marker1, label=label1_legend)
            cbar = fig.colorbar(line1, ax=ax1)
            cbar.set_label("phi normalized")
            if plot_vline:
                ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
        else:
            line1 = ax1.scatter(x, y1, color=color1)#, marker=marker1, label=label1_legend)
            if plot_vline:
                ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
    else:
        line1, = ax1.plot(x, y1, linestyle=style1, marker=marker1,
                          color=color1, label=label1_legend)
        if plot_vline:
            ax1.axvline(x=plot_vline,color='k', linestyle='--', linewidth=2)
    
    if plot_line_over_scatter_x is not None and plot_line_over_scatter_y is not None:
        ax1.plot(plot_line_over_scatter_x, plot_line_over_scatter_y,color=color1)#, linestyle=style, marker=marker, label=label)
    
    # Axis labels and formatting
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Log scales
    if log_x:
        ax1.set_xscale('log')
    if log_y1:
        ax1.set_yscale('log')
    if title:
        ax1.set_title(title)
    if grid:
        ax1.grid(True)

    # Legend
    if show_legend:
        # For scatter, wrap into list to be iterable
        lines = [line1]
        labels = [label1_legend]
        ax1.legend(lines, labels, loc='best')

    if save_fig_path:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print("Figure saved at: %s" % save_fig_path)

    plt.show()    

# %%


# import numpy as np

# x = np.linspace(1, 100, 100)
# y1 = x ** 0.5
# y2 = x
# y3 = x ** 2

# fig, ax = plt.subplots(figsize=(10, 6))
# plot_lines_on_ax(
#     ax,
#     fig,
#     [x,x,x],
#     [y1, y2, y3],
#     labels=["sqrt(x)", "x", "x^2"],
#     xlabel="X",
#     ylabel="Y",
#     title="Log-Log Plot",
#     log_x=True,
#     log_y=True,
#     line_styles=["--", "-", ":"],
#     markers=["o", "", "s"]
# )
# plt.tight_layout()
# plt.show()
# # %%