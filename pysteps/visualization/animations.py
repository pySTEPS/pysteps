"""
pysteps.visualization.animations
================================

Functions to produce animations for pysteps.

.. autosummary::
    :toctree: ../generated/

    animate
"""

import matplotlib.pylab as plt
import numpy as np
import pysteps as st

# TODO: Add documentation for the output files.
def animate(R_obs, nloops=2, timestamps=None, R_fct=None, timestep_min=5,
            UV=None, motion_plot="quiver", geodata=None, map=None,
            colorscale="pysteps", units="mm/h", colorbar=True, type="ensemble",
            prob_thr=None, plotanimation=True, savefig=False,
            fig_dpi=150, fig_format="png", path_outputs="", **kwargs):
    """Function to animate observations and forecasts in pysteps.

    Parameters
    ----------
    R_obs : array-like
        Three-dimensional array containing the time series of observed
        precipitation fields.

    Other parameters
    ----------------
    nloops : int
        Optional, the number of loops in the animation.
    R_fct : array-like
        Optional, the three or four-dimensional (for ensembles) array containing
        the time series of forecasted precipitation field.
    timestep_min : float
        The time resolution in minutes of the forecast.
    UV : array-like
        Optional, the motion field used for the forecast.
    motion_plot : string
        The method to plot the motion field.
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+----------------------------------------------------+
        |        Key      |                  Value                             |
        +=================+====================================================+
        |   projection    | PROJ.4-compatible projection definition            |
        +-----------------+----------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    yorigin      | a string specifying the location of the first      |
        |                 | element in the data raster w.r.t. y-axis:          |
        |                 | 'upper' = upper border, 'lower' = lower border     |
        +-----------------+----------------------------------------------------+

    map : str
        Optional method for plotting a map.
        See pysteps.visualization.precipifields.plot_precip.field.
    units : str
        Units of the input array (mm/h or dBZ)
    colorscale : str
        Which colorscale to use.
    title : str
        If not None, print the title on top of the plot.
    colorbar : bool
        If set to True, add a colorbar on the right side of the plot.
    type : {'ensemble', 'mean', 'prob'}, str
        Type of the map to animate. 'ensemble' = ensemble members, 
        'mean' = ensemble mean, 'prob' = exceedance probability (using threshold
        defined in prob_thrs).
    prob_thr : float
        Intensity threshold for the exceedance probability maps. Applicable
        if type = 'prob'.
    plotanimation : bool
        If set to True, visualize the animation (useful when one is only interested
        in saving the individual frames).
    savefig : bool
        If set to True, save the individual frames to path_outputs.
    fig_dpi : scalar > 0
        Resolution of the output figures, see the documentation of
        matplotlib.pyplot.savefig. Applicable if savefig is True.
    path_outputs : string
        Path to folder where to save the frames.
    kwargs : dict
        Optional keyword arguments that are supplied to plot_precip_field
        and quiver/streamplot.

    Returns
    -------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """
    if timestamps is not None:
        startdate_str = timestamps[-1].strftime("%Y%m%d%H%M")
    else:
        startdate_str = None

    if R_fct is not None:
        if len(R_fct.shape) == 3:
            R_fct = R_fct[None, :, :, :]

    if R_fct is not None:
        n_lead_times = R_fct.shape[1]
        n_members = R_fct.shape[0]
    else:
        n_lead_times = 0
        n_members = 1
    
    if type == "prob" and prob_thr is None:
        raise ValueError("type 'prob' needs a prob_thr value")
    
    if type != "ensemble":
        n_members = 1
        
    n_obs = R_obs.shape[0]

    loop = 0
    while loop < nloops:
        for n in range(n_members):
            for i in range(n_obs + n_lead_times):
                plt.clf()

                # Observations
                if i < n_obs and (plotanimation or n == 0):
                    
                    title = ""
                    if timestamps is not None:
                        title += timestamps[i].strftime("%Y-%m-%d %H:%M\n")
                        
                    plt.clf()
                    if type == "prob":
                        title += "Observed Probability"
                        R_obs_ = R_obs[np.newaxis, ::]
                        P_obs = st.postprocessing.ensemblestats.excprob(R_obs_[:, i, :, :], prob_thr)
                        ax = st.plt.plot_precip_field(P_obs, type="prob", map=map,
                                                 geodata=geodata, units=units,
                                                 probthr=prob_thr, title=title,
                                                 **kwargs)
                    else:
                        title += "Observed Rainfall"
                        ax = st.plt.plot_precip_field(R_obs[i,:,:], map=map,
                            geodata=geodata, units=units, colorscale=colorscale,
                            title=title, colorbar=colorbar, **kwargs)
                        
                    if UV is not None and motion_plot is not None:
                        if motion_plot.lower() == "quiver":
                            st.plt.quiver(UV, ax=ax, geodata=geodata, **kwargs)
                        elif motion_plot.lower() == "streamplot":
                            st.plt.streamplot(UV, ax=ax, geodata=geodata, **kwargs)
                            
                    if savefig & (loop == 0):
                        if type == "prob":
                            figname = "%s/%s_frame_%02d_binmap_%.1f.%s" % \
                                    (path_outputs, startdate_str, i, prob_thr, fig_format)
                        else:
                            figname = "%s/%s_frame_%02d.%s" % \
                                (path_outputs, startdate_str, i, fig_format)
                        plt.savefig(figname, bbox_inches="tight", dpi=fig_dpi)
                        print(figname, 'saved.')

                # Forecasts
                elif i >= n_obs and R_fct is not None:

                    title = ""
                    if timestamps is not None:
                        title += timestamps[-1].strftime("%Y-%m-%d %H:%M\n")
                    leadtime = "+%02d min" % ((1 + i - n_obs)*timestep_min)
                    
                    plt.clf()
                    if type == "prob":
                        title += "Forecast Probability"
                        P = st.postprocessing.ensemblestats.excprob(R_fct[:, i - n_obs, :, :], prob_thr)
                        ax = st.plt.plot_precip_field(P, type="prob", map=map,
                                                 geodata=geodata, units=units,
                                                 probthr=prob_thr, title=title,
                                                 **kwargs)    
                    elif type == "mean":
                        title += "Forecast Ensemble Mean"
                        EM = st.postprocessing.ensemblestats.mean(R_fct[:, i - n_obs, :, :])
                        ax = st.plt.plot_precip_field(EM, map=map, geodata=geodata, units=units,
                                                 title=title, colorscale=colorscale,
                                                 colorbar=colorbar, **kwargs)
                    else:
                        title += "Forecast Rainfall"
                        ax = st.plt.plot_precip_field(R_fct[n, i - n_obs, :, :], map=map,
                                      geodata=geodata, units=units, title=title,
                                      colorscale=colorscale, colorbar=colorbar,
                                      **kwargs)
               
                    if UV is not None and motion_plot is not None:
                        if motion_plot.lower() == "quiver":
                            st.plt.quiver(UV, ax=ax, geodata=geodata, **kwargs)
                        elif motion_plot.lower() == "streamplot":
                            st.plt.streamplot(UV, ax=ax, geodata=geodata, **kwargs)
                    
                    if leadtime is not None:
                        plt.text(0.99, 0.99, leadtime, transform=ax.transAxes,
                                 ha="right", va="top")
                    if type == "ensemble" and n_members > 1:
                        plt.text(0.01, 0.99, "m %02d" % (n+1), transform=ax.transAxes,
                                 ha="left", va="top")
                        
                    if savefig & (loop == 0):
                        if type == "prob":
                            figname = "%s/%s_frame_%02d_probmap_%.1f.%s" % \
                                (path_outputs, startdate_str, i, prob_thr, fig_format)
                        elif type == "mean":
                            figname = "%s/%s_frame_%02d_ensmean.%s" % \
                                (path_outputs, startdate_str, i, fig_format)
                        else:
                            figname = "%s/%s_member_%02d_frame_%02d.%s" % \
                                (path_outputs, startdate_str, (n+1), i, fig_format)
                        plt.savefig(figname, bbox_inches="tight", dpi=fig_dpi)
                        print(figname, "saved.")

                if plotanimation:
                    plt.pause(.2)

            if plotanimation:
                plt.pause(.5)

        loop += 1

    plt.close()
