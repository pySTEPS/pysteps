"""Functions to produce animations for pysteps."""

import matplotlib.pylab as plt
import numpy as np
import pysteps as st

def animate(R_obs, nloops=2, timestamps=None, R_fct=None, timestep_min=5, 
            UV=None, motion_plot="quiver", geodata=None, colorscale="MeteoSwiss", 
            units="mm/h", colorbar=True, plotanimation=True, savefig=False, 
            path_outputs=""):
    """Function to animate observations and forecasts in pysteps.
    
    Parameters
    ----------
    R_obs : array-like
        Three-dimensional array containing the time series of observed
        precipitation field.
    
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
    units : str
        Units of the input array (mm/h or dBZ)
    colorscale : str
        Which colorscale to use.
    title : str
        If not None, print the title on top of the plot.
    colorbar : bool
        If set to True, add a colorbar on the right side of the plot.
    plotanimation : bool
        If set to True, visualize the animation (useful when one is only interested
        in saving the individual frames).
    savefig : bool
        If set to True, save the individual frames to path_outputs.
    path_outputs : string
        Path to folder where to save the frames.

    Returns
    -------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    if timestamps is not None:
        startdate_str = timestamps[-1].strftime("%Y-%m-%d %H:%M")
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

    loop = 0
    while loop < nloops:

        for n in range(n_members):

            for i in range(R_obs.shape[0] + n_lead_times):
                plt.clf()

                # Observations
                if i < R_obs.shape[0]:

                    if timestamps is not None:
                        title = timestamps[i].strftime("%Y-%m-%d %H:%M")
                    else:
                        title = None

                    st.plt.plot_precip_field(R_obs[i,:,:], geodata=geodata,
                                  units=units, colorscale=colorscale,
                                  title=title,
                                  colorbar=colorbar)
                    if UV is not None and motion_plot is not None:
                        if motion_plot.lower() == "quiver":
                            st.plt.quiver(UV, geodata)
                        elif motion_plot.lower() == "streamplot":
                            st.plt.streamplot(UV, geodata)
                    if savefig & (loop == 0):
                        figname = "%s/%s_frame_%02d.png" % (path_outputs, startdate_str, i)
                        plt.savefig(figname)
                        print(figname, 'saved.')

                # Forecasts
                elif i >= R_obs.shape[0] and R_fct is not None:

                    if timestamps is not None:
                        title = "%s +%02d min" % (timestamps[-1].strftime("%Y-%m-%d %H:%M"),
                                (1 + i - R_obs.shape[0])*timestep_min)
                    else:
                        title = "+%02d min" % ((1 + i - R_obs.shape[0])*timestep_min)

                    if n_members > 1:
                        title = "%s (member %02d)" % (title, n)


                    st.plt.plot_precip_field(R_fct[n, i - R_obs.shape[0],:,:], 
                                  geodata=geodata, units=units,
                                  title=title,
                                  colorscale=colorscale, colorbar=colorbar)
                    if UV is not None and motion_plot is not None:
                        if motion_plot.lower() == "quiver":
                            st.plt.quiver(UV, geodata)
                        elif motion_plot.lower() == "streamplot":
                            st.plt.streamplot(UV, geodata)
                    if savefig & (loop == 0):
                        figname = "%s/%s_frame_%02d.png" % (path_outputs, startdate_str, i)
                        plt.savefig(figname)
                        print(figname, "saved.")

                if plotanimation:
                    plt.pause(.2)

            if plotanimation:
                plt.pause(.5)


        loop += 1

    plt.close()
