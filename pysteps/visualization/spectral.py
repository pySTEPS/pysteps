"""Methods for plotting Fourier spectra."""

import matplotlib.pylab as plt
import numpy as np

def plot_rapsd(fft_freq, fft_power, x_units=None, y_units=None, wavelength_ticks=None, color='k', lw=1.0, label=None, ax=None, **kwargs):
    """
    Function to plot in log-log a radially averaged Fourier spectrum.
    
    Parameters
    ---------- 
    fft_freq: array-like
        1d array containing the Fourier frequencies computed by the function 'rapsd' in utils/spectral.py
    fft_power: array-like
        1d array containing the radially averaged Fourier power spectrum computed by the function 'rapsd' in utils/spectral.py
    x_units: str
        Units of the X variable (distance, e.g. km)
    y_units: str
        Units of the Y variable (amplitude, e.g. dBR)
    wavelength_ticks_ array-like
        List of wavelengths where to show xticklabels
    color: str
        Line color
    lw: float
        Line width
    label: str
        Label (for legend)
    ax: Axes
        Plot axes
        
    Returns
    -------
    ax: Axes
        Plot axes
    """
    
    # Check input dimensions
    n_freq = len(fft_freq)
    n_pow = len(fft_power)
    if n_freq != n_pow:
        raise ValueError("Dimensions of the 1d input arrays must be equal. %s vs %s" % (n_freq, n_pow))
    
    if ax is None:
        ax = plt.subplot(111)
    
    # Plot spectrum in log-log scale
    ax.plot(10.0*np.log10(fft_freq), 10.0*np.log10(fft_power), color=color, linewidth=lw, label=label, **kwargs)
    
    # X-axis
    if wavelength_ticks is not None:
        wavelength_ticks = np.array(wavelength_ticks)
        freq_ticks = 1.0/wavelength_ticks
        ax.set_xticks(10.0*np.log10(freq_ticks))
        ax.set_xticklabels(wavelength_ticks)
        if x_units is not None:
            ax.set_xlabel('Wavelength [' + x_units + ']')
    else:
        if x_units is not None:
            ax.set_xlabel('Frequency [1/' + x_units + ']')
    
    # Y-axis
    if y_units is not None:
        ax.set_ylabel(r'Power [10log$_{10}(\frac{' + y_units + '^2}{' + x_units + '})$]')
      
    return(ax)
    