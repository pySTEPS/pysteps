.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_cascade_decomposition.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_cascade_decomposition.py:


Cascade decomposition
=====================

This example script shows how to compute and plot the cascade decompositon of 
a single radar precipitation field in pysteps.


.. code-block:: default


    from matplotlib import cm, pyplot
    import numpy as np
    from pprint import pprint
    from pysteps.cascade.bandpass_filters import filter_gaussian
    from pysteps import io
    from pysteps.cascade.decomposition import decomposition_fft
    from pysteps.utils import transformation







Read precipitation field
------------------------

As a first thing, the radar composite is imported and transformed in units
of dB.


.. code-block:: default


    # Import the example radar composite
    fn = "sample_mch_radar_composite_00.gif"
    R, _, metadata = io.import_mch_gif(fn)

    # Log-transform the data
    R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

    # nicely print the metadata
    pprint(metadata)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'accutime': 5.0,
     'institution': 'MeteoSwiss',
     'product': 'AQC',
     'projection': '+proj=somerc  +lon_0=7.43958333333333 +lat_0=46.9524055555556 '
                   '+k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel '
                   '+towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs',
     'threshold': -10.0,
     'transform': 'dB',
     'unit': 'mm',
     'x1': 255000.0,
     'x2': 965000.0,
     'xpixelsize': 1000.0,
     'y1': -160000.0,
     'y2': 480000.0,
     'yorigin': 'upper',
     'ypixelsize': 1000.0,
     'zerovalue': -15.0}


2D Fourier spectrum
--------------------

Compute and plot the 2D Fourier power spectrum of the precipitaton field.


.. code-block:: default


    # Set Nans as the fill value
    R[~np.isfinite(R)] = metadata["zerovalue"]

    # Compute the Fourier transform of the input field
    F = abs(np.fft.fftshift(np.fft.fft2(R)))

    # Plot the power spectrum
    M, N = F.shape
    fig, ax = pyplot.subplots()
    im = ax.imshow(
        np.log(F ** 2), vmin=4, vmax=24, cmap=cm.jet, extent=(-N / 2, N / 2, -M / 2, M / 2)
    )
    cb = fig.colorbar(im)
    ax.set_xlabel("Wavenumber $k_x$")
    ax.set_ylabel("Wavenumber $k_y$")
    ax.set_title("Log-power spectrum of R")




.. image:: /auto_examples/images/sphx_glr_plot_cascade_decomposition_001.png
    :class: sphx-glr-single-img




Cascade decomposition
---------------------

First, construct a set of Gaussian bandpass filters and plot the corresponding
1D filters.


.. code-block:: default


    num_cascade_levels = 7

    # Construct the Gaussian bandpass filters
    filter = filter_gaussian(R.shape, num_cascade_levels)

    # Plot the bandpass filter weights
    L = max(N, M)
    fig, ax = pyplot.subplots()
    for k in range(num_cascade_levels):
        ax.semilogx(
            np.linspace(0, L / 2, len(filter["weights_1d"][k, :])),
            filter["weights_1d"][k, :],
            "k-",
            basex=pow(0.5 * L / 3, 1.0 / (num_cascade_levels - 2)),
        )
    ax.set_xlim(1, L / 2)
    ax.set_ylim(0, 1)
    xt = np.hstack([[1.0], filter["central_wavenumbers"][1:]])
    ax.set_xticks(xt)
    ax.set_xticklabels(["%.2f" % cf for cf in filter["central_wavenumbers"]])
    ax.set_xlabel("Radial wavenumber $|\mathbf{k}|$")
    ax.set_ylabel("Normalized weight")
    ax.set_title("Bandpass filter weights")




.. image:: /auto_examples/images/sphx_glr_plot_cascade_decomposition_002.png
    :class: sphx-glr-single-img




Finally, apply the 2D Gaussian filters to decompose the radar rainfall field
into a set of cascade levels of decreasing spatial scale and plot them.


.. code-block:: default


    decomp = decomposition_fft(R, filter)

    # Plot the normalized cascade levels
    for i in range(num_cascade_levels):
        mu = decomp["means"][i]
        sigma = decomp["stds"][i]
        decomp["cascade_levels"][i] = (decomp["cascade_levels"][i] - mu) / sigma

    fig, ax = pyplot.subplots(nrows=2, ncols=4)

    ax[0, 0].imshow(R, cmap=cm.RdBu_r, vmin=-10, vmax=10)
    ax[0, 1].imshow(decomp["cascade_levels"][0], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[0, 2].imshow(decomp["cascade_levels"][1], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[0, 3].imshow(decomp["cascade_levels"][2], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 0].imshow(decomp["cascade_levels"][3], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 1].imshow(decomp["cascade_levels"][4], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 2].imshow(decomp["cascade_levels"][5], cmap=cm.RdBu_r, vmin=-3, vmax=3)
    ax[1, 3].imshow(decomp["cascade_levels"][6], cmap=cm.RdBu_r, vmin=-3, vmax=3)

    ax[0, 0].set_title("Observed")
    ax[0, 1].set_title("Level 1")
    ax[0, 2].set_title("Level 2")
    ax[0, 3].set_title("Level 3")
    ax[1, 0].set_title("Level 4")
    ax[1, 1].set_title("Level 5")
    ax[1, 2].set_title("Level 6")
    ax[1, 3].set_title("Level 7")

    for i in range(2):
        for j in range(4):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    pyplot.tight_layout()



.. image:: /auto_examples/images/sphx_glr_plot_cascade_decomposition_003.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.401 seconds)


.. _sphx_glr_download_auto_examples_plot_cascade_decomposition.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_cascade_decomposition.py <plot_cascade_decomposition.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_cascade_decomposition.ipynb <plot_cascade_decomposition.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
