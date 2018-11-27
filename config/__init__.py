# -*- coding: utf-8 -*-

#  Mantained for compatibility purposes with old config module
#  To be deprecated in the next version!
import warnings

from pysteps import rcparams

warnings.warn("config module will be deprecated in the next release; "
              + "use rcparams instead.",
              category=FutureWarning
              )

outputs_params = rcparams['outputs']
path_outputs = outputs_params['path_outputs']

plot_params = rcparams['plot']
colorscale = plot_params['colorscale']
motion_plot = plot_params['motion_plot']

def get_specifications(data_source):
    """"
    Return a datasource specification from the cofiguration file,
    where <data_source> is the datasource identifier (e.g. "fmi", "mch", "bom", etc.)

    Maintained for compatibility purposes.
    This will be deprecated in the next version.
    """
    return rcparams["data_sources"][data_source]
