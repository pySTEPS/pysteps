"""Implementations of deterministic and ensemble nowcasting methods."""

from pysteps.nowcasts.interface import *
discover_nowcasts()

try:
    from dgmr_module_plugin import dgmr
except ImportError:
    print("Error: DGMR plugin is required but not installed.")
    print("Please install the plugin to use this feature.")
    print("You can install it using the following command:")
    print("pip install git+https://github.com/LoicKemajou/dgmr_plugin.git")





