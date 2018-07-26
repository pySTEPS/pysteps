# data source specifications
root_path  = ""
path_fmt   = "radar/bom/prcp-cscn/2/%Y/%m/%d"
fn_pattern = "2_%Y%m%d_%H%M00.prcp-cscn"
fn_ext     = "nc"
importer   = "bom_rf3"
timestep   = 6.

# importer arguments
importer_kwargs = {}