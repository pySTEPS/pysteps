import importlib

def get_specifications(name):
    """Return a datasource specification file according to the following pattern:\n\
    
        datasource_<name>.py
        
    where <name> is the datasource identifier (e.g. "fmi", "mch", "bom", etc.)

    """
    datasource_name = ".datasource_%s" % name
    return importlib.import_module(datasource_name, "config")
