""" Methods for converting physical units."""

import numpy as np
import warnings

# TODO: This should not be done. Instead fix the code so that it doesn't 
# produce the warnings.
# to deactivate warnings for comparison operators with NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)

from . import transformation

def to_rainrate(R, metadata, a=None, b=None):
    """Convert to rain rate [mm/h].
    
    Parameters 
    ---------- 
    R : array-like 
        Array of any shape to be (back-)transformed.
    metadata : dict
        The metadata dictionary contains all data-related information.
    a,b : float
        Optional, the a and b coefficients of the Z-R relationship. 
    
    Returns
    -------
    R : array-like 
        Array of any shape containing the converted units.
    metadata : dict 
        The metadata with updated attributes.
        
    """
    
    R = R.copy()
    metadata = metadata.copy()
    
    if metadata["unit"].lower() == "mm/h" and metadata["transform"] is None: 
        
        pass
            
    elif metadata["unit"].lower() == "mm" and metadata["transform"] is None: 
        
        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too
        
        R = R/float(metadata["accutime"])*60.0
        threshold = threshold/float(metadata["accutime"])*60.0
        zerovalue = zerovalue/float(metadata["accutime"])*60.0
        
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue
            
    elif metadata["unit"].lower() == "dbz" and metadata["transform"].lower() == "db": 
                  
        # dBZ to Z
        R, metadata = transformation.dB_transform(R, metadata, inverse=True)
        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too
        
        # Z to R
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = (R/a)**(1.0/b)
        threshold = (threshold/a)**(1.0/b)
        zerovalue = (zerovalue/a)**(1.0/b)
                
        metadata["zr_a"] = a
        metadata["zr_b"] = b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue
        
    else:
        raise ValueError("Cannot convert unit %s and transform %s to mm/h" % (metadata["unit"], metadata["transform"]))
        
    metadata["unit"] = "mm/h"
    
    return R, metadata

def to_raindepth(R, metadata, a=None, b=None):
    """Convert to rain depth [mm].
    
    Parameters
    ----------
    R : array-like 
        Array of any shape to be (back-)transformed.
    metadata : dict
        The metadata dictionary contains all data-related information.
    a,b : float
        Optional, the a and b coefficients of the Z-R relationship. 
        
    Returns
    -------
    R : array-like 
        Array of any shape containing the converted units.
    metadata : dict 
        The metadata with updated attributes.
        
    """
    
    R = R.copy()
    metadata = metadata.copy()
      
    if metadata["unit"].lower() == "mm" and metadata["transform"] is None: 
        pass
            
    elif metadata["unit"].lower() == "mm/h" and metadata["transform"] is None: 
    
        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too
        
        R = R/60.0*metadata["accutime"]
        threshold = threshold/60.0*metadata["accutime"]
        zerovalue = zerovalue/60.0*metadata["accutime"]
        
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue
            
    elif metadata["unit"].lower() == "dbz" and metadata["transform"].lower() == "db": 
                  
        # dBZ to Z
        R, metadata = transformation.dB_transform(R, metadata, inverse=True)
        threshold = metadata["threshold"] # convert the threshold, too
        zerovalue = metadata["zerovalue"] # convert the zerovalue, too
        
        # Z to R
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = (R/a)**(1.0/b)/60.0*metadata["accutime"]
        threshold = (threshold/a)**(1.0/b)/60.0*metadata["accutime"]
        zerovalue = (zerovalue/a)**(1.0/b)/60.0*metadata["accutime"]
                
        metadata["zr_a"] = a
        metadata["zr_b"] = b
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue
        
    else:
        raise ValueError("Cannot convert unit %s and transform %s to mm" % (metadata["unit"], metadata["transform"]))
        
    metadata["unit"] = "mm"
    
    return R, metadata
    
def to_reflectivity(R, metadata, a=None, b=None):
    """Convert to reflectivity [dBZ].
    
    Parameters 
    ---------- 
    R : array-like 
        Array of any shape to be (back-)transformed.
    metadata : dict
        The metadata dictionary contains all data-related information.
    a,b : float
        Optional, the a and b coefficients of the Z-R relationship. 
    
    Returns
    -------
    R : array-like 
        Array of any shape containing the converted units.
    metadata : dict 
        The metadata with updated attributes.
        
    """
    
    R = R.copy()
    metadata = metadata.copy()
      
    if metadata["unit"].lower() == "mm/h" and metadata["transform"] is None: 
        
        # R to Z
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
            
        R = a*R**b
        metadata["threshold"] = a*threshold**b
        metadata["zerovalue"] = a*zerovalue**b
        
        # Z to dBZ
        R, metadata = transformation.dB_transform(R, metadata)
        
            
    elif metadata["unit"].lower() == "mm" and metadata["transform"] is None: 
    
        # depth to rate
        R, metadata = to_rainrate(R, metadata)
        
        # R to Z
        if a is None:
            a = metadata.get("zr_a", 316.0)
        if b is None:
            b = metadata.get("zr_b", 1.5)
        R = a*R**b
        metadata["threshold"] = a*threshold**b
        metadata["zerovalue"] = a*zerovalue**b
        
        # Z to dBZ
        R, metadata = transformation.dB_transform(R, metadata)
            
    elif metadata["unit"].lower() == "dbz" and metadata["transform"].lower() == "db": 
                  
        pass
        
    else:
        raise ValueError("Cannot convert unit %s and transform %s to mm/h" % (metadata["unit"], metadata["transform"]))
        
    metadata["unit"] = "dBZ"
    
    return R, metadata
