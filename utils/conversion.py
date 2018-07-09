''' Functions to convert rainrate to dBR or reflectivity (and viceversa)'''

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # To desactivate warnings for comparison operators with NaNs

def mmhr2dBR(R, R_threshold=0.1):
    """Function to convert rainrates to dBR.
    The values below the R_threshold receive all the dBR value corresponding to the threshold.
    This procedure removes the sharp "step" at the rain/no-rain transition, which should have a positive impact on power spectrum estimates.
    
    Parameters 
    ---------- 
    R : array-like 
        Array of any shape containing the rainrates.
    R_threshold : float 
        Rainfall threshold value.
    
    Returns: 
    ----------
    dBR : array-like 
        Array of any shape containing the dBR of rainrates.
    dBR_threshold : float 
        dBR value associated to the rainfall threshold.
    """
    
    # Set all values below threshold to zero
    zeros = (R < R_threshold)
    
    # Convert mmhr to dBR
    dBR = np.zeros_like(R)
    dBR[~zeros] = 10.0*np.log10(R[~zeros])
    
    # Set the zeros ot the mindBR value of the threshold
    dBR_threshold = 10.0*np.log10(R_threshold) 
    dBR[zeros] = dBR_threshold - 1e-9 # remove small offset to account for numerical precision
    
    return dBR, dBR_threshold

def dBR2mmhr(dBR, R_threshold=0.1):
    """Function to convert dBR to rainrates.
    
    Parameters 
    ---------- 
    dBR : array-like 
        Array of any shape containing the dBR values
    R_threshold : float 
        Rainfall threshold value.
    
    Returns: 
    ----------
    R : array-like 
        Array of any shape containing the rainrates.
    """
    
    # Convert dBR to mmhr
    R = 10.0**(dBR/10.0)
    
    # Set all values below or equal to threshold to zero
    zeros = (R <= R_threshold)
    R[zeros] = 0.0
    
    return R
    
def mmmhr2dBZ(R, R_threshold=0.1, A=316.0, b=1.5):
    """Function to convert rainrates to dBZ.
    The values below the R_threshold receive all the dBR value corresponding to the threshold.
    
    Parameters 
    ---------- 
    R : array-like 
        Array of any shape containing the rainrates.
    R_threshold : float 
        Rainfall threshold value.
    A: float
        Slope parameter of the Z-R relationship.
    B: float
        Power parameter of the Z-R relationship.
    
    Returns: 
    ----------
    dBZ : array-like 
        Array of any shape containing the reflectivity in dBZ.
    dBZ_threshold : float 
        dBZ value associated to the R_threshold.
    """
    
    # Set all values below threshold to zero
    zeros = (R < R_threshold)
    
    # Convert mmhr to dBZ
    dBZ = np.zeros_like(R)
    dBZ[~zeros] = 10.0*np.log10(A*R[~zeros]**b)
    
    # Set the zeros ot the mindBR value of the threshold
    dBZ_threshold = 10.0*np.log10(A*R**b)
    dBZ[zeros] = dBZ_threshold - 1e-9 # remove small offset to account for numerical precision
    
    return dBR, dBZ_threshold
    
def dBZ2mmhr(dBZ, R_threshold=0.1, A=316.0, b=1.5):
    """Function to convert dBZ to rainrates.
    
    Parameters 
    ---------- 
    dBZ : array-like 
        Array of any shape containing the dBZ values.
    R_threshold : float 
        Rainfall threshold value.
    A: float
        Slope parameter of the Z-R relationship.
    B: float
        Power parameter of the Z-R relationship.
    
    Returns: 
    ----------
    R : array-like 
        Array of any shape containing the rainrates.
    """
    
    # Convert dBZ to mmhr
    R = (10.0**(dBZ/10.0)/A)**(1.0/b)
    
    # Set all values below or equal to threshold to zero
    zeros = (R <= R_threshold)
    R[zeros] = 0.0
    
    return R