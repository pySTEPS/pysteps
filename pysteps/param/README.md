# param  

## executable scripts  

### pysteps_param.py  

This is the main script to generate an ensemble nowcast using the parametric algoithms  

### make_cascades.py  

Script to decompose and track the rainfall fields. The cascade states are written back into a GridFS bucket for later processing.  

### make_parameters.py  

Script to read the rainfall and cascade state data and calculate the STEPS parameters. The parameters are written back into a Mongo collection.  

### nwp_param_qc.py  

The NWP rainfall fields are derived from interpolating hourly ensembles onto a 10-min, 2 km resolution and contain significant errors as a result. This script cleans and smoothes the parameters that have been derived from the NWP ensemble and makes them ready for use.  

### calibrate_ar_model.py  

This scripts reads the radar rain fields and calibrates the dynamic scaling model. The output is a set of figures for quality assurance and a JSON file with the model parameters that can be included in the main configuration JSON file.  

## Modules  

### broken_line.py  

Implementation of the broken line model, not used at this stage but could be used to generate time series of STEPS parameters in the future.  

### cascade_utils.py  

A simple function to calculate the scale for each cascade level  

### shared_utils.py  

Functions that are likely to be used by the various forms of pySTEPS_param  

### steps_params.py  

Data class to manage the parameters and functions to operate on them  

### stochastic_generator.py  

Function to generate a single stochastic field given the parameters  
