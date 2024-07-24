# from pysteps.pysteps.nowcasts import interface

# def test_nowcast_registration():
#     # Replace 'your_method_name' with the name you registered
#     method_name = 'probability'
    
#     # Attempt to retrieve the method from the registry
#     try:
#         info=interface.importers_info()
#         method = interface.get_method(method_name)
#         print(f"Successfully retrieved method: {method_name}")
#     except KeyError:
#         print(f"Method {method_name} not found in the registry.")

    # Optionally, test the method with dummy input
    # Replace 'dummy_input' with appropriate test data
    # result = method(dummy_input)
    # print("Method output:", result)

# if __name__ == "__main__":
#     test_nowcast_registration()

# from pysteps.pysteps.io import interface

# interface.importers_info()

# import pkg_resources
# import importlib
# importlib.reload(pkg_resources)



import tensorflow as tf
import numpy as np
import os
from wradlib.io import read_opera_hdf5
import xarray as xr
import pandas as pd


## Uncomment the next lines if pyproj is needed for the importer.
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False



"""
Generates and preprocessing input that will be used for prediction on the model


When run, this file creates two sets of inputs. One set of .nf files 
which can be used in plotting.ipynb for visualization with cartopy, 
and a set of .npy files which can be used in 
pred_on_colab.ipynb to generate the predictions.
"""


def _read_hdf_to_numpy(filename,**kwargs):
    '''Code by Simon De Kock <simon.de.kock@vub.be>
    Generate the input to be trained by the model based in the file folder
   
    
    Parameters
    ----------
    filename : String
        contains path to the hdf files.
    start_time_frame: string
         contains the time of the frame from which you wish to start
         e.g(202311062000500)
         that is '%Year%month%day%Hour%Minute%Second'

    
    Returns
    -------
    precip, a numpy array
    '''
    fns = []
    # A slice of the files was selected to produce nowcasts with DGMR and LDCast
    # Such that those nowcast start as close as possible to the startime of the PySTEPS and INCA nowcasts
    for file_name in sorted(os.listdir(filename)):
        if file_name.endswith('.hdf'):
            fns.append(f"{filename}/{file_name}")
        else:
            raise Exception('Incorrect file name')
        
    
    dataset = []
    for i, file_name in enumerate(fns):
        # Read the content
        file_content = read_opera_hdf5(file_name)

        # Extract time information
        time_str = os.path.splitext(os.path.basename(file_name))[0].split('.', 1)[0]
        time = pd.to_datetime(time_str, format='%Y%m%d%H%M%S')

        # Extract quantity information
        try:
            quantity = file_content['dataset1/data1/what']['quantity'].decode()
        except:
            quantity = file_content['dataset1/data1/what']['quantity']

        # Set variable properties based on quantity
        if quantity == 'RATE':
            short_name = 'precip_intensity'
            long_name = 'instantaneous precipitation rate'
            units = 'mm h-1'
        else:
            raise Exception(f"Quantity {quantity} not yet implemented.")

        # Create the grid
        projection = file_content.get("where", {}).get("projdef", "")
        if type(projection) is not str:
            projection = projection.decode("UTF-8")

        gridspec = file_content.get("dataset1/where", {})

        x = np.linspace(gridspec.get('UL_x', 0),
                        gridspec.get('UL_x', 0) + gridspec.get('xsize', 0) * gridspec.get('xscale', 0),
                        num=gridspec.get('xsize', 0), endpoint=False)
        x += gridspec.get('xscale', 0)
        y = np.linspace(gridspec.get('UL_y', 0),
                        gridspec.get('UL_y', 0) - gridspec.get('ysize', 0) * gridspec.get('yscale', 0),
                        num=gridspec.get('ysize', 0), endpoint=False)
        y -= gridspec.get('yscale', 0) / 2

        x_2d, y_2d = np.meshgrid(x, y)

        pr = pyproj.Proj(projection)
        
        lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
        lon = lon.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        lat = lat.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        
        # Build the xarray dataset
        ds = xr.Dataset(
            data_vars={
                short_name: (['x', 'y'], file_content.get("dataset1/data1/data", np.nan),
                            {'long_name': long_name, 'units': units})
            },
            coords={
                'x': (['x'], x, {'axis': 'X', 'standard_name': 'projection_x_coordinate',
                                'long_name': 'x-coordinate in Cartesian system', 'units': 'm'}),
                'y': (['y'], y, {'axis': 'Y', 'standard_name': 'projection_y_coordinate',
                                'long_name': 'y-coordinate in Cartesian system', 'units': 'm'}),
                'lon': (['y', 'x'], lon, {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                                        'units': 'degrees_east'}),
                'lat': (['y', 'x'], lat, {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                                        'units': 'degrees_north'})
            }
        )
        ds['time'] = time

        # Append the dataset to the list
        dataset.append(ds)
    
       
        
    # Concatenate datasets along the time dimension
    dataset = xr.concat(dataset, dim='time')
    final_dataset=dataset.sortby(dataset.time)
    precip=final_dataset['precip_intensity'].to_numpy()
    
    return precip




def import_dgmr_preprocessed_input(filename,**kwargs):
    '''
    Parameters
    ----------
    filename : String
        contains path to the hdf files.


    
    Returns
    -------
    A tupple whose first element is the dgmr input and the second element is the next_input_frame
     
    A tensor of shape (num_samples,T_out,H,W,C), where T_out is either 18 or 22
    as described above.

    
    - Crop xarray data to required dimensions (700x700 to 256x256)
    - Reshape it to:
        [B, T, C, H, W] - Batch, Time, Channel, Heigh, Width
    - Turn it into a tensor
    '''
    field=_read_hdf_to_numpy(filename)
    
    # Verifies if there are sufficiently correct number of frames for the model.
    # DGMR takes 4 frames at a time 
    if field.shape[0]>=4:
        
        # Crop the center of the field and get a 256x256 image
        # Intervals of +/- 256/2 around the center (which is 700/2)
        low = (700//2) - (256//2)
        high = (700//2) + (256//2)
        cropped = field[:, low:high, low:high]
        cropped=tf.reshape(cropped, [cropped.shape[0], 256, 256, 1])
        dgmr_input=cropped[:4]
        try:
            next_observation_frames=cropped[4:]
            return dgmr_input.numpy(),next_observation_frames.numpy()
        except IndexError:
            return dgmr_input.numpy()
        
    else:
        raise Exception('Incorrect number of frames for DGMR. DGMR needs four frames')
   
 
 
from pysteps.nowcasts import interface
import pkg_resources
import importlib
importlib.reload(pkg_resources)

# for entry_point in pkg_resources.iter_entry_points(
#         group="pysteps.pysteps.nowcasts", name=None
#     ):

#     _new_nowcast= entry_point.load()
#     print(entry_point.name)
#     nowcast_module_name=entry_point.module_name
#     print(nowcast_module_name)
#     print(importlib.import_module(entry_point.module_name))

from pysteps.nowcasts import interface
interface.nowcasts_info()
# data=import_dgmr_preprocessed_input(r"C:\Users\user\Desktop\meteo_france_data")
    
# print(data[0].shape)

dgmr=interface.get_method('dgmr') 
# forecast=dgmr(data[0])
# print(forecast.shape)
from pysteps.nowcasts import dgmr













