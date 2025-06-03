# mongo

## Executable scripts  

### create_mongo_user.py  

This script is run by the database administrator to register a new user for the STEPS database  

### delete_files.py  

House keeping utility to delete records from the database  

### init_steps_db.py  

This script creates the STEPS database with the expected colletions and indices.  

### load_config.py  

This script loads the JSON configuration file into the STEPS database.  

### write_nc_files.py  

Read the database and generate the netCDF files for exporting to users.  

### write_ensembles.py  

An example of a product that is supplied to an end-user.  

## modules  

### gridfs_io.py  

Functions to read and write the binary data to GridFS  

### mongo_access.py  

Functions to read and write the metadata and parameters  

### nc_utils.py  

Functions to read and write the rain fields as CF netCDF binaries.  

