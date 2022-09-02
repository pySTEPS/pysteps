import numpy as np
import pytest

from pysteps import io
from pysteps.tests.helpers import get_precipitation_fields


class TestImportNetcdfPysteps:
    @pytest.fixture(autouse=True)
    def _setup(self):
        precip, metadata = get_precipitation_fields(
            num_prev_files=1,
            num_next_files=0,
            return_raw=False,
            metadata=True,
            upscale=2000,
        )
        self.precip = precip
        self.metadata = metadata

    def test_import_netcdf(self, tmp_path):

        precip = self.precip
        metadata = self.metadata
        field_shape = (precip.shape[1], precip.shape[2])
        startdate = metadata["timestamps"][-1]
        timestep = metadata["accutime"]
        exporter = io.exporters.initialize_forecast_exporter_netcdf(
            outpath=tmp_path.as_posix(),
            outfnprefix="test",
            startdate=startdate,
            timestep=timestep,
            n_timesteps=precip.shape[0],
            shape=field_shape,
            metadata=metadata,
        )
        io.exporters.export_forecast_dataset(precip, exporter)
        io.exporters.close_forecast_files(exporter)

        tmp_file = tmp_path / "test.nc"
        precip_netcdf, metadata_netcdf = io.import_netcdf_pysteps(
            tmp_file, dtype="float64"
        )

        assert isinstance(precip_netcdf, np.ndarray)
        assert isinstance(metadata_netcdf, dict)
        assert precip_netcdf.ndim == precip.ndim, "Wrong number of dimensions"
        assert precip_netcdf.shape[0] == precip.shape[0], "Wrong number of lead times"
        assert precip_netcdf.shape[1:] == field_shape, "Wrong field shape"

    def test_import_netcdf_norain(self, tmp_path):

        precip = self.precip
        metadata = self.metadata
        field_shape = (precip.shape[1], precip.shape[2])
        startdate = metadata["timestamps"][-1]
        timestep = metadata["accutime"]
        exporter = io.exporters.initialize_forecast_exporter_netcdf(
            outpath=tmp_path.as_posix(),
            outfnprefix="test_norain",
            startdate=startdate,
            timestep=timestep,
            n_timesteps=precip.shape[0],
            shape=field_shape,
            metadata=metadata,
        )
        io.exporters.export_forecast_dataset(precip * 0, exporter)
        io.exporters.close_forecast_files(exporter)

        tmp_file = tmp_path / "test_norain.nc"
        io.import_netcdf_pysteps(tmp_file, dtype="float64")
