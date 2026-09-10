"""
File for example near field scanner file format.

Copyright 2026 LEmmA, Electromagnetics Research Group, Michigan State University.
"""

import logging

import h5py
import numpy as np
import xarray as xr
from xnft.common.enums import (
    CoordinateSystemType,
    DataPolarizationType,
    FieldType,
    PolarizationType,
    ScanType,
    SParameterType,
)
from xnft.common.registry import registry
from xnft.common.types import FieldDataset, FileInfo

logger = logging.getLogger(__name__)


# Constant for scanner distnaces
POINTS_PER_METER = 200_000


# === REGISTER LOADERS ===========================================================

# Register DataSource
EXAMPLE_SOURCE = registry.data_sources.register(
    id="xnft-example.example",
    display_name="Example Scanner",
)


@registry.loaders.fields.register(
    source=EXAMPLE_SOURCE,
    field_type=FieldType.NEAR,
    scan_type=ScanType.PLANAR,
    data_polarization=DataPolarizationType.SINGLE_POL,
    extension=".h5",
    has_sp=True,
)
def load_example_h5(file_list: list[FileInfo]) -> xr.Dataset:

    co_file_info = cx_file_info = None
    for file_info in file_list:
        if file_info.polarization == PolarizationType.CO_POL:
            co_file_info = file_info
        elif file_info.polarization == PolarizationType.CROSS_POL:
            cx_file_info = file_info
        else:
            raise TypeError(
                "LEmmA loader only accepts polarization types CO_POL and CROSS_POL, not both "
                "or none"
            )

    if co_file_info is None and cx_file_info is None:
        raise RuntimeError("At least `co_pol` or `cross_pol` must be specified.")

    # Open file
    for file_info in file_list:
        with h5py.File(file_info.path, "r") as dataset:
            sample_frequencies = dataset["f"][()][0]
            sample_locations = dataset["points"][()] / POINTS_PER_METER
            x_locs = sample_locations[:, 0]
            # NOTE: Swap y and z axes
            y_locs = sample_locations[:, 2]

            # TODO/FIXME: Hardcoded z-distance!!
            z_locs = sample_locations[:, 1] + 0.069

            # NOTE: We're only taking the first item in the set
            if file_info.s_parameters != set():
                sp = next(iter(file_info.s_parameters))
            else:
                sp = SParameterType.S21

            field_amplitude = dataset[f"{sp}real"][()] + 1.0j * dataset[f"{sp}imag"][()]

            if file_info.polarization == PolarizationType.CO_POL:
                ey_array = field_amplitude
                if cx_file_info is None:
                    ex_array = np.zeros_like(ey_array)
            if file_info.polarization == PolarizationType.CROSS_POL:
                ex_array = field_amplitude
                if co_file_info is None:
                    ey_array = np.zeros_like(ex_array)

    x_list = x_locs.tolist()
    y_list = y_locs.tolist()
    z_list = z_locs.tolist()

    frequency_list = sample_frequencies.tolist()

    ez_array = np.zeros_like(ex_array)

    x_sample_location_array = np.sort(np.array(list(set(x_list))))
    y_sample_location_array = np.sort(np.array(list(set(y_list))))
    z_sample_location_array = np.sort(np.array(list(set(z_list))))

    sample_frequency_array = np.sort(np.array(list(set(frequency_list))))

    logger.debug(
        "Sample location array shape: (x:%d, y:%d, z:%d, freq:%d)",
        x_sample_location_array.size,
        y_sample_location_array.size,
        z_sample_location_array.size,
        sample_frequency_array.size,
    )

    # Allocate new data array in multidimensional format
    fields_shape = (
        3,  # 3 for x,y,z component of the field
        x_sample_location_array.size,
        y_sample_location_array.size,
        z_sample_location_array.size,
        sample_frequency_array.size,
    )
    e_field = np.empty(fields_shape, dtype=np.complex128)

    # Find to insert field values in e_field array
    x_idx = np.searchsorted(x_sample_location_array, x_locs)
    y_idx = np.searchsorted(y_sample_location_array, y_locs)
    z_idx = np.searchsorted(z_sample_location_array, z_locs)

    for f_idx in range(sample_frequencies.size):
        e_field[0, x_idx, y_idx, z_idx, f_idx] = ex_array[:, f_idx]
        e_field[1, x_idx, y_idx, z_idx, f_idx] = ey_array[:, f_idx]
        e_field[2, x_idx, y_idx, z_idx, f_idx] = ez_array[:, f_idx]

    fields = FieldDataset.create(
        e_field=e_field,
        h_field=None,
        spatial_coords={
            "x": x_sample_location_array,
            "y": y_sample_location_array,
            "z": z_sample_location_array,
        },
        frequency=sample_frequency_array,
        coordinate_system=CoordinateSystemType.CARTESIAN,
        field_type=FieldType.NEAR,
        data_source=DataSourceType.LEMMA,
    )

    return fields
