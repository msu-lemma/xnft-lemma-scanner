"""
Test file loaders.

Copyright 2025-2026 LEmmA, Electromagnetics Research Group, Michigan State University.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from xnft.common.enums import (
    FieldType,
    PolarizationType,
    ScanType,
)
from xnft.common.registry import registry
from xnft.common.types import FileInfo

from xnft_lemma_scanner.plugin import *

root_dir = Path(__file__).parent.parent
test_dict = {
    "source": registry.loaders.fields.items(),  # Test all registered loaders
    "field": list(FieldType),
    "scan": list(ScanType),
}


def generate_test_id(c):
    source, field, scan, file_list = c

    return (
        f"{source.id} {field.value} {scan.value} {file_list[0].path.stem.split('_')[0]}"
    )


def generate_tests(
    fields=test_dict["field"],
    scans=test_dict["scan"],
):

    tests_list = []

    print(f"{test_dict["source"]=}")

    for source_key, source_func in test_dict["source"]:
        for field in fields:
            for scan in scans:
                test_dir_path = (
                    root_dir
                    / "data"
                    / field.lower().replace(" ", "_")
                    / scan.lower().replace(" ", "_")
                    / source_key[0].id
                )

                if not test_dir_path.exists():
                    continue

                test_file_dict = defaultdict(list)

                for path in test_dir_path.glob("*"):
                    if path.is_file() and path.name[:3].isnumeric():
                        test_id = int(path.name[:3])

                        polarization = {
                            PolarizationType.CO_POL,
                            PolarizationType.CROSS_POL,
                        }

                        suffix = path.stem.split("_")[-1]
                        if suffix == "co":
                            polarization = PolarizationType.CO_POL
                        elif suffix == "cx":
                            polarization = PolarizationType.CROSS_POL

                        test_file_dict[test_id].append(
                            FileInfo(
                                path=path,
                                polarization=polarization,
                            ),
                        )

                for file_info_list in test_file_dict.values():
                    tests_list.append(
                        (
                            source_key[0],
                            field,
                            scan,
                            file_info_list,
                        ),
                    )

    return tests_list


@pytest.fixture
def load_test_file(request: pytest.FixtureRequest):
    """Load file to test."""
    source, field, scan, file_list = request.param
    result = registry.loaders.fields.load(source, field, scan, file_list)
    return result, request.param


@pytest.mark.parametrize(
    "load_test_file",
    generate_tests(),
    ids=generate_test_id,
    indirect=True,
)
def test_file_loader(load_test_file: tuple[Any, Any]):
    result, params = load_test_file
    try:
        assert isinstance(result, xr.Dataset)
    except AssertionError as e:
        raise AssertionError(f"Error processing test: {params} / {result}") from e
