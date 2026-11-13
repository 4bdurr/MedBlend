import bpy
import numpy as np
from pathlib import Path

from .volume_utils import create_volume_from_numpy as _create_volume


def create_volume_from_array(
    data_array,
    name="Volume",
    location=(0, 0, 0),
    scale=(1, 1, 1),
    grid_name="density",
    threshold=None,
):
    """Create a volume object from a numpy array using the native volume API."""
    if data_array is None:
        raise ValueError("data_array must not be None")

    volume_obj = _create_volume(
        np.asarray(data_array),
        name=name,
        location=location,
        scale=scale,
        threshold=threshold,
        grid_name=grid_name,
    )
    volume_obj["grid_name"] = grid_name
    return volume_obj


def create_volume_from_numpy(data_array, filename="volume.vdb", transform=None):
    """Bridge for mock OpenVDB writes to the Geometry Nodes volume creator."""
    name = Path(filename).stem or "Volume"
    location = (0.0, 0.0, 0.0)
    scale = (1.0, 1.0, 1.0)

    if transform is not None and hasattr(transform, "matrix"):
        matrix = getattr(transform, "matrix")
        try:
            scale = (
                abs(float(matrix[0][0])),
                abs(float(matrix[1][1])),
                abs(float(matrix[2][2])),
            )
        except (TypeError, ValueError, IndexError):
            print("MedBlend: Unable to extract scale from transform; using defaults.")

    return _create_volume(np.asarray(data_array), name=name, location=location, scale=scale)


def add_data_fields(mesh, data_fields):
    """Add FLOAT point attributes to a mesh for custom data columns."""
    for data_field in data_fields:
        attribute_name = data_field if data_field else "empty_key_string"
        if attribute_name not in mesh.attributes:
            mesh.attributes.new(name=attribute_name, type='FLOAT', domain='POINT')


def create_object(mesh, name):
    """Link a mesh to the current collection and make it active."""
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


