def create_ct_volume_blender(data, name="CT_Volume", location=(0,0,0), scale=(1,1,1)):
    """
    Creates a native Blender Volume object from numpy array.
    This creates a true volumetric object that can be rendered as volume.
    """
    import bpy
    import numpy as np
    import tempfile
    import os
    print(f"Creating Blender Volume '{name}' from {data.shape} array (OpenVDB)")
    # Pastikan data dalam format float32 dan normalisasi
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    # Normalisasi ke range 0-1 untuk volume density
    data_min, data_max = np.min(data), np.max(data)
    if data_max > 1.0 or data_min < 0.0:
        data = (data - data_min) / (data_max - data_min)

    # Transpose ke (x, y, z) untuk OpenVDB
    data_vdb = np.transpose(data, (2, 1, 0)).copy(order='C')

    # Simpan ke file .vdb sementara
    vdb_path = ""
    vdb = None
    # Cari library OpenVDB yang valid
    for mod_name in ['openvdb', 'pyopenvdb']:
        try:
            temp_vdb = __import__(mod_name)
            if hasattr(temp_vdb, 'FloatGrid') and "Mock" not in str(temp_vdb):
                vdb = temp_vdb
                break
        except ImportError:
            continue

    if vdb:
        try:
            grid = vdb.FloatGrid()
            grid.copyFromArray(data_vdb)
            grid.name = "density"
            try:
                grid.transform = vdb.createLinearTransform(1.0)
            except:
                pass
            with tempfile.NamedTemporaryFile(suffix=".vdb", delete=False) as tmp:
                vdb_path = tmp.name
                vdb.write(vdb_path, [grid])
        except Exception as e:
            print(f"MedBlend: OpenVDB native write failed: {e}")
            vdb_path = ""

    if not vdb_path or not os.path.exists(vdb_path) or os.path.getsize(vdb_path) == 0:
        print("MedBlend: OpenVDB real library unavailable or file write failed; using Geometry Nodes proxy.")
        return create_volume_proxy_geo_nodes(data, name, location, scale)

    # Buat objek volume Blender dan load file .vdb
    volume_data = bpy.data.volumes.new(name=name)
    volume_data.filepath = vdb_path
    volume_data.is_sequence = False
    volume_obj = bpy.data.objects.new(name, volume_data)
    bpy.context.collection.objects.link(volume_obj)
    volume_obj.location = location
    volume_obj.scale = scale
    print(f"Volume created from VDB: {vdb_path}")
    return volume_obj
"""Helpers for constructing Blender volume grids from numpy arrays."""

import bpy
import numpy as np

from mathutils import Matrix

_POINT_FALLBACK_LIMIT = 1_000_000


def _as_float32(array):
    arr = np.asarray(array)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    else:
        arr = np.ascontiguousarray(arr)
    return arr


def _set_grid_resolution(grid, shape):
    resolution_xyz = (int(shape[2]), int(shape[1]), int(shape[0]))
    if hasattr(grid, "resolution"):
        grid.resolution = resolution_xyz
    elif hasattr(grid, "dimensions"):
        grid.dimensions = shape


def _set_grid_data(grid, data_flat):
    for attr in ("data", "values", "density", "packed_data"):
        if hasattr(grid, attr):
            try:
                setattr(grid, attr, data_flat)
                return True
            except Exception:
                pass

    points = getattr(grid, "points", None)
    if points is not None and hasattr(points, "foreach_set"):
        try:
            points.foreach_set("value", data_flat)
            return True
        except Exception:
            pass

    pack = getattr(grid, "pack", None)
    if callable(pack):
        try:
            pack(data_flat)
            return True
        except Exception:
            pass

    return False


def _create_point_cloud(data, name, location, scale):
    # Step A: Buat mesh padat dari voxel grid
    indices = np.argwhere(data > 0.5)  # Asumsi data sudah dinormalisasi 0-1
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Geometry Nodes: Mesh to Volume -> Volume to Mesh
    mod = obj.modifiers.new(name="MedBlend_Solidifier", type='NODES')
    # User can adjust node group after import for best results
    return obj


def create_volume_from_numpy(
    data,
    name="Volume",
    location=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    threshold=None,
    grid_name="density",
):
    """Create a Blender volume object from a 3D numpy array."""

    if data is None:
        raise ValueError("data must not be None")

    volume_data = np.asarray(data)
    if volume_data.ndim != 3:
        raise ValueError("data must be a 3D array")

    if threshold is not None:
        volume_data = np.where(volume_data > threshold, 1.0, 0.0)

    volume_data = _as_float32(volume_data)

    try:
        volume = bpy.data.volumes.new(name=name)
        obj = bpy.data.objects.new(name, volume)
        obj.location = location
        bpy.context.collection.objects.link(obj)

        grid = volume.grids.new(name=grid_name, type='FOG_VOLUME')

        transform = Matrix(
            (
                (scale[0], 0.0, 0.0, 0.0),
                (0.0, scale[1], 0.0, 0.0),
                (0.0, 0.0, scale[2], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        if hasattr(grid, "transform"):
            try:
                grid.transform = transform
            except AttributeError:
                if hasattr(grid.transform, "matrix"):
                    grid.transform.matrix = transform

        _set_grid_resolution(grid, volume_data.shape)

        flat_data = volume_data.ravel(order='C')
        if not _set_grid_data(grid, flat_data):
            raise RuntimeError("Unable to populate volume grid data")

        return obj

    except Exception as error:
        print(f"MedBlend: Native volume creation failed ({error}); falling back to point cloud.")

        if 'obj' in locals():
            try:
                bpy.data.objects.remove(obj)
            except Exception:
                pass
        if 'volume' in locals():
            try:
                bpy.data.volumes.remove(volume)
            except Exception:
                pass

        return _create_point_cloud(volume_data, name, location, scale)


def create_ct_volume_visualization(
    data,
    name="CT_Volume",
    location=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    threshold=None,
    grid_name="density",
    intensity_data=None,
):
    """Convert CT intensities to a volume; optional threshold operates on HU values."""

    data = np.asarray(data)

    if threshold is not None:
        # Step B: Perketat threshold dan hilangkan noise udara
        mask = np.where(data > threshold, 1.0, 0.0).astype(np.float32)
        try:
            from scipy import ndimage
            mask = ndimage.binary_opening(mask).astype(np.float32)
        except ImportError:
            pass
        if intensity_data is not None:
            intensities = np.asarray(intensity_data)
            if intensities.shape != mask.shape:
                raise ValueError("intensity_data must match data shape")
            scaled_data = intensities.astype(np.float32, copy=False) * mask
        else:
            scaled_data = mask
    else:
        if intensity_data is not None:
            scaled_data = np.asarray(intensity_data).astype(np.float32, copy=False)
        else:
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            if data_max > data_min:
                scaled_data = (data - data_min) / (data_max - data_min)
            else:
                scaled_data = np.zeros_like(data, dtype=np.float32)

            scaled_data = scaled_data.astype(np.float32, copy=False)

    return create_volume_from_numpy(
        scaled_data,
        name=name,
        location=location,
        scale=scale,
        threshold=None,
        grid_name=grid_name,
    )


def create_volume_proxy_geo_nodes(data, name="CT_Volume", location=(0,0,0), scale=(1,1,1)):
    """
    Creates a volume proxy using a Cube and Geometry Nodes (Mesh to Volume).
    This is used when pyopenvdb is unavailable.
    """
    import bpy
    # Create a cube as a container
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    
    # Scale cube appropriately
    # Data shape is (z, y, x), target scale is mm/1000
    obj.scale = (
        data.shape[2] * scale[0],
        data.shape[1] * scale[1],
        data.shape[0] * scale[2]
    )
    
    # Add Geometry Nodes modifier
    mod = obj.modifiers.new(name="MedBlend_Volume", type='NODES')
    
    # Create or get node group
    group_name = "MedBlend_MeshToVolume"
    node_group = bpy.data.node_groups.get(group_name)
    if not node_group:
        node_group = bpy.data.node_groups.new(name=group_name, type='GeometryNodeTree')
        node_group.is_modifier = True
        
        # Interface
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        
        # Nodes
        group_input = node_group.nodes.new('NodeGroupInput')
        group_output = node_group.nodes.new('NodeGroupOutput')
        mesh_to_volume = node_group.nodes.new('GeometryNodeMeshToVolume')
        
        # Links
        node_group.links.new(group_input.outputs[0], mesh_to_volume.inputs[0])
        node_group.links.new(mesh_to_volume.outputs[0], group_output.inputs[0])
        
        # Adjust density
        mesh_to_volume.inputs['Voxel Amount'].default_value = 128
    
    mod.node_group = node_group
    
    # Force display as volume in viewport
    obj.display_type = 'WIRE' 
    
    return obj
