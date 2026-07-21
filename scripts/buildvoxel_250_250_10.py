'''
Generates 6 files:
    voxel.dat = voxel metadata, centroid, where its well ids are stored
    voxel_wells.dat = flat list of well ids
    voxel_lookup.dat = maps (vx, vy, vz) to voxel id
    voxel_meta.npy = all metadata (e.g. voxel size, maximum depth)
    voxel_neighbours.dat = flat list of neighbouring voxel IDs
    voxel_neighbours_offset.dat = used for lookup
'''

import pandas as pd
import numpy as np
import os
from collections import defaultdict

CSV_PATH = "../data/wells.csv"
OUT_FOLDER = "../data/voxels/"
os.makedirs(OUT_FOLDER, exist_ok = True)
voxelSize = np.array([250,250,10],dtype=np.float32)

df = pd.read_csv(CSV_PATH)
print(f"{len(df)} total entries")
df = df.dropna().reset_index(drop=True)
print(f"{len(df)} usable datapoints")

x = df["X"].values.astype(np.float32)
y = df["Y"].values.astype(np.float32)
z = -df["Depth"].values.astype(np.float32)

xmin = (np.floor(x.min() / voxelSize[0]) * voxelSize[0])
ymin = (np.floor(y.min() / voxelSize[1]) * voxelSize[1])
max_depth = (np.ceil(df["Depth"].max() / voxelSize[2]) * voxelSize[2])

vx = np.floor((x - xmin) / voxelSize[0]).astype(np.int32)
vy = np.floor((y - ymin) / voxelSize[1]).astype(np.int32)
vz = np.floor((-z) / voxelSize[2]).astype(np.int32)

voxel_map = defaultdict(list)

for well_id, (ix,iy,iz) in enumerate(zip(vx,vy,vz)):
    voxel_map[(ix,iy,iz)].append(well_id)

print(f"Occupied voxels: {len(voxel_map)}")

voxel_dtype = np.dtype(
    [
        (
            "voxel_id",
            np.uint32
        ),

        (
            "centroid_x",
            np.float32
        ),

        (
            "centroid_y",
            np.float32
        ),

        (
            "centroid_z",
            np.float32
        ),

        (
            "well_start",
            np.uint64
        ),

        (
            "well_count",
            np.uint32
        )
    ]
)

lookup_dtype = np.dtype(
    [
        (
            "vx",
            np.int32
        ),

        (
            "vy",
            np.int32
        ),

        (
            "vz",
            np.int32
        ),

        (
            "voxel_id",
            np.uint32
        )
    ]
)

voxel_array = np.zeros(
    len(voxel_map),
    dtype=voxel_dtype
)

lookup_array = np.zeros(
    len(voxel_map),
    dtype=lookup_dtype
)

all_well_ids = []

for voxel_id, (key,wells) in enumerate(voxel_map.items()):

    ix, iy, iz = key
    centroid_x = (xmin + ix * voxelSize[0] + voxelSize[0]/2)
    centroid_y = (ymin + iy * voxelSize[1] + voxelSize[1]/2)
    centroid_z = -(iz * voxelSize[2] + voxelSize[2]/2)

    well_start = len(all_well_ids)
    all_well_ids.extend(wells)

    voxel_array[voxel_id] = (
        voxel_id,
        centroid_x,
        centroid_y,
        centroid_z,
        well_start,
        len(wells)
    )

    lookup_array[voxel_id] = (
        ix,
        iy,
        iz,
        voxel_id
    )

lookup_dict = {}

for row in lookup_array:

    lookup_dict[
        (
            row["vx"],
            row["vy"],
            row["vz"]
        )
    ] = int(row["voxel_id"])



neighbour_offset_dtype = np.dtype(
    [
        (
            "neighbour_start",
            np.uint64
        ),

        (
            "neighbour_count",
            np.uint32
        )
    ]
)

voxel_neighbour_offsets = np.zeros(len(voxel_array), dtype=neighbour_offset_dtype)
all_neighbour_ids = []

for voxel_id, row in enumerate(lookup_array):

    vx = row["vx"]
    vy = row["vy"]
    vz = row["vz"]

    neighbours = []

    for dx in range(-4,4):
        for dy in range(-4,4):
            for dz in range(-2,2):
                if (
                    dx == 0
                    and
                    dy == 0
                    and
                    dz == 0
                ):
                    continue

                neighbour_id = lookup_dict.get((vx + dx, vy + dy, vz + dz))

                if neighbour_id is not None:
                    neighbours.append(neighbour_id)

    start = len(all_neighbour_ids)
    all_neighbour_ids.extend(neighbours)
    voxel_neighbour_offsets[voxel_id] = (start,len(neighbours))

np.array(all_neighbour_ids,dtype=np.uint32).tofile(OUT_FOLDER + "voxel_neighbours.dat")
voxel_neighbour_offsets.tofile(OUT_FOLDER + "voxel_neighbour_offsets.dat")
voxel_array.tofile(OUT_FOLDER + "voxels.dat")
np.array(all_well_ids,dtype=np.uint32).tofile(OUT_FOLDER + "voxel_wells.dat")
lookup_array.tofile(OUT_FOLDER + "voxel_lookup.dat")
np.save(OUT_FOLDER + "voxel_meta.npy",{"xmin": xmin, "ymin": ymin, "voxelSize": voxelSize, "max_depth": max_depth})

print(f"\nSaved:\nvoxels.dat: {len(voxel_array)}")
print(f"voxel_wells.dat: {len(all_well_ids)}")
print(f"voxel_lookup.dat: {len(lookup_array)}")
print(f"voxel_neighbours.dat: {len(all_neighbour_ids)}")
print(f"voxel_neighbour_offsets.dat: {len(voxel_neighbour_offsets)}")