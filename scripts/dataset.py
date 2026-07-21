import pandas as pd
import numpy as np
import rasterio
import os
from pyproj import Transformer
from dataloader import RISK_CLASSES

transformer = Transformer.from_crs("EPSG:32646","EPSG:4326",always_xy=True)
MAIN_FOLDER = "../data"
RASTER_FOLDER = "./rasters"
VOXELS_FOLDER = "./voxels"
CSV_FILE = "wells.csv"
TOTAL_PATCH_SIZE = [2250, 2250, 50]

class ArsenicDataset:
    def __init__(self):
        self.df = pd.read_csv(os.path.join(MAIN_FOLDER, CSV_FILE))
        self.df = (self.df.dropna().reset_index(drop=True))

        self.X = self.df["X"].values.astype(np.float32)
        self.Y = self.df["Y"].values.astype(np.float32)
        self.Depth = self.df["Depth"].values.astype(np.float32)
        self.Arsenic = self.df["Arsenic"].values.astype(np.float32)
        self.logArsenic = np.log1p(self.Arsenic)

        self.x_mean = self.X.mean()
        self.x_std = self.X.std()
        self.y_mean = self.Y.mean()
        self.y_std = self.Y.std()
        self.depth_std = self.Depth.std()

        self.maxDepth = self.Depth.max()
        self.maxLogArsenic = np.percentile(self.logArsenic, 99)

        self.maxDistance = np.sqrt(TOTAL_PATCH_SIZE[0]**2 + TOTAL_PATCH_SIZE[1]**2 + TOTAL_PATCH_SIZE[2]**2)
        lats = self.df["lat"].values
        lons = self.df["lon"].values
        self.lon_mean = lons.mean()
        self.lon_std = lons.std()
        self.lat_mean = lats.mean()
        self.lat_std = lats.std()

        self.rasters = {}
        raster_folder = os.path.join(MAIN_FOLDER, RASTER_FOLDER)

        for file in os.listdir(raster_folder):
            if file.endswith((".tif",".tiff")):
                path = os.path.join(raster_folder,file)
                src = rasterio.open(path)
                data = src.read(1, boundless=True, fill_value=np.nan)
                valid = data[~np.isnan(data)]
                self.rasters[file] = {
                    "data":
                        data,
                    "transform":
                        src.transform,
                    "mean":
                        valid.mean(),
                    "std":
                        valid.std(),
                }

        lookup_dtype = np.dtype([("vx", np.int32),("vy", np.int32),("vz", np.int32),("voxel_id", np.uint32)])
        lookup = np.memmap(os.path.join(MAIN_FOLDER, VOXELS_FOLDER, "voxel_lookup.dat"),dtype=lookup_dtype,mode="r")
        self.lookup = {}

        for row in lookup:
            self.lookup[(row["vx"],row["vy"],row["vz"])] = row["voxel_id"]

        meta = np.load(os.path.join(MAIN_FOLDER, VOXELS_FOLDER, "voxel_meta.npy"),allow_pickle=True).item()

        neighbour_offset_dtype = np.dtype([("neighbour_start", np.uint64),("neighbour_count", np.uint32)])

        self.voxel_neighbours = np.memmap(os.path.join(MAIN_FOLDER, VOXELS_FOLDER,"voxel_neighbours.dat"),dtype=np.uint32,mode="r")
        self.voxel_neighbour_offsets = np.memmap(os.path.join(MAIN_FOLDER, VOXELS_FOLDER,"voxel_neighbour_offsets.dat"),dtype=neighbour_offset_dtype,mode="r")

        voxel_dtype = np.dtype([("voxel_id", np.uint32),("centroid_x", np.float32),("centroid_y", np.float32),("centroid_z", np.float32),("well_start", np.uint64),("well_count", np.uint32)])
        self.voxels = np.memmap(os.path.join(MAIN_FOLDER, VOXELS_FOLDER,"voxels.dat"),dtype=voxel_dtype,mode="r")

        self.voxel_wells = np.memmap(os.path.join(MAIN_FOLDER,VOXELS_FOLDER,"voxel_wells.dat"),dtype=np.uint32,mode="r")

        self.voxel_size = meta["voxelSize"]
        self.xmin = meta["xmin"]
        self.ymin = meta["ymin"]

        self.xrange = int(TOTAL_PATCH_SIZE[0] / self.voxel_size[0])
        self.yrange = int(TOTAL_PATCH_SIZE[1] / self.voxel_size[1])
        self.zrange = int(TOTAL_PATCH_SIZE[2] / self.voxel_size[2])

        print(f"\nDataset Information:\n Wells: {len(self.X)}")
        print(f" Rasters: {len(self.rasters)}")
        print(f" Target Area: {int(TOTAL_PATCH_SIZE[0])}m by {int(TOTAL_PATCH_SIZE[1])}m by {int(TOTAL_PATCH_SIZE[2])}m")
        print(f" Voxel Size: {int(self.voxel_size[0])}m by {int(self.voxel_size[1])}m by {int(self.voxel_size[2])}m\n")

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        arsenic = self.Arsenic[idx]

        if arsenic <= RISK_CLASSES[0]:
            risk = 0
        elif arsenic <= RISK_CLASSES[1]:
            risk = 1
        else:
            risk = 2

        return {
        "voxel": self.cnnInput(idx),
        "points": self.pointNet(idx),
        "label": self.logArsenic[idx],
        "risk": risk
        }
    
    def getVoxelID(self, well_index):
        x = self.X[well_index]
        y = self.Y[well_index]
        depth = self.Depth[well_index]

        vx = int(np.floor((x - self.xmin) / self.voxel_size[0]))
        vy = int(np.floor((y - self.ymin) / self.voxel_size[1]))
        vz = int(np.floor(depth / self.voxel_size[2]))

        voxel_id = self.lookup.get((vx,vy,vz))

        return voxel_id

    def getVoxelCoords(self, voxel_id):
        voxel = self.voxels[voxel_id]
        return [voxel["centroid_x"],voxel["centroid_y"],voxel["centroid_z"]]

    def getNeighbours(self, voxel_index):
        offset = self.voxel_neighbour_offsets[voxel_index]

        start = offset["neighbour_start"]
        count = offset["neighbour_count"]

        neighbours = self.voxel_neighbours[start:start + count]

        return neighbours

    def getRasterValue(self, x, y):
        values = []

        for raster in self.rasters.values():

            data = raster["data"]
            transform = raster["transform"]

            col, row = ~transform * (x, y)

            col = int(col)
            row = int(row)

            if row < 0 or col < 0 or row >= data.shape[0] or col >= data.shape[1]:
                values.append(np.nan)
            else:
                values.append(data[row, col])

        return values

    def cnnInput(self, target_index):
        targetVoxel = self.getVoxelID(target_index)
        voxelCoords = self.getVoxelCoords(targetVoxel)
        neighbours = set(self.getNeighbours(targetVoxel))
        neighbours.add(targetVoxel)
        tensor = np.zeros((len(self.rasters)+8,self.xrange,self.yrange,self.zrange),dtype=np.float32)

        for x in range(0, self.xrange):
            for y in range(0, self.yrange):
                xchange = x - (self.xrange // 2)
                ychange = y - (self.yrange // 2)
                coordx = voxelCoords[0] + (xchange * self.voxel_size[0])
                coordy = voxelCoords[1] + (ychange * self.voxel_size[1])
                current_vx = int(np.floor((coordx - self.xmin) / self.voxel_size[0]))
                current_vy = int(np.floor((coordy - self.ymin) / self.voxel_size[1]))

                patches = self.getRasterValue(coordx, coordy)
                raster_channels = len(self.rasters)

                for z in range(0, self.zrange):
                    for r, file in enumerate(self.rasters.keys()):
                        value = patches[r]
                        tensor[r,x,y,z] = np.nan_to_num((value - self.rasters[file]["mean"]) / self.rasters[file]["std"],nan=0)

                    zchange = z - (self.zrange // 2)
                    coordz = voxelCoords[2] + (zchange * self.voxel_size[2])
                    current_vz = int(np.floor((-coordz) / self.voxel_size[2]))
                    thisVoxel = self.lookup.get((current_vx,current_vy,current_vz))

                    if thisVoxel in neighbours:
                        voxel = self.voxels[thisVoxel]
                        start = voxel["well_start"]
                        count = voxel["well_count"]
                        well_ids = self.voxel_wells[start:start+count]

                        if thisVoxel == targetVoxel:
                            well_ids = well_ids[well_ids != target_index]

                        if len(well_ids) > 0:
                            count_wells = len(well_ids)
                            arsenic = self.Arsenic[well_ids]
                            depth = self.Depth[well_ids]

                            tensor[raster_channels,x,y,z] = np.log1p(count_wells)
                            tensor[raster_channels + 1,x,y,z] = np.clip(np.log1p(arsenic).mean() / self.maxLogArsenic, 0, 1)
                            tensor[raster_channels + 2,x,y,z] = np.clip(np.log1p(arsenic).std() / self.maxLogArsenic, 0, 1)
                            tensor[raster_channels + 3,x,y,z] = depth.mean() / self.maxDepth
                            tensor[raster_channels + 4,x,y,z] = depth.std() / self.maxDepth
                            tensor[raster_channels + 5,x,y,z] = 1 #has data?
                        else:
                            tensor[raster_channels + 0,x,y,z] = -1
                            tensor[raster_channels + 1,x,y,z] = -1
                            tensor[raster_channels + 2,x,y,z] = -1
                            tensor[raster_channels + 3,x,y,z] = -1
                            tensor[raster_channels + 4,x,y,z] = -1
                            tensor[raster_channels + 5,x,y,z] = 0 #has data?

                        lon, lat = transformer.transform(coordx, coordy)
                        tensor[raster_channels + 6,x,y,z] = (lon - self.lon_mean) / self.lon_std
                        tensor[raster_channels + 7,x,y,z] = (lat - self.lat_mean) / self.lat_std

        return tensor

    def pointNet(self, target_index):
        targetVoxel = self.getVoxelID(target_index)
        well_ids = set()

        for voxel in self.getNeighbours(targetVoxel):
            start,count = self.voxels[voxel]["well_start"],self.voxels[voxel]["well_count"]
            well_ids.update(self.voxel_wells[start:start+count])

        well_ids.discard(target_index)

        if not well_ids:
            return np.empty((0,15),dtype=np.float32)

        well_ids = np.fromiter(well_ids,dtype=np.uint32)

        tx,ty,tz = self.X[target_index],self.Y[target_index],self.Depth[target_index]

        dx = self.X[well_ids]-tx
        dy = self.Y[well_ids]-ty
        dz = self.Depth[well_ids]-tz
        distance = np.sqrt(dx*dx+dy*dy+dz*dz)+1e-6

        depth = self.Depth[well_ids]
        arsenic = self.logArsenic[well_ids]

        weights = 1/distance
        local_mean = np.sum(arsenic*weights)/np.sum(weights)
        local_std = np.sqrt(np.mean((arsenic-local_mean)**2))

        depth_diff = depth-depth.mean()

        target_stratum = np.digitize(tz,[15.3,45,65,90,150])
        strata = np.digitize(depth,[15.3,45,65,90,150])
        same_stratum = arsenic[strata==target_stratum].mean() if np.any(strata==target_stratum) else local_mean

        cloud = np.stack([
            dx/TOTAL_PATCH_SIZE[0],
            dy/TOTAL_PATCH_SIZE[1],
            dz/TOTAL_PATCH_SIZE[2],

            # target global coords (repeat for every neighbour)
            np.full(len(well_ids), (tx - self.x_mean) / (self.x_std + 1e-6)),
            np.full(len(well_ids), (ty - self.y_mean) / (self.y_std + 1e-6)),
            np.full(len(well_ids), tz / self.maxDepth),

            # neighbour global coords
            (self.X[well_ids] - self.x_mean) / (self.x_std + 1e-6),
            (self.Y[well_ids] - self.y_mean) / (self.y_std + 1e-6),
            self.Depth[well_ids] / self.maxDepth,

            distance/self.maxDistance,
            depth_diff / (self.depth_std + 1e-6),

            np.clip(arsenic/self.maxLogArsenic,0,1),

            np.full(len(well_ids),np.clip(local_mean/self.maxLogArsenic,0,1)),
            np.full(len(well_ids),np.clip(local_std/self.maxLogArsenic,0,1)),
            np.full(len(well_ids),np.clip(same_stratum/self.maxLogArsenic,0,1))
        ],axis=1).astype(np.float32)

        return cloud