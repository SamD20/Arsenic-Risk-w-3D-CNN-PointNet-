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
        isEmbedded = {"geology_specific_250m.tif" : 32}

        for file in os.listdir(raster_folder):
            if file.endswith((".tif",".tiff")):
                path = os.path.join(raster_folder,file)
                src = rasterio.open(path)
                data = src.read(1, boundless=True, masked=True).astype(np.float32)
                data = data.filled(-1)
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
                    "isEmbedded" : file in isEmbedded,
                    "EmbeddingSize" : isEmbedded[file] if file in isEmbedded else None,
                    "classes": int(np.nanmax(data)) + 1 if file in isEmbedded else None,
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

        print("\nBuilding voxel statistics cache...")

        voxel_count = len(self.voxels)

        self.voxel_stats = np.zeros(
            voxel_count,
            dtype=[
                ("well_count", np.uint32),
                ("arsenic_sum", np.float32),
                ("arsenic_sq_sum", np.float32),
                ("depth_sum", np.float32),
                ("depth_sq_sum", np.float32),
            ],
        )

        for voxel in self.voxels:
            voxel_id = voxel["voxel_id"]
            start = voxel["well_start"]
            count = voxel["well_count"]

            if count == 0:
                continue

            wells = self.voxel_wells[start:start + count]

            arsenic = self.logArsenic[wells]
            depth = self.Depth[wells]

            self.voxel_stats["well_count"][voxel_id] = count

            self.voxel_stats["arsenic_sum"][voxel_id] = arsenic.sum()
            self.voxel_stats["arsenic_sq_sum"][voxel_id] = np.square(arsenic).sum()

            self.voxel_stats["depth_sum"][voxel_id] = depth.sum()
            self.voxel_stats["depth_sq_sum"][voxel_id] = np.square(depth).sum()

        self.voxel_layout = {}
        print("\nComputing Voxel Layout...")
        self.buildVoxelLayout()

        self.raster_cache = {}
        print("\nBuilding raster cache...")
        self.buildRasterCache()

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

        ordinal = np.array([
            risk >= 1,   # above 10
            risk >= 2    # above 50
        ], dtype=np.float32)

        return {
        "voxel": self.cnnInput(idx),
        "points": self.pointNet(idx),
        "label": self.logArsenic[idx],
        "risk": risk,
        "ordinal": ordinal
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

    def calculateIDW(self, neighbour_voxels, target_index):

        voxels = np.array(list(neighbour_voxels), dtype=np.uint32)
        stats = self.voxel_stats[voxels]

        valid = stats["well_count"] > 0

        voxels = voxels[valid]
        stats = stats[valid]

        if len(voxels) == 0:
            return 0, 0, 0

        target_voxel = self.getVoxelID(target_index)

        stats = stats.copy()

        target_mask = voxels == target_voxel

        if np.any(target_mask):

            count = stats["well_count"][target_mask][0]

            if count > 1:
                stats["well_count"][target_mask] -= 1

                stats["arsenic_sum"][target_mask] -= self.logArsenic[target_index]
                stats["arsenic_sq_sum"][target_mask] -= self.logArsenic[target_index] ** 2

                stats["depth_sum"][target_mask] -= self.Depth[target_index]
                stats["depth_sq_sum"][target_mask] -= self.Depth[target_index] ** 2

            else:
                keep = ~target_mask
                voxels = voxels[keep]
                stats = stats[keep]

                if len(voxels) == 0:
                    return 0, 0, 0

        coords = self.voxels[voxels]

        dx = coords["centroid_x"] - self.X[target_index]
        dy = coords["centroid_y"] - self.Y[target_index]
        dz = coords["centroid_z"] - self.Depth[target_index]

        distance = np.sqrt(
            dx * dx +
            dy * dy +
            5 * dz * dz
        ) + 1e-6

        weights = 1 / distance

        voxel_mean = (
            stats["arsenic_sum"] /
            stats["well_count"]
        )

        mean = (
            np.sum(weights * voxel_mean) /
            weights.sum()
        )

        voxel_var = (
            stats["arsenic_sq_sum"] /
            stats["well_count"]
            -
            voxel_mean ** 2
        )

        voxel_var = np.maximum(voxel_var, 0)

        std = np.sqrt(
            np.sum(weights * voxel_var) /
            weights.sum()
        )

        wells = int(stats["well_count"].sum())

        confidence = (
            (1 - np.exp(-wells / 20.0)) *
            np.mean(np.exp(-distance / 500.0))
        )

        return (
            np.clip(mean / self.maxLogArsenic, 0, 1),
            np.clip(std / self.maxLogArsenic, 0, 1),
            np.clip(confidence, 0, 1)
        )

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

    def buildVoxelLayout(self):
        for voxel_id in range(len(self.voxels)):
            voxelCoords = self.getVoxelCoords(voxel_id)

            neighbours = set(self.getNeighbours(voxel_id))
            neighbours.add(voxel_id)

            layout = []

            for neighbour in neighbours:

                voxel = self.voxels[neighbour]

                vx = int(
                    np.floor(
                        (voxel["centroid_x"] - voxelCoords[0])
                        / self.voxel_size[0]
                    )
                ) + self.xrange // 2

                vy = int(
                    np.floor(
                        (voxel["centroid_y"] - voxelCoords[1])
                        / self.voxel_size[1]
                    )
                ) + self.yrange // 2

                vz = int(
                    np.floor(
                        -(voxel["centroid_z"] - voxelCoords[2])
                        / self.voxel_size[2]
                    )
                ) + self.zrange // 2


                if (
                    vx < 0 or vx >= self.xrange or
                    vy < 0 or vy >= self.yrange or
                    vz < 0 or vz >= self.zrange
                ):
                    continue


                norm_x = (
                    voxel["centroid_x"] - self.x_mean
                ) / (self.x_std + 1e-6)

                norm_y = (
                    voxel["centroid_y"] - self.y_mean
                ) / (self.y_std + 1e-6)


                layout.append(
                    (
                        neighbour,
                        vx,
                        vy,
                        vz,
                        norm_x,
                        norm_y
                    )
                )

            self.voxel_layout[voxel_id] = layout

    def buildRasterCache(self):

        for voxel in self.voxels:

            voxel_id = voxel["voxel_id"]

            cx = voxel["centroid_x"]
            cy = voxel["centroid_y"]

            patch = []

            for raster in self.rasters.values():

                data = raster["data"]
                transform = raster["transform"]

                px, py = ~transform * (cx, cy)

                px=int(px)
                py=int(py)

                half = self.xrange//2

                crop = data[
                    py-half:py+half+1,
                    px-half:px+half+1
                ].copy()

                if raster["isEmbedded"]:
                    crop[crop < 0] = 0
                else:
                    crop = np.nan_to_num(
                        (crop - raster["mean"]) / raster["std"],
                        nan=0
                    )

                if crop.shape != (self.xrange,self.yrange):
                    padded=np.zeros(
                        (self.xrange,self.yrange),
                        dtype=np.float32
                    )
                    padded[:crop.shape[0],:crop.shape[1]]=crop
                    crop=padded

                patch.append(crop)

            self.raster_cache[voxel_id]=np.stack(patch)

    def cnnInput(self, target_index):
        targetVoxel = self.getVoxelID(target_index)
        voxelCoords = self.getVoxelCoords(targetVoxel)

        layout = self.voxel_layout[targetVoxel]
        neighbours = [x[0] for x in layout]

        raster_channels = len(self.rasters)

        tensor = np.zeros(
            (raster_channels + 11, self.xrange, self.yrange, self.zrange),
            dtype=np.float32
        )
        
        raster_values = self.raster_cache[targetVoxel]
        tensor[:raster_channels] = raster_values[:,:,:,None]

        x_indices = np.arange(self.xrange) - self.xrange//2
        y_indices = np.arange(self.yrange) - self.yrange//2
        z_indices = np.arange(self.zrange) - self.zrange//2

        idw_mean, idw_std, idw_confidence = self.calculateIDW(neighbours, target_index)
        tensor[raster_channels + 8] = idw_mean #idw mean as
        tensor[raster_channels + 9] =  idw_std #idw std as
        tensor[raster_channels + 10] = idw_confidence #idw confidence

        for thisVoxel, vx, vy, vz, norm_x, norm_y in self.voxel_layout[targetVoxel]:
            stats = self.voxel_stats[thisVoxel]

            n = int(stats["well_count"])


            # remove target well to prevent leakage
            if thisVoxel == targetVoxel:
                n -= 1

                if n <= 0:
                    continue

                arsenic_sum = (stats["arsenic_sum"] - self.logArsenic[target_index])
                arsenic_sq_sum = (stats["arsenic_sq_sum"]  - self.logArsenic[target_index]**2)

                depth_sum = ( stats["depth_sum"] - self.Depth[target_index])
                depth_sq_sum = (stats["depth_sq_sum"]  - self.Depth[target_index]**2)
                
            else:
                arsenic_sum = stats["arsenic_sum"]
                arsenic_sq_sum = stats["arsenic_sq_sum"]

                depth_sum = stats["depth_sum"]
                depth_sq_sum = stats["depth_sq_sum"]

            ars_mean = arsenic_sum / n
            ars_std = np.sqrt(max(0, arsenic_sq_sum / n - ars_mean**2))

            dep_mean = depth_sum / n
            dep_std = np.sqrt(max(0, depth_sq_sum / n - dep_mean**2))

            tensor[raster_channels, vx, vy, vz] = np.log1p(n)
            tensor[raster_channels+1, vx, vy, vz] = np.clip(ars_mean / self.maxLogArsenic,0,1)
            tensor[raster_channels+2, vx, vy, vz] = np.clip(ars_std / self.maxLogArsenic,0,1)
            tensor[raster_channels+3, vx, vy, vz] = dep_mean / self.maxDepth
            tensor[raster_channels+4, vx, vy, vz] = dep_std / self.maxDepth
            tensor[raster_channels+5, vx, vy, vz] = 1
            tensor[raster_channels + 6, vx, vy, vz] = norm_x
            tensor[raster_channels + 7, vx, vy, vz] = norm_y

            if thisVoxel == targetVoxel:
                tensor[raster_channels + 8] = np.clip(ars_mean / self.maxLogArsenic,0,1) #idw mean as
                tensor[raster_channels + 9] =  np.clip(ars_std / self.maxLogArsenic,0,1) #idw std as
                tensor[raster_channels + 10] = 1 #idw confidence
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