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
        isEmbedded = {"geology_specific_250m.tif" : 32, "flood_index_250m.tif" : 8}

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

        self.raster_channels = 0

        for raster in self.rasters.values():

            if raster["isEmbedded"]:
                self.raster_channels += raster["classes"]
            else:
                self.raster_channels += 1
                
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

        self.empty_tensor = np.zeros((self.raster_channels+18,self.xrange,self.yrange,self.zrange),dtype=np.float32)

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
                ("mean", np.float32),
                ("std", np.float32),
                ("median", np.float32),
                ("p10", np.float32),
                ("p25", np.float32),
                ("p75", np.float32),
                ("p90", np.float32),
                ("p95", np.float32),
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

            self.voxel_stats["median"][voxel_id] = np.quantile(arsenic, 0.50)
            self.voxel_stats["p10"][voxel_id] = np.quantile(arsenic, 0.10)
            self.voxel_stats["p25"][voxel_id] = np.quantile(arsenic, 0.25)
            self.voxel_stats["p75"][voxel_id] = np.quantile(arsenic, 0.75)
            self.voxel_stats["p90"][voxel_id] = np.quantile(arsenic, 0.90)
            self.voxel_stats["p95"][voxel_id] = np.quantile(arsenic, 0.95)

        self.voxel_layout = {}
        print("\nComputing Voxel Layout...")
        self.buildVoxelLayout()

        self.idw_cache = {}

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

    def calculateVoxelIDW(self, voxel_id):

        if voxel_id in self.idw_cache:
            return self.idw_cache[voxel_id]

        voxel = self.voxels[voxel_id]

        cx = voxel["centroid_x"]
        cy = voxel["centroid_y"]
        cz = voxel["centroid_z"]

        well_ids = []

        for n in self.getNeighbours(voxel_id):
            start = self.voxels[n]["well_start"]
            count = self.voxels[n]["well_count"]

            if count:
                well_ids.extend(
                    self.voxel_wells[start:start+count]
                )

        if len(well_ids) == 0:
            result = np.zeros(9,dtype=np.float32)
            self.idw_cache[voxel_id] = result
            return result

        well_ids = np.asarray(well_ids)

        dx = self.X[well_ids] - cx
        dy = self.Y[well_ids] - cy
        dz = self.Depth[well_ids] - cz

        distance = np.sqrt(
            dx**2 +
            dy**2 +
            5*dz**2
        ) + 1e-6

        weights = 1 / distance
        weights /= weights.sum()

        arsenic = self.logArsenic[well_ids]

        mean = np.sum(weights * arsenic)

        std = np.sqrt(
            np.sum(weights*(arsenic-mean)**2)
        )

        order = np.argsort(arsenic)

        values = arsenic[order]
        w = weights[order]

        cumulative = np.cumsum(w)

        result = np.array([
            mean/self.maxLogArsenic,
            std/self.maxLogArsenic,
            np.interp(0.50,cumulative,values),
            np.interp(0.10,cumulative,values),
            np.interp(0.25,cumulative,values),
            np.interp(0.75,cumulative,values),
            np.interp(0.90,cumulative,values),
            np.interp(0.95,cumulative,values),
            (
                (1-np.exp(-len(well_ids)/20))
                *
                np.mean(np.exp(-distance/500))
            )
        ],dtype=np.float32)

        self.idw_cache[voxel_id] = result

        return result

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

            patch_channels = []

            for raster in self.rasters.values():

                data = raster["data"]
                transform = raster["transform"]

                px, py = ~transform * (cx, cy)

                px = int(px)
                py = int(py)

                half = self.xrange // 2

                crop = data[
                    py-half:py+half+1,
                    px-half:px+half+1
                ].copy()


                # =========================
                # EMBEDDED RASTERS
                # =========================

                if raster["isEmbedded"]:

                    # remove invalid values
                    crop[crop < 0] = 0

                    classes = raster["classes"]

                    # one-hot encode
                    embedding = np.eye(
                        classes,
                        dtype=np.float32
                    )[crop.astype(np.int32)]


                    # (H,W,C) -> (C,H,W)
                    embedding = np.moveaxis(
                        embedding,
                        -1,
                        0
                    )

                    for channel in embedding:
                        patch_channels.append(channel)


                # =========================
                # NORMAL RASTERS
                # =========================

                else:

                    crop = np.nan_to_num(
                        (crop - raster["mean"]) /
                        (raster["std"] + 1e-6),
                        nan=0
                    )

                    patch_channels.append(crop)


            # =========================
            # PAD ALL CHANNELS
            # =========================

            padded_channels = []

            for channel in patch_channels:

                if channel.shape != (
                    self.xrange,
                    self.yrange
                ):

                    padded = np.zeros(
                        (
                            self.xrange,
                            self.yrange
                        ),
                        dtype=np.float32
                    )

                    h = min(
                        channel.shape[0],
                        self.xrange
                    )

                    w = min(
                        channel.shape[1],
                        self.yrange
                    )

                    padded[:h,:w] = channel

                    channel = padded

                padded_channels.append(channel)


            self.raster_cache[voxel_id] = np.stack(
                padded_channels
            )

    def cnnInput(self, target_index):

        targetVoxel = self.getVoxelID(target_index)
        voxelCoords = self.getVoxelCoords(targetVoxel)

        layout = self.voxel_layout[targetVoxel]

        tensor = self.empty_tensor.copy()


        tensor[:self.raster_channels] = (
            self.raster_cache[targetVoxel][:,:,:,None]
        )


        z_indices = np.arange(self.zrange) - self.zrange//2

        voxel_depth_grid = (
            voxelCoords[2]
            -
            z_indices[None,None,:] * self.voxel_size[2]
        )

        voxel_depth_grid = np.broadcast_to(
            voxel_depth_grid,
            (
                self.xrange,
                self.yrange,
                self.zrange
            )
        )


        tensor[self.raster_channels+17] = (
            voxel_depth_grid - self.Depth[target_index]
        ) / TOTAL_PATCH_SIZE[2]


        tensor[self.raster_channels+16] = (
            voxel_depth_grid / self.maxDepth
        )


        for thisVoxel,vx,vy,vz,norm_x,norm_y in layout:

            stats = self.voxel_stats[thisVoxel]

            n = int(stats["well_count"])


            # =========================
            # REAL VOXEL
            # =========================

            if n > 0:


                mean = (
                    stats["arsenic_sum"]
                    /
                    n
                )

                median = stats["median"]
                p10 = stats["p10"]
                p25 = stats["p25"]
                p75 = stats["p75"]
                p90 = stats["p90"]
                p95 = stats["p95"]


                depth_mean = (
                    stats["depth_sum"]
                    /
                    n
                )


                depth_std = np.sqrt(
                    max(
                        0,
                        stats["depth_sq_sum"]/n
                        -
                        depth_mean**2
                    )
                )


                confidence = 1.0



            # =========================
            # EMPTY VOXEL -> IDW
            # =========================

            else:


                (
                    mean,
                    _,
                    median,
                    p10,
                    p25,
                    p75,
                    p90,
                    p95,
                    confidence

                ) = self.calculateVoxelIDW(thisVoxel)


                depth_mean = (
                    self.voxels[thisVoxel]["centroid_z"]
                )

                depth_std = 0



            # =========================
            # ARSENIC DISTRIBUTION
            # =========================


            tensor[
                self.raster_channels+0,
                vx,vy,vz
            ] = np.log1p(n)


            tensor[
                self.raster_channels+1,
                vx,vy,vz
            ] = mean / self.maxLogArsenic


            tensor[
                self.raster_channels+2,
                vx,vy,vz
            ] = median


            tensor[
                self.raster_channels+3,
                vx,vy,vz
            ] = p10


            tensor[
                self.raster_channels+4,
                vx,vy,vz
            ] = p90


            tensor[
                self.raster_channels+5,
                vx,vy,vz
            ] = p25


            tensor[
                self.raster_channels+6,
                vx,vy,vz
            ] = p75


            tensor[
                self.raster_channels+7,
                vx,vy,vz
            ] = p95



            # spread features

            tensor[
                self.raster_channels+8,
                vx,vy,vz
            ] = (
                p75-p25
            )


            tensor[
                self.raster_channels+9,
                vx,vy,vz
            ] = (
                p95-p90
            )


            tensor[
                self.raster_channels+10,
                vx,vy,vz
            ] = (
                p25-p10
            )



            # =========================
            # DEPTH FEATURES
            # =========================


            tensor[
                self.raster_channels+11,
                vx,vy,vz
            ] = (
                depth_mean /
                self.maxDepth
            )


            tensor[
                self.raster_channels+12,
                vx,vy,vz
            ] = (
                depth_std /
                self.maxDepth
            )



            # confidence

            tensor[
                self.raster_channels+13,
                vx,vy,vz
            ] = confidence



            # voxel location

            tensor[
                self.raster_channels+14,
                vx,vy,vz
            ] = norm_x


            tensor[
                self.raster_channels+15,
                vx,vy,vz
            ] = norm_y

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