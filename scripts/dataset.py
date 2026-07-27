import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import rowcol
import os
from sklearn.neighbors import KDTree
from dataloader import RISK_CLASSES

MAIN_FOLDER = "../data"
RASTER_FOLDER = "./rasters"
VOXELS_FOLDER = "./voxels"
CSV_FILE = "wells.csv"
CHEM_FILE = "rawChemistryDataAdm4.csv"
TOTAL_PATCH_SIZE = [2250, 2250, 50]

class ArsenicDataset:
    def __init__(self):
        self.df = pd.read_csv(os.path.join(MAIN_FOLDER, CSV_FILE))
        self.df = (self.df.dropna().reset_index(drop=True))
        counts = self.df.groupby("mou")["mou"].transform("size")
        self.df = self.df[counts >= 7]

        chem=pd.read_csv(os.path.join(MAIN_FOLDER,CHEM_FILE))
        chem=chem.dropna(subset=["X","Y","WELL_DEPTH"]).reset_index(drop=True)
        chem_features=["Fe","Mn","SO4","Ca","Mg","Na","Si","P_"]
        for col in chem_features:
            chem[col] = chem[col].apply(self.clean_censored)
        chem[chem_features] = (chem[chem_features].fillna(chem[chem_features].median()))

        chem_values = chem[chem_features].values.astype(np.float32)
        self.chem_mean = chem_values.mean(axis=0)
        self.chem_std = chem_values.std(axis=0) + 1e-6
        chem_values = (chem_values - self.chem_mean) / self.chem_std
        self.chem_values = chem_values
        chem_coords=np.column_stack([chem["X"],chem["Y"],chem["WELL_DEPTH"]])
        self.chem_tree=KDTree(chem_coords)

        chem_distances, _ = self.chem_tree.query(
            chem_coords,
            k=5
        )

        chem_distances = chem_distances[:,1:] # remove itself

        self.chem_dist_mean = np.log1p(
            chem_distances.mean()
        )

        self.chem_dist_std = np.log1p(
            chem_distances.std()
        ) + 1e-6

        self.X = self.df["X"].values.astype(np.float32)
        self.Y = self.df["Y"].values.astype(np.float32)

        self.maxX = self.X.max()
        self.maxY = self.Y.max()

        self.Depth = self.df["Depth"].values.astype(np.float32)
        self.Arsenic = self.df["Arsenic"].values.astype(np.float32)
        self.logArsenic = np.log1p(self.Arsenic)

        self.x_mean = self.X.mean()
        self.x_std = self.X.std()
        self.y_mean = self.Y.mean()
        self.y_std = self.Y.std()
        self.depth_mean = self.Depth.mean()
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

        self.empty_tensor = np.zeros((self.raster_channels+28,self.xrange,self.yrange,self.zrange),dtype=np.float32)

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
                ("chem_Fe",np.float32),
                ("chem_Mn",np.float32),
                ("chem_SO4",np.float32),
                ("chem_Ca",np.float32),
                ("chem_Mg",np.float32),
                ("chem_Na",np.float32),
                ("chem_Si",np.float32),
                ("chem_P",np.float32),
                ("chem_distance",np.float32),
                ("chem_count",np.float32)
            ],
        )

        for voxel in self.voxels:
            voxel_id = voxel["voxel_id"]
            start = voxel["well_start"]
            count = voxel["well_count"]

            if count == 0:
                continue

            chem=self.calculateVoxelChemistry(voxel_id)
            wells = self.voxel_wells[start:start + count]

            arsenic = self.logArsenic[wells]
            depth = self.Depth[wells]

            self.voxel_stats["mean"][voxel_id] = arsenic.mean()
            self.voxel_stats["std"][voxel_id] = arsenic.std()

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

            self.voxel_stats["chem_Fe"][voxel_id]=chem[0]
            self.voxel_stats["chem_Mn"][voxel_id]=chem[1]
            self.voxel_stats["chem_SO4"][voxel_id]=chem[2]
            self.voxel_stats["chem_Ca"][voxel_id]=chem[3]
            self.voxel_stats["chem_Mg"][voxel_id]=chem[4]
            self.voxel_stats["chem_Na"][voxel_id]=chem[5]
            self.voxel_stats["chem_Si"][voxel_id]=chem[6]
            self.voxel_stats["chem_P"][voxel_id]=chem[7]

            self.voxel_stats["chem_distance"][voxel_id]=chem[8]
            self.voxel_stats["chem_count"][voxel_id]=chem[9]

        self.maxWellCount = self.voxel_stats["well_count"].max()

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

    def calculateVoxelChemistry(self,voxel_id):
        voxel=self.voxels[voxel_id]
        point=np.array([voxel["centroid_x"],voxel["centroid_y"],voxel["centroid_z"]]).reshape(1,-1)
        dist,index=self.chem_tree.query(point,k=5)
        values=self.chem_values[index[0]]
        weights=1/(dist[0]+1e-6)
        weights/=weights.sum()
        chem=np.sum(values*weights[:,None],axis=0)
        return np.append(chem,[dist.mean(),len(values)]).astype(np.float32)

    def clean_censored(self,value):

        if isinstance(value,str):

            value=value.strip()

            if value.startswith("<"):
                number=float(value.replace("<","").strip())
                return number/2

        return float(value)

    def getVoxelCoords(self, voxel_id):
        voxel = self.voxels[voxel_id]
        return [voxel["centroid_x"],voxel["centroid_y"],voxel["centroid_z"]]

    def getNeighbours(self, voxel_index):
        offset = self.voxel_neighbour_offsets[voxel_index]

        start = offset["neighbour_start"]
        count = offset["neighbour_count"]

        neighbours = self.voxel_neighbours[start:start + count]

        return neighbours

    def getVoxelStats(self, voxel_id, target_index):
        voxel = self.voxels[voxel_id]

        start = voxel["well_start"]
        count = voxel["well_count"]

        if count == 0:
            return None

        wells = self.voxel_wells[start:start+count]

        # Remove target if it is in this voxel
        wells = wells[wells != target_index]

        if len(wells) == 0:
            return None

        arsenic = self.logArsenic[wells]
        depth = self.Depth[wells]

        return {
            "well_count": len(wells),
            "mean": arsenic.mean(),
            "median": np.quantile(arsenic, 0.50),
            "p10": np.quantile(arsenic, 0.10),
            "p25": np.quantile(arsenic, 0.25),
            "p75": np.quantile(arsenic, 0.75),
            "p90": np.quantile(arsenic, 0.90),
            "p95": np.quantile(arsenic, 0.95),
            "arsenic_sum": arsenic.sum(),
            "depth_sum": depth.sum(),
            "depth_sq_sum": np.square(depth).sum(),
        }

    def calculateIDWVoxel(self, coords, neighbours, target_voxel, target_well):

        vox = self.voxels[neighbours]

        stats = self.voxel_stats[neighbours].copy()

        target_pos = np.where(neighbours == target_voxel)[0]

        if len(target_pos):

            i = target_pos[0]

            target_stats = self.getVoxelStats(
                target_voxel,
                target_well
            )

            if target_stats is not None:
                for key, value in target_stats.items():
                    stats[key][i] = value
            else:
                stats["well_count"][i] = 0


        # remove empty voxels
        mask = stats["well_count"] > 0

        if not np.any(mask):
            return np.zeros((len(coords), 12), dtype=np.float32)

        vox = vox[mask]
        stats = stats[mask]

        target_z = self.voxels[target_voxel]["centroid_z"]

        mask = np.isclose(
            vox["centroid_z"],
            target_z
        )

        if not np.any(mask):
            return np.zeros((len(coords), 12), dtype=np.float32)

        vox = vox[mask]
        stats = stats[mask]

        vx = vox["centroid_x"]
        vy = vox["centroid_y"]


        cx = coords[:, 0, None]
        cy = coords[:, 1, None]


        voxel_dx = (vx - cx) / self.voxel_size[0]
        voxel_dy = (vy - cy) / self.voxel_size[1]


        distance = np.sqrt(
            voxel_dx ** 2 +
            voxel_dy ** 2
        ) + 1e-6


        weights = 1 / (distance * 5)

        weights *= np.log1p(
            stats["well_count"]
        )

        weights /= weights.sum(
            axis=1,
            keepdims=True
        )


        mean = stats["mean"]
        median = stats["median"]
        p10 = stats["p10"]
        p25 = stats["p25"]
        p75 = stats["p75"]
        p90 = stats["p90"]
        p95 = stats["p95"]


        results = np.zeros(
            (len(coords), 12),
            dtype=np.float32
        )


        results[:,0] = (
            np.sum(weights * mean, axis=1)
            / self.maxLogArsenic
        )

        results[:,2] = np.sum(weights * median, axis=1)
        results[:,3] = np.sum(weights * p10, axis=1)
        results[:,4] = np.sum(weights * p25, axis=1)
        results[:,5] = np.sum(weights * p75, axis=1)
        results[:,6] = np.sum(weights * p90, axis=1)
        results[:,7] = np.sum(weights * p95, axis=1)


        results[:,1] = (
            results[:,5]
            -
            results[:,4]
        )


        well_strength = np.mean(
            np.log1p(stats["well_count"])
        )

        well_strength /= np.log1p(
            self.maxWellCount
        )


        distance_strength = np.max(
            np.exp(-distance),
            axis=1
        )


        confidence = (
            np.clip(well_strength,0,0.5)
            +
            np.clip(distance_strength,0,0.5)
        )


        results[:,8] = confidence


        results[:,9] = (
            coords[:,0] - self.x_mean
        ) / self.x_std

        results[:,10] = (
            coords[:,1] - self.y_mean
        ) / self.y_std

        results[:,11] = (
            coords[:,2] - self.depth_mean
        ) / self.depth_std


        return results

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

                py, px = rowcol(transform, cx, cy)

                half = self.xrange // 2
                crop = data[py-half:py+half+1,px-half:px+half+1 ].copy()

                if raster["isEmbedded"]:
                    crop[crop < 0] = 0
                    classes = raster["classes"]

                    embedding = np.eye(classes,dtype=np.float32)[crop.astype(np.int32)]
                    embedding = np.moveaxis(embedding,-1,0)

                    for channel in embedding:
                        patch_channels.append(channel)

                else:
                    crop = np.nan_to_num((crop - raster["mean"]) /(raster["std"] + 1e-6),nan=0)
                    patch_channels.append(crop)

            padded_channels = []

            for channel in patch_channels:
                if channel.shape != (self.xrange,self.yrange):
                    padded = np.zeros((self.xrange,self.yrange),dtype=np.float32)
                    h = min(channel.shape[0], self.xrange)
                    w = min(channel.shape[1],self.yrange)
                    padded[:h,:w] = channel
                    channel = padded

                padded_channels.append(channel)

            self.raster_cache[voxel_id] = np.stack(padded_channels)

    def cnnInput(self, target_index):

        targetVoxel = self.getVoxelID(target_index)
        voxelCoords = self.getVoxelCoords(targetVoxel)

        tensor = self.empty_tensor.copy()

        tensor[:self.raster_channels] = (
            self.raster_cache[targetVoxel][:,:,:,None]
        )

        z_indices = np.arange(self.zrange) - self.zrange//2


        # generate all voxel coordinates
        x = (
            np.arange(self.xrange)-self.xrange//2
        ) * self.voxel_size[0]

        y = (
            np.arange(self.yrange)-self.yrange//2
        ) * self.voxel_size[1]

        z = (
            np.arange(self.zrange)-self.zrange//2
        ) * self.voxel_size[2]


        gx,gy,gz = np.meshgrid(
            x,y,z,
            indexing="ij"
        )

        coords = np.stack([
            gx.ravel()+voxelCoords[0],
            gy.ravel()+voxelCoords[1],
            voxelCoords[2]-gz.ravel()
        ],axis=1)


        neighbours = self.getNeighbours(targetVoxel)

        neighbours = np.append(
            neighbours,
            targetVoxel
        )

        idw = self.calculateIDWVoxel(
            coords,
            neighbours,
            targetVoxel,
            target_index
        )

        idw = idw.reshape(
            self.xrange,
            self.yrange,
            self.zrange,
            12
        )

        tensor[self.raster_channels+1] = idw[:,:,:,0]
        tensor[self.raster_channels+2] = idw[:,:,:,2]
        tensor[self.raster_channels+3] = idw[:,:,:,3]
        tensor[self.raster_channels+5] = idw[:,:,:,4]
        tensor[self.raster_channels+6] = idw[:,:,:,5]
        tensor[self.raster_channels+4] = idw[:,:,:,6]
        tensor[self.raster_channels+7] = idw[:,:,:,7]

        tensor[self.raster_channels+8] = (
            idw[:,:,:,5]-idw[:,:,:,4]
        )

        tensor[self.raster_channels+9] = (
            idw[:,:,:,7]-idw[:,:,:,6]
        )

        tensor[self.raster_channels+10] = (
            idw[:,:,:,4]-idw[:,:,:,3]
        )

        tensor[self.raster_channels+13] = idw[:,:,:,8]
        tensor[self.raster_channels+14] = idw[:,:,:,9]
        tensor[self.raster_channels+15] = idw[:,:,:,10]
        tensor[self.raster_channels+16] = idw[:,:,:,11] #absolute depth
        tensor[self.raster_channels+17] = -z / 20 #relative depth

        # overwrite measured voxels
        for thisVoxel,vx,vy,vz,norm_x,norm_y in self.voxel_layout[targetVoxel]:

            stats = self.voxel_stats[thisVoxel]
            n = int(stats["well_count"])

            if n == 0:
                continue


            mean = stats["arsenic_sum"]/n

            median = stats["median"]
            p10 = stats["p10"]
            p25 = stats["p25"]
            p75 = stats["p75"]
            p90 = stats["p90"]
            p95 = stats["p95"]


            depth_mean = stats["depth_sum"]/n

            depth_std = np.sqrt(
                max(
                    0,
                    stats["depth_sq_sum"]/n-depth_mean**2
                )
            )


            tensor[
                self.raster_channels+0,
                vx,vy,vz
            ] = np.log1p(n)

            tensor[
                self.raster_channels+1,
                vx,vy,vz
            ] = mean/self.maxLogArsenic

            tensor[
                self.raster_channels+2:self.raster_channels+8,
                vx,vy,vz
            ] = [
                median,
                p10,
                p90,
                p25,
                p75,
                p95
            ]

            tensor[
                self.raster_channels+8,
                vx,vy,vz
            ] = p75-p25

            tensor[
                self.raster_channels+9,
                vx,vy,vz
            ] = p95-p90

            tensor[
                self.raster_channels+10,
                vx,vy,vz
            ] = p25-p10


            tensor[
                self.raster_channels+11,
                vx,vy,vz
            ] = depth_mean / self.maxDepth

            tensor[
                self.raster_channels+12,
                vx,vy,vz
            ] = depth_std / (self.maxDepth / (2 * TOTAL_PATCH_SIZE[2]))


            tensor[
                self.raster_channels+13,
                vx,vy,vz
            ] = 1.0


            tensor[
                self.raster_channels+18:self.raster_channels+26,
                vx,vy,vz
            ] = [
                stats["chem_Fe"],
                stats["chem_Mn"],
                stats["chem_SO4"],
                stats["chem_Ca"],
                stats["chem_Mg"],
                stats["chem_Na"],
                stats["chem_Si"],
                stats["chem_P"]
            ]

            tensor[
                self.raster_channels+26,
                vx,vy,vz
            ] = (
                np.log1p(stats["chem_distance"])
                -
                self.chem_dist_mean
            ) / self.chem_dist_std


            tensor[
                self.raster_channels+27,
                vx,vy,vz
            ] = np.log1p(stats["chem_count"])


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