import os
import sys
import numpy as np
from plyfile import PlyData

src = sys.argv[1]
ply1 = PlyData.read(os.path.join(src, "point_cloud_quantized.ply"))
ply2 = PlyData.read(os.path.join(src, "point_cloud_quantized.drc.decode.ply"))
for prop in ply1['vertex'].properties:
    print(prop.name, np.abs(ply1['vertex'][prop.name] - ply2['vertex'][prop.name]).max())
