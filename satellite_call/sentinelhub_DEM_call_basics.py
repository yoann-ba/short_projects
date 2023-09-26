



#%% imports

import numpy as np
import matplotlib.pyplot as plt

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, DataCollection, bbox_to_dimensions, SHConfig

import cv2


#%% Configure SH account

# trial sentinelhub accounts are free for 30 days
config = SHConfig()
config.sh_client_id = 'your client id here'
config.sh_client_secret = 'your secret here'


#%% BBox

# points in lat, lon format
# wrote so that they're easy to copy paste in and out of
# here is vesuvius
point_0 = [
    40.86778718112351, 14.376045123413437
    ]
point_1 = [
    40.78006745292518, 14.499006026619918
    ]
BBOX = [min(point_0[1], point_1[1]), min(point_0[0], point_1[0]), 
        max(point_0[1], point_1[1]), max(point_0[0], point_1[0])]
image_bbox = BBox(bbox=BBOX, crs=CRS.WGS84)
image_size = bbox_to_dimensions(image_bbox, resolution=30)

#%% params + eval script

# timelaps
date_start = '2022-01-01'
date_end = '2022-12-21'

# list of evalscripts to get
# this one just returns the raw height value instead of a colormapped value
evalscript_DEM = """
//VERSION=3
function setup() {
  return {
    input: ["DEM"],
    output:{
      id: "default",
      bands: 1,
      sampleType: SampleType.FLOAT32
    }
  }
}
function evaluatePixel(sample) {
  return [sample.DEM]
}
"""

# Get the right data collection
data_col = DataCollection.DEM_COPERNICUS_30

#%% API call

# Initialize the request
request = SentinelHubRequest(
    data_folder='.',
    evalscript=evalscript_DEM,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=data_col,
            time_interval=(date_start, date_end),
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=image_bbox,
    size=image_size,
    config=config
)

# save the image directly
# request.save_data()

# save and show the image
imgs = request.get_data(save_data=True)
image = imgs[0]


#%% work with data

# plot, save, w/e






