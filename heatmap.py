from heatmappy import Heatmapper
from PIL import Image
import cv2
import os
import pandas as pd

coorList = list()
csv = "workdir/mot.mp4.csv"       # csv file to use
csvData = pd.read_csv(csv, header=0)   # csv import
video = "workdir/mot.mp4"         # video file to use

heatmapper = Heatmapper(
    point_diameter=35,                 # the size of each point to be drawn
    point_strength=0.2,                # the strength, between 0 and 1, of each point to be drawn
    opacity=0.65,                      # the opacity of the heatmap layer
    colours='default',                 # 'default' or 'reveal'
                                       # OR a matplotlib LinearSegmentedColorMap object 
                                       # OR the path to a horizontal scale image
    grey_heatmapper='PIL'              # the object responsible for drawing the points
                                       # pillow used by default, 'PySide' option available if installed
)

assert os.path.isfile(csv), \
'csv {} does not exist'.format(csv)
assert os.path.isfile(video), \
'video {} does not exist'.format(video)

# Capture the first frame of the video
videoFeed = cv2.VideoCapture(video)

assert videoFeed.isOpened(), \
'Cannot capture source'

_, cv2_image = videoFeed.read()
cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
image = Image.fromarray(cv2_image)

for i in range(1, len(csvData)):
    coorList.append((csvData.iloc[i, 2], csvData.iloc[i, 3]))

heatmap = heatmapper.heatmap_on_img(coorList, image)
heatmap.save('workdir/heatmap.png')
videoFeed.release()
cv2.destroyAllWindows()
exit('Finished')