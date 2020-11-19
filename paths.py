from PIL import Image
from pykalman import KalmanFilter
import cv2 as cv
import os
import numpy as np
import pandas as pd

# configuration
csv = "workdir/mot.mp4.csv"       # csv file to use
video = "workdir/mot.mp4"         # video file to use

# check if the file does exist or not
assert os.path.isfile(csv), \
'{} not found or missing'.format(csv)
assert os.path.isfile(video), \
'{} not found or missing'.format(video)

# csv import
print("Importing csv...")
df = pd.read_csv(csv, header=0)
df.sort_values(['track_id', 'frame_id'], inplace=True) # sort the data based on track_id and frame_id
df.drop(['frame_id', 'track_id'], axis=1, inplace=True) # drop the id as we no longer need those
paths = df.to_numpy() # convert it to numpy for further manipulation

print("Running kalman filter smoothing...")
# kalman filter smoothing
initial_state_mean = [paths[0, 0],
                      0,
                      paths[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

kf1 = kf1.em(paths, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(paths)
smoothed_state_means = smoothed_state_means.round() # round the data to 0 decimal
smoothed_state_means = smoothed_state_means.astype(int) # cast into int for opencv

print("Capturing the video frame...")
# capture the last frame of the video
videoFeed = cv.VideoCapture(video)
videoFeed.set(1, videoFeed.get(7)-1) 
assert videoFeed.isOpened(), \
'Cannot capture source' # check if the camera/video can be opened
_, image = videoFeed.read()

print("Drawing the path...")
# draw the path to the frame
for i in range(0, len(df) - 1):
    cv.line(image, (smoothed_state_means[i, 0], smoothed_state_means[i, 2]), (smoothed_state_means[i + 1, 0], smoothed_state_means[i + 1, 2]), (0, 0, 210), 1, cv.LINE_AA)

# write it to img
cv.imwrite('workdir/path.png', image) 
videoFeed.release()
cv.destroyAllWindows()
exit('Finished')