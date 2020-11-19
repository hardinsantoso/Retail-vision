import os
from darkflow.darkflow.defaults import argHandler
from darkflow.darkflow.net.build import TFNet


FLAGS = argHandler()
FLAGS.setDefaults()

FLAGS.demo = "workdir/mot.mp4"      # path to video, or if camera put "camera"
FLAGS.model = "darkflow/cfg/yolo.cfg"    # tensorflow model
FLAGS.load = "darkflow/bin/yolo.weights" # tensorflow weights
FLAGS.threshold = 0.4                    # threshold of detection confidence
FLAGS.gpu = 0.6                          # how much GPU resource to use (0..1; 0 means cpu only) 
FLAGS.track = True                       # track the object or only do object detection
FLAGS.trackObj = ['person']              # object to be tracked
FLAGS.saveVideo = True                   # flag to save the processed video
FLAGS.BK_MOG = True                      # activate background substraction using cv2 MOG substraction
FLAGS.tracker = "deep_sort"              # tracking algorithm (deep_sort/sort)
FLAGS.skip = 1                           # how many frames to skip between each detection to speed up the network
FLAGS.csv = True                         # flag to save the tracking CSV
FLAGS.display = True                     # flag to display the process in real time

tfnet = TFNet(FLAGS)

tfnet.camera()
exit('Demo stopped, exit.')
