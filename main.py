import config, thread
from alert import Mailer
from yolo import detect_people
from imutils.video import VideoStream
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import schedule

# ------------------------Parse command line arguments--------------------------#
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, default="", help="(optional) path to input video file"
)
parser.add_argument(
    "-o", "--output", type=str, default="", help="(optional) path to output video file"
)
parser.add_argument(
    "-d",
    "--display",
    type=int,
    default=1,
    help="toggle output display frame [0 = disable, 1 = enable (default)]",
)
args = vars(parser.parse_args())
# ------------------------------------------------------------------------------#
LABELS_PATH = r"yolo\coco.names"
WEIGHTS_PATH = r"yolo\yolov3.weights"
CONFIG_PATH = r"yolo\yolov3.cfg"

# load the COCO class labels our YOLO model was trained on
cocoLabels = open(LABELS_PATH).read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print()
    print("[INFO] Looking for GPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the output layer names that we need from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if no input video source, grab a reference to webcam/ip camera
if not args.get("input", False):
    print("[INFO] Starting live stream analysis")
    videoSource = cv2.VideoCapture(config.SOURCE_URL)
    if config.USE_THREADING:
        cap = thread.ThreadingClass(config.SOURCE_URL)
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] Starting video analysis")
    videoSource = cv2.VideoCapture(args["input"])
    # start background thread for reading frames if multi-threading is enabled
    if config.USE_THREADING:
        cap = thread.ThreadingClass(args["input"])
        time.sleep(2)

writer = None

# start main loop
while True:
    # read the next frame from ThreadingClass if multi-threading is enabled
    if config.USE_THREADING:
        frame = cap.read()
    # otherwise read the frame directly from video source
    else:
        (grabbed, frame) = videoSource.read()
        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

    # resize the frame and then detect people in it using YOLO
    frame = imutils.resize(frame, width=700)
    results = detect_people(
        frame, net, layerNames, personIdx=cocoLabels.index("person")
    )

    # initialize the set of indexes that violate the max/min social distance limits
    serious = set()
    abnormal = set()

    # ensure there are at least two people (required in order to compute our
    # pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update our violation set with the indexes of the centroid pairs
                    serious.add(i)
                    serious.add(j)
                # update our abnormal set if the centroid distance is below max distance limit
                elif D[i, j] < config.MAX_DISTANCE:
                    abnormal.add(i)
                    abnormal.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)  # green

        # if the index pair exists within the violation/abnormal sets, then update the color
        if i in serious:
            color = (0, 0, 255)  # red
        elif i in abnormal:
            color = (0, 255, 255)  # yellow

        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
        cv2.circle(frame, (cX, cY), 2, color, 1)

    # ------------------------------------------------------------------------------#
    # display the information on frame
    # draw a black rectangle at bottom of frame
    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 65),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1,
    )
    # draw text indicating safe distance info
    safeDistanceText = "Safe distance: >{} px".format(config.MAX_DISTANCE)
    cv2.putText(
        frame,
        safeDistanceText,
        (160, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    # draw text indicating threshold limit info
    thresholdLimitText = "Threshold limit: {}".format(config.UNSAFE_THRESHOLD)
    cv2.putText(
        frame,
        thresholdLimitText,
        (160, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    # draw text displaying human count
    humanCountText = "Human count: {}".format(len(results))
    cv2.putText(
        frame,
        humanCountText,
        (160, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    # draw text displaying no. of serious violations
    seriousViolationsText = "Total serious violations: {}".format(len(serious))
    cv2.putText(
        frame,
        seriousViolationsText,
        (330, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    # draw text displaying no. of abnormal violations
    abnormalViolationsText = "Total abnormal violations: {}".format(len(abnormal))
    cv2.putText(
        frame,
        abnormalViolationsText,
        (330, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )

    # ------------------------------Alert function----------------------------------#
    if len(serious) >= config.UNSAFE_THRESHOLD:
        # display alert text in case of threshold limit crossed
        cv2.putText(
            frame,
            "-ALERT: Violations over limit-",
            (330, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 255),
            2,
        )
        if config.SEND_MAIL_ALERTS:
            print("[INFO] Sending mail alert")
            Mailer().send(config.MAIL)
            print("[INFO] Mail sent")
        # config.ALERT = False
    # ------------------------------------------------------------------------------#
    # check to see if the output frame should be displayed on screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Social Distance Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True
        )

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(frame)

# close any open windows
cv2.destroyAllWindows()
