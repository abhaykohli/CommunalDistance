# ---------------------------yolo.py config options-----------------------------#
# minimum probability to filter weak detections along with
MIN_CONFIDENCE = 0.3
# the threshold when applying non-maxima suppression
NMS_THRESHOLD = 0.3


# --------------------------thread.py config options----------------------------#
# turn multi-threading ON or OFF
USE_THREADING = True
# configure FPS rate for input source
FPS = 30


# ---------------------------alert.py config options----------------------------#
# set threshold no. of unsafe people over which alert needs to be sent
UNSAFE_THRESHOLD = 15
# turn email alert feature ON or OFF
SEND_MAIL_ALERTS = False
# set the mail address on which to receive the real-time alerts
MAIL = ""


# ---------------------------main.py config options-----------------------------#
# Configure the input source
#   for webcam, set url = 0
#   for ip camera, set ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
SOURCE_URL = 0
# set if GPU should be used for computations; otherwise uses the CPU by default
USE_GPU = True
# define the max/min safe distance limits (in pixels) between 2 people
MAX_DISTANCE = 80
MIN_DISTANCE = 50
