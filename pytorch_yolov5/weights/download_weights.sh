#!/bin/bash
# Download common models

python -c "
from utils.google_utils import *;
attempt_download('weights/yolov5x.pt')
"
