# Readme

## install required libraries
- mediapipe : https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=fr 
- opencv
- yolo (if using the yolo model for pose estimation) : https://docs.ultralytics.com/fr/tasks/pose/ 

`pip install mediapipe opencv-python ultralytics`

For yolo, please also install the correct pytorch for the CUDA version of your GPU device. CPU will be used if no GPU is found.

## Running the analysis

There are 2 versions : 
- ArcheryPoseAnalysis.py --> use Yolo for pose estimation and mediapipe for hand landmarks
- ArcheryPoseAnalysis_MPonly --> use only mediapipe for pose and hand landmarks

### interactive session:
`python ArcheryPoseEstimation.py` or `python ArcheryPoseEstimation_MPonly.py`
Then, choose between : 
- webcam feed
- video
- images

And whether to include Hands detection.

### Command line running
```
# Webcam with hands tracking 
python ArcheryPoseEstimation.py --mode webcam --hands --output ma_video

# Vidéo without hands tracking
python ArcheryPoseEstimation.py --mode video --input video.mp4 --no-hands

# Folder with images and hands tracking
python ArcheryPoseEstimation.py --mode images --input ./my_images --hands
```
