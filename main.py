# install ultralytics
!pip install ultralytics
# install libraries
from ultralytics import YOLO
from ultralytics.solutions import ai_gym
import cv2

model = YOLO("yolov8n-pose.pt")
# Enter the video link
cap = cv2.VideoCapture("ENTER THE VIDEO LINK...")
assert cap.isOpened(), "Error reading video file"

video_writer = cv2.VideoWriter("workouts.avi",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                int(cap.get(5)),
                                (int(cap.get(3)), int(cap.get(4))))

gym_object = ai_gym.AIGym()  # init AI GYM module
gym_object.set_args(line_thickness=2,
                    view_img=True,
                    pose_type="pushup",
                    kpts_to_check=[6, 8, 10])

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
      print("Video frame is empty or video processing has been successfully completed.")
      break
    frame_count += 1
    results = model.predict(im0, verbose=False)
    im0 = gym_object.start_counting(im0, results, frame_count)
    video_writer.write(im0)

cv2.destroyAllWindows()
video_writer.release()
