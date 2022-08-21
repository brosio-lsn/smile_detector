import cv2 as cv
import mediapipe as mp
from datetime import datetime

# initialize mediapipe face detection elements
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


# function to calculate square distance between 2 points
def square_distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


# function to calculate the lips length / chin length ratio
def get_lips_chin_ratio(landmarks, image):
    # calculate lips square length
    left_corner_lip = landmarks.landmark[76]
    x_left_lip, y_left_lip = int(left_corner_lip.x * image.shape[1]), int(left_corner_lip.y * image.shape[0])

    right_corner_lip = landmarks.landmark[308]
    x_right_lip, y_right_lip = int(right_corner_lip.x * image.shape[1]), int(right_corner_lip.y * image.shape[0])

    lips_square_length = square_distance(x_left_lip, y_left_lip, x_right_lip, y_right_lip)

    # calculate chin's square length
    left_corner_chin = landmarks.landmark[140]
    x_left_chin, y_left_chin = int(left_corner_chin.x * image.shape[1]), int(left_corner_chin.y * image.shape[0])

    right_corner_chin = landmarks.landmark[378]
    x_right_chin, y_right_chin = int(right_corner_chin.x * image.shape[1]), int(right_corner_chin.y * image.shape[0])

    chin_square_length = square_distance(x_left_chin, y_left_chin, x_right_chin, y_right_chin)

    return lips_square_length / chin_square_length


# getting the pc video
capture = cv.VideoCapture(0)
# getting time ref
time_ref = datetime.now()
# define the time after which the no smile ratio is measured
wait_time_before_non_smile_measures = 3
# boolean to calculate the non smile ratio only once
non_smile_ratio_done = True
# boolean to write the selfy in the computer only once
selfy_done = False
# time ref for the time to wait to save the frame after the first smile has been detected (cause the process of
# spreading lips takes time ) here we estimate it to be 0.6 seconds
new_time_ref_taken = False
time_to_wait_before_selfy = 0.6

# reading the video
while capture.isOpened():
    isTrue, frame = capture.read()
    convert = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # before the no smile ratio is calculated a message is displayed
    if ((datetime.now() - time_ref).total_seconds() < wait_time_before_non_smile_measures):
        cv.putText(frame, "dont't smile in " + "{:.2f}".format(
            (wait_time_before_non_smile_measures - (datetime.now() - time_ref).total_seconds())) + "seconds",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 2)
    # detect/track the face in the image and calculate the landmarks
    if ((datetime.now() - time_ref).total_seconds() > wait_time_before_non_smile_measures):
        results = face_mesh.process(convert)
        if results.multi_face_landmarks:
            # calculate the no smile ratio
            # if the no smile ratio hasn't been calculated it is calculated
            if non_smile_ratio_done:
                non_smile_ratio = get_lips_chin_ratio(results.multi_face_landmarks[0], frame)
                non_smile_ratio_done = False
            # else we calculate the current ratio to detect a smile
            else:
                current_ratio = get_lips_chin_ratio(results.multi_face_landmarks[0], frame)
                if (current_ratio > 1.4 * non_smile_ratio):
                    # write a message if the smile is detected
                    cv.putText(frame, "smile detected", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 2)
                    # when the first smile is detected, we take the selfy
                    if not selfy_done:
                        if not new_time_ref_taken:
                            new_time_ref = datetime.now()
                            new_time_ref_taken = True
                        # we wait a short time to take the selfy after the first smile has been detected
                        if (datetime.now() - new_time_ref).total_seconds() > time_to_wait_before_selfy:
                            cv.imwrite("selfy_image.jpg", frame)
                            selfy_done = True

    # show the frame
    cv.imshow('video', frame)
    # exit the video by pressing q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
