import face_recognition
import cv2
import math

video_capture = cv2.VideoCapture(0)

# Reduce the resolution to make processing faster
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Check if frame is read properly
    if not ret:
        print("Failed to capture image. Retrying...")
        continue

    # Resize frame to 50% of its original size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect face landmarks in the resized frame
    face_landmarks = face_recognition.face_landmarks(small_frame)
    # If face landmarks are detected, process them
    if face_landmarks:
        try:
            # Scale back the landmarks to the original frame size
            for face_landmark in face_landmarks:
                for feature in face_landmark.keys():
                    face_landmark[feature] = [(int(x * 2), int(y * 2)) for (x, y) in face_landmark[feature]]


            p1 = face_landmarks[0]['top_lip']
            p2 = face_landmarks[0]['bottom_lip']

            # Calculate the distance between the upper and lower lips
            x1, y1 = p1[9]
            x3, y3 = p1[8]
            x4, y4 = p1[10]
            x2, y2 = p2[9]
            x5, y5 = p2[8]
            x6, y6 = p2[10]
            dist = math.sqrt(((x2 + x5 + x6) - (x1 + x3 + x4)) ** 2 + ((y2 + y5 + y6) - (y1 + y3 + y4)) ** 2)
            if dist < 20:
                print("CLOSED - ", dist)
            else:
                print("OPEN - ", dist)

            # Draw circles around the relevant lip points on the original frame
            image = cv2.circle(frame, p1[8], 1, (255, 255, 255, 0), 2)
            image = cv2.circle(frame, p1[9], 1, (255, 255, 255, 0), 2)
            image = cv2.circle(frame, p1[10], 1, (255, 255, 255, 0), 2)

            image = cv2.circle(frame, p2[8], 1, (255, 255, 255, 0), 2)
            image = cv2.circle(frame, p2[9], 1, (255, 255, 255, 0), 2)
            image = cv2.circle(frame, p2[10], 1, (255, 255, 255, 0), 2)

            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.imshow('Video', frame)
        except Exception as e:
            print(f"Error processing landmarks: {e}")
    else:
        print("No face detected. Waiting for face...")

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
