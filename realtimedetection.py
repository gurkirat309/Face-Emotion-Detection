import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

# Load the model
json_file = open("C:\Face_motion_detectioon\emotiondectector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:\Face_motion_detectioon\emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Set up webcam
webcam = cv2.VideoCapture(0)
# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read frame from webcam
    ret, im = webcam.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y + h, x:x + w]

        # Draw a black rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Resize and preprocess the face for prediction
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)

        # Predict emotion
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        # Add emotion label to the frame with black text
        cv2.putText(im, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the output
    cv2.imshow("Emotion Detection", im)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
