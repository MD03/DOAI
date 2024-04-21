!pip install opencv-python
import cv2
from IPython.display import Image, display

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained Haar cascade classifier for full body detection
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Load the image
image_path = 'people.jpg'  # Modify this path to the location of your uploaded image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Fine-tune parameters here

# Detect full bodies in the image
full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Fine-tune parameters here

# Count the total number of people
total_people = len(faces) + len(full_bodies)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Draw rectangles around the detected full bodies
for (x, y, w, h) in full_bodies:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the image with rectangles around the detected faces and full bodies
cv2.imwrite('output_image.jpg', image)

# Display the image with rectangles around the detected faces and full bodies
display(Image(filename='output_image.jpg'))

# Print the total number of people
print("Total number of people:", total_people)
