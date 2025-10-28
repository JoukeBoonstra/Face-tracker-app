import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kan de camera niet openen.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan geen frame lezen van de camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            
            age = result[0]['age']
            gender = result[0]['dominant_gender']
            emotion = result[0]['dominant_emotion']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{gender}, {age} jaar", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception as e:
            print("Analyse mislukt:", e)

    cv2.imshow('Face Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
