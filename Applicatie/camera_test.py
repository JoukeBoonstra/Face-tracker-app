import cv2
from deepface import DeepFace
from ultralytics import YOLO

yolo = YOLO("yolov8m.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Kan de camera niet openen.")
    exit()

frame_count = 0
last_results = []

gender_map = {
    "Man": "Man",
    "Male": "Man",
    "man": "Man",
    "Female": "Vrouw",
    "female": "Vrouw",
    "Vrouw": "Vrouw"
}

emotion_map = {
    "happy": "Blij",
    "sad": "Verdrietig",
    "angry": "Boos",
    "surprise": "Verrast",
    "fear": "Bang",
    "disgust": "Walging",
    "neutral": "Neutraal"
}

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        print("Kan geen frame lezen van de camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))

    current_results = []

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]

            if frame_count % 15 == 0:
                try:
                    small_face = cv2.resize(face, (224, 224))

                    result = DeepFace.analyze(
                        small_face,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False,
                        detector_backend="opencv"
                    )

                    age = int(result[0]['age'])
                    gender = gender_map.get(result[0]['dominant_gender'], "Onbekend")
                    emotion = emotion_map.get(result[0]['dominant_emotion'], "Onbekend")

                    current_results.append((x, y, w, h, gender, age, emotion))

                except Exception as e:
                    print("Analyse mislukt:", e)
                    current_results.append((x, y, w, h, None, None, None))
            else:
                if i < len(last_results):
                    current_results.append(last_results[i])
                else:
                    current_results.append((x, y, w, h, None, None, None))

        for res in current_results:
            x, y, w, h, gender, age, emotion = res

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if gender and age and emotion:
                cv2.putText(frame, f"{gender}, {age} jaar, {emotion}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if emotion == "Blij":
                    line1 = "Je bent geschikt voor de opleiding Mediavormgever"
                elif emotion == "Verdrietig":
                    line1 = "Je bent geschikt voor de opleiding Software Developer"
                elif emotion == "Boos":
                    line1 = "Je bent geschikt voor de opleiding Systeembeheer"
                elif emotion == "Verrast":
                    line1 = "Je bent geschikt voor de opleiding Mediavormgever"
                elif emotion == "Bang":
                    line1 = "Je bent geschikt voor de opleiding Mediavormgever"
                elif emotion == "Walging":
                    line1 = "Je bent geschikt voor de opleiding Mediavormgever"
                else:
                    line1 = "Je bent geschikt voor de opleiding Software Developer"

                line2 = "Dit resultaat is niet serieus, maar wel een leuke feature"

                base_x = x
                base_y = y + h + 40

                cv2.putText(frame, line1, (base_x, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(frame, line2, (base_x, base_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        last_results = current_results

    else:
        results = yolo(frame, stream=True)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, "Druk op Q om het programma af te sluiten", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Face + Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
