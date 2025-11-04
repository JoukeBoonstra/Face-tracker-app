import cv2
from deepface import DeepFace
from ultralytics import YOLO

# YOLO model laden (klein en snel)
yolo = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Kan de camera niet openen.")
    exit()

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        print("Kan geen frame lezen van de camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))

    current_results = []

    # ✅ Eerst: gezicht detecteren en analyseren
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
                    gender = result[0]['dominant_gender']
                    emotion = result[0]['dominant_emotion']

                    current_results.append((x, y, w, h, gender, age, emotion))

                except Exception as e:
                    print("Analyse mislukt:", e)
                    current_results.append((x, y, w, h, None, None, None))
            else:
                if i < len(last_results):
                    current_results.append(last_results[i])
                else:
                    current_results.append((x, y, w, h, None, None, None))

        # ✅ Tekenen van face + analyses
        for res in current_results:
            x, y, w, h, gender, age, emotion = res

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if gender and age and emotion:
                cv2.putText(frame, f"{gender}, {age} jaar, {emotion}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Tekst op basis van emotie 🧠
                if emotion == "happy":
                    reactie = "Wat ben je vandaag vrolijk!"
                elif emotion == "sad":
                    reactie = "Het is bijna weekend, nog even doorzetten"
                elif emotion == "angry":
                    reactie = "Ik begrijp het, het is maandagochtend"
                elif emotion == "surprise":
                    reactie = "Is het eerder weekend?"
                elif emotion == "fear":
                    reactie = "Is het later weekend?"
                elif emotion == "disgust":
                    reactie = "Vakantie is zeker voorbij?"
                else:
                    reactie = "Kin gebeure"

                cv2.putText(frame, reactie, (x, y + h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        last_results = current_results

    else:
        # ⚠️ Geen gezicht gevonden → YOLO objectdetectie uitvoeren
        results = yolo(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    cv2.imshow("Face + Object Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
