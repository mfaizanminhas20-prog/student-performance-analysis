import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"       # folder with student photos
ATTENDANCE_DIR  = "attendance_records"
CONFIDENCE      = 0.5                 # lower = stricter matching

# ─────────────────────────────────────────
#  STEP 1 — Load known faces from folder
# ─────────────────────────────────────────
def load_known_faces():
    known_encodings = []
    known_names     = []

    print("[INFO] Loading known faces...")

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]   # filename = student name
            path = os.path.join(KNOWN_FACES_DIR, filename)

            image    = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"  ✅ Loaded: {name}")
            else:
                print(f"  ⚠️  No face found in: {filename}")

    print(f"[INFO] Total students loaded: {len(known_names)}\n")
    return known_encodings, known_names


# ─────────────────────────────────────────
#  STEP 2 — Mark attendance in Excel
# ─────────────────────────────────────────
def mark_attendance(name, attendance_log):
    if name not in attendance_log:
        now  = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        attendance_log[name] = {"Name": name, "Date": date, "Time": time, "Status": "Present"}
        print(f"  📋 Marked Present: {name} at {time}")


# ─────────────────────────────────────────
#  STEP 3 — Save to Excel
# ─────────────────────────────────────────
def save_to_excel(attendance_log):
    if not attendance_log:
        print("[INFO] No attendance to save.")
        return

    date_str  = datetime.now().strftime("%Y-%m-%d")
    filename  = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.xlsx")

    df = pd.DataFrame(list(attendance_log.values()))
    df.to_excel(filename, index=False)
    print(f"\n[✅] Attendance saved to: {filename}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────
#  STEP 4 — Run Camera + Detection
# ─────────────────────────────────────────
def run_attendance_system():
    known_encodings, known_names = load_known_faces()

    if not known_encodings:
        print("[ERROR] No known faces loaded! Add student photos to 'known_faces/' folder.")
        return

    attendance_log = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found!")
        return

    print("[INFO] Camera started. Press 'Q' to quit and save attendance.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in current frame
        face_locations  = face_recognition.face_locations(rgb_frame)
        face_encodings  = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):

            matches     = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=CONFIDENCE)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match  = np.argmin(face_distances)

            name  = "Unknown"
            color = (0, 0, 255)  # Red for unknown

            if matches[best_match]:
                name  = known_names[best_match]
                color = (0, 255, 0)  # Green for known
                mark_attendance(name, attendance_log)

            # Scale back up face location
            top, right, bottom, left = [v * 4 for v in face_location]

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Show attendance count on screen
        count_text = f"Present: {len(attendance_log)}"
        cv2.putText(frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_to_excel(attendance_log)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_attendance_system()