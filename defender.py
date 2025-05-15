import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
import os
from datetime import datetime

def play_alarm():
    try:
        os.system("aplay alarm.wav &")
    except Exception as e:
        print("Alarm playback failed:", e)

def ir_pulse_simulation():
    print("[DEFENSE] Simulating IR pulse...")
    for _ in range(5):
        print("IR PULSE BLAST")
        time.sleep(0.2)

def log_to_skymind(event):
    timestamp = datetime.now().isoformat()
    with open("skymind_log.txt", "a") as f:
        f.write(f"{timestamp} - {event}\n")
    print(f"[SkyMind] Logged: {event}")

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("[INFO] Starting RealSense stream...")
    pipeline.start(config)
    time.sleep(2)

    motion_detected = False
    last_frame = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if last_frame is None:
                last_frame = gray
                continue

            delta = cv2.absdiff(last_frame, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    if not motion_detected:
                        print("[ALERT] Motion detected!")
                        threading.Thread(target=play_alarm).start()
                        threading.Thread(target=ir_pulse_simulation).start()
                        log_to_skymind("Motion detected - defense triggered")
                        motion_detected = True

            last_frame = gray

            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("OSIRIS Defender (Pi)", color_image)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
