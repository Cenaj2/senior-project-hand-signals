# Hand direction script for Pi Zero
# now with a simple calibration screen

import cv2
import numpy as np
import time

# arrow window
def draw_arrow_demo(direction, hold_time):

    img = np.zeros((340,480,3), dtype=np.uint8)

    cx = 240
    cy = 170

    arrow_color = (180,180,180)
    highlight = (0,255,255)

    # arrows (head at second point)
    cv2.arrowedLine(img, (240,130), (240,40), arrow_color, 10, tipLength=0.4)
    cv2.arrowedLine(img, (240,210), (240,300), arrow_color, 10, tipLength=0.4)
    cv2.arrowedLine(img, (160,170), (80,170), arrow_color, 10, tipLength=0.4)
    cv2.arrowedLine(img, (320,170), (400,170), arrow_color, 10, tipLength=0.4)

    cv2.circle(img, (cx,cy), 45, (255,255,255), -1)

    if direction == "UP":
        cv2.arrowedLine(img, (240,130), (240,40), highlight, 14, tipLength=0.4)
    if direction == "DOWN":
        cv2.arrowedLine(img, (240,210), (240,300), highlight, 14, tipLength=0.4)
    if direction == "LEFT":
        cv2.arrowedLine(img, (160,170), (80,170), highlight, 14, tipLength=0.4)
    if direction == "RIGHT":
        cv2.arrowedLine(img, (320,170), (400,170), highlight, 14, tipLength=0.4)
    if direction == "CENTER":
        cv2.circle(img, (cx,cy), 45, highlight, -1)

    ms = hold_time * 1000
    cv2.putText(img, f"Hold: {ms:.0f} ms", (10,330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return img


# simple calibration screen
def run_calibration(cap, backsub):
    while True:
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        msg = "Stand still - Press SPACE to calibrate"
        cv2.putText(frame, msg, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # feed some frames so background subtractor learns
            for i in range(20):
                ret2, f2 = cap.read()
                if not ret2: break
                f2 = cv2.flip(f2,1)
                backsub.apply(f2)
                cv2.waitKey(10)
            cv2.destroyWindow("Calibration")
            return

        if key == ord('q'):
            exit()


def main():

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    if not cap.isOpened():
        print("camera not found")
        return

    # background subtractor
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=25,
        detectShadows=False
    )

    # run calibration first
    run_calibration(cap, backsub)

    last_fps_time = time.time()
    frame_counter = 0
    current_fps = 0

    CENTER_R = 60
    MOVE_THRESH = 60
    HOLD_TIME_REQUIRED = 0.3

    detected_direction = "NONE"
    hold_start_time = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        cx = w//2
        cy = h//2

        gui = frame.copy()

        # mask
        fgmask = backsub.apply(frame)
        fgmask = cv2.erode(fgmask, np.ones((3,3),np.uint8), 1)
        fgmask = cv2.dilate(fgmask, np.ones((3,3),np.uint8), 2)

        contours,_ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_raw_dir = "NONE"

        if len(contours) > 0:
            big = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(big)

            if area > 600:
                x,y,bw,bh = cv2.boundingRect(big)
                hx = x + bw//2
                hy = y + bh//2

                cv2.circle(gui,(hx,hy),5,(0,0,255),-1)
                cv2.rectangle(gui,(x,y),(x+bw,y+bh),(0,255,0),2)

                dx = hx - cx
                dy = hy - cy

                if abs(dx) < CENTER_R and abs(dy) < CENTER_R:
                    current_raw_dir = "CENTER"
                else:
                    if abs(dx) > abs(dy):
                        if dx > MOVE_THRESH: current_raw_dir = "RIGHT"
                        if dx < -MOVE_THRESH: current_raw_dir = "LEFT"
                    else:
                        if dy > MOVE_THRESH: current_raw_dir = "DOWN"
                        if dy < -MOVE_THRESH: current_raw_dir = "UP"

        cv2.circle(gui, (cx,cy), CENTER_R, (255,255,255), 2)

        now = time.time()

        # hold logic
        if current_raw_dir != "NONE":
            if detected_direction != current_raw_dir:
                detected_direction = current_raw_dir
                hold_start_time = now
            hold_time = now - hold_start_time
        else:
            detected_direction = "NONE"
            hold_start_time = None
            hold_time = 0

        if hold_start_time and hold_time >= HOLD_TIME_REQUIRED:
            active_dir = detected_direction
        else:
            active_dir = "NONE"

        # fps
        frame_counter += 1
        if now - last_fps_time >= 1:
            current_fps = frame_counter / (now - last_fps_time)
            last_fps_time = now
            frame_counter = 0

        cv2.putText(gui, f"Raw: {current_raw_dir}", (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
        cv2.putText(gui, f"Active: {active_dir}", (5,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        cv2.putText(gui, f"FPS: {current_fps:.1f}", (5,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        demo = draw_arrow_demo(active_dir, hold_time)

        cv2.imshow("Camera View", gui)
        cv2.imshow("Motion Mask", fgmask)
        cv2.imshow("Arrow Demo", demo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
