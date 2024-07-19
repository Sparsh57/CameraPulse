import json
import random
from threading import Thread, local
import cv2
import numpy as np
import mimetypes
from ultralytics import YOLO
import os
import time
import datetime
import sqlite3
import signal
import sys

# SQL query for inserting data into the database
insert_query = """
INSERT INTO Maindb ("companyCode", "exhibitionCode", "bootcode", "alertType", "filePath", "mimeType", "createdAt", "status") VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


class DatabaseHandler:
    """
    Handles database operations such as connection, queries, and transactions.
    """

    def __init__(self, db_name='maindatabase.db'):
        self.db_name = db_name
        self.local_storage = local()

    def get_connection(self):
        """
        Gets or creates a database connection for the current thread.
        """
        if not hasattr(self.local_storage, 'conn'):
            self.local_storage.conn = self.connect_database(self.db_name)
        return self.local_storage.conn

    def connect_database(self, db_name):
        """
        Connects to the specified SQLite database.
        """
        try:
            return sqlite3.connect(db_name)
        except Exception as e:
            print("Error. Database doesn't exist:", e)
            raise

    def execute_query(self, query, params=()):
        """
        Executes a given SQL query with optional parameters.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            print(f"Database operation error: {e}")
            conn.rollback()
            return None
        finally:
            if cursor:
                cursor.close()

    def fetch_all(self, query, params=()):
        """
        Executes a SQL query and fetches all results.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Database operation error: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

    def update_status(self, ftp_location, row_id):
        """
        Updates the status and file path of a database entry.
        """
        update_query = "UPDATE Maindb SET filePath = ?, status = 'Y' WHERE rowid = ?"
        self.execute_query(update_query, (ftp_location, row_id))

    def delete_entry(self, file_path):
        """
        Deletes a database entry based on the file path.
        """
        delete_query = "DELETE FROM Maindb WHERE filePath = ?"
        self.execute_query(delete_query, (file_path,))

    def insert_data(self, data):
        """
        Inserts data into the database.
        """
        self.execute_query(insert_query, data)


# Global instance of DatabaseHandler
global db_handler
db_handler = DatabaseHandler()

# Initialize global variables
id_defined = False
SERIAL_ID = None
roi_defined = False
roi_points = []
phone_roi_defined = False
phone_roi_points = []
paused = False
person_not_detected_start_time = None
intrusion_detected_start_time = None
phone_detected_start_time = None
temp_images = None
previous_phone_detected = time.time() - 300
previous_intrusion_detected = time.time() - 300
previous_tampering_detected = time.time() - 1700


def load_model(yolo_weights='yolov8s_ncnn_model', mobile_weights='yolov8s_ncnn_model'):
    """
    Loads YOLO models for person and phone detection.
    """
    return YOLO(yolo_weights), YOLO(mobile_weights)


def init_video_stream(rtsp_url):
    """
    Initializes a video stream from the given RTSP URL.
    """
    return VideoStream(rtsp_url)


class VideoStream:
    """
    Handles video streaming from a given source.
    """

    def __init__(self, src=""):
        self.src = src
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.reconnect_attempts = 0
        self.update_thread = Thread(target=self.update, args=())
        self.update_thread.start()

    def update(self):
        """
        Continuously reads frames from the video stream.
        """
        while True:
            if self.stopped:
                return
            grabbed, frame = self.stream.read()
            self.grabbed = grabbed
            self.frame = frame
            if not grabbed:
                self.handle_reconnect()

    def read(self):
        """
        Returns the current frame from the video stream.
        """
        frame = self.frame.copy() if self.frame is not None else None
        return self.grabbed, frame

    def stop(self):
        """
        Stops the video stream.
        """
        self.stopped = True
        self.stream.release()

    def handle_reconnect(self):
        """
        Handles reconnection attempts to the video stream.
        """
        self.stream.release()
        self.reconnect_attempts += 1
        print("Reconnecting...")
        time.sleep(5)
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.grabbed, self.frame = self.stream.read()


def load_configuration(config_file='configurations.json'):
    """
    Loads configuration settings from a JSON file.
    """
    global id_defined, SERIAL_ID, roi_defined, roi_points, phone_roi_defined, phone_roi_points, conf_phone, rtsp_link, tampering_index, no_person_time, phone_time, close_time, open_time, show_frame, temp_images
    try:
        with open(config_file) as json_file:
            data = json.load(json_file)
            if data["UNIQUE_NUMBER"] is not None:
                id_defined = True
                SERIAL_ID = data["UNIQUE_NUMBER"]
            if data["MAIN_ROI"] is not None:
                roi_defined = True
                roi_points = [(int(point[0]), int(point[1])) for point in data["MAIN_ROI"]]
            if data["PHONE_ROI"] is not None:
                phone_roi_defined = True
                phone_roi_points = [(int(point[0]), int(point[1])) for point in data["PHONE_ROI"]]
            if data["PHONE_CONFIDENCE"] is not None:
                conf_phone = data["PHONE_CONFIDENCE"]
            if data["RTSP_LINK"] is not None:
                rtsp_link = data["RTSP_LINK"]
            if data["TAMPERING_INDEX"] is not None:
                tampering_index = data["TAMPERING_INDEX"]
            if data["NO_PERSON_TIME"] is not None:
                no_person_time = data["NO_PERSON_TIME"]
            if data["PHONE_TIME"] is not None:
                phone_time = data["PHONE_TIME"]
            if data['CLOSE_TIME'] is not None:
                close_time = data['CLOSE_TIME']
            if data["OPEN_TIME"] is not None:
                open_time = data["OPEN_TIME"]
            if data["SHOW_FRAME"] is not None:
                show_frame = data["SHOW_FRAME"]
            if data.get("TAMPERING_TEMP_IMAGES") is not None:
                temp_images = [np.array(img) for img in data.get("TAMPERING_TEMP_IMAGES")]
    except Exception as e:
        print(f"Error loading configuration: {e}")


def save_configuration(config_file='configurations.json'):
    """
    Saves configuration settings to a JSON file.
    """
    global SERIAL_ID, roi_points, phone_roi_points, conf_phone, rtsp_link, tampering_index, no_person_time, phone_time, close_time, open_time, show_frame, temp_images
    try:
        with open(config_file, 'r') as file:
            data = json.load(file)
        data['UNIQUE_NUMBER'] = SERIAL_ID
        data['MAIN_ROI'] = roi_points
        data['PHONE_ROI'] = phone_roi_points
        data['RTSP_LINK'] = rtsp_link
        data['TAMPERING_INDEX'] = tampering_index
        data["PHONE_CONFIDENCE"] = conf_phone
        data["NO_PERSON_TIME"] = no_person_time
        data["PHONE_TIME"] = phone_time
        data["CLOSE_TIME"] = close_time
        data["OPEN_TIME"] = open_time
        data["SHOW_FRAME"] = show_frame
        data["TAMPERING_TEMP_IMAGES"] = [img.tolist() for img in temp_images] if temp_images is not None else None
        with open(config_file, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving configuration: {e}")


def box_text(frame, text_position, text, font_scale=1, font_thickness=2):
    """
    Draws a text box on the frame at the specified position.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)
    box_padding = 10

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    box_width = text_width + 2 * box_padding
    box_height = text_height + 2 * box_padding

    box_top_left = (text_position[0], text_position[1] - text_height - box_padding)
    box_bottom_right = (text_position[0] + box_width, text_position[1] + box_height)

    cv2.rectangle(frame, box_top_left, box_bottom_right, box_color, cv2.FILLED)
    cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness)


def manage_folder(folder):
    """
    Manages the number of images in the specified folder by deleting the oldest if over 100.
    """
    images = sorted([os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.png')])
    if len(images) > 100:
        oldest_image = images[0]
        os.remove(oldest_image)
        db_handler.delete_entry(oldest_image)


def get_user_id(frame):
    """
    Prompts the user to enter their ID via the video stream.
    """
    user_id = ""
    input_prompt = "Please enter your ID:"

    while True:
        input_box = frame.copy()
        box_text(input_box, (100, 100), input_prompt)
        box_text(input_box, (100, 150), user_id)
        if show_frame:
            cv2.imshow("Video", input_box)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key
            break
        elif key in [8, 127]:  # Backspace key
            user_id = user_id[:-1]
        elif key != 255:
            print(key)
            user_id += chr(key)

    return user_id


def select_roi(event, x, y, flags, param):
    """
    Handles mouse events to select regions of interest (ROI) for detection.
    """
    global roi_points, roi_defined, phone_roi_points, phone_roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_defined:
            roi_points.append((x, y))
            if len(roi_points) == 4:
                roi_defined = True
                print("ROI selected:", roi_points)
                save_configuration()
        elif not phone_roi_defined:
            phone_roi_points.append((x, y))
            if len(phone_roi_points) == 4:
                phone_roi_defined = True
                print("Phone ROI selected:", phone_roi_points)
                save_configuration()


def get_templates(image, points, n):
    """
    Extracts template images from the specified points in the image.
    """
    template_images = []
    for (x, y) in points:
        template_images.append(image[y:y + n, x:x + n])
    return template_images


def match(temp, frame_templates):
    """
    Matches templates with frame templates and returns the count of matches.
    """
    count = 0
    for (img0, img1) in zip(temp, frame_templates):
        res = cv2.matchTemplate(img0, img1, cv2.TM_CCOEFF_NORMED)
        if res[0][0] > 0.8:
            count += 1
    return count


def get_random_boxes(templates_size, image):
    """
    Generates random points for template extraction.
    """
    points = []
    h = image.shape[0]
    w = image.shape[1]
    i = list(range(0, h, 20))
    j = list(range(0, w, 20))
    while len(points) < templates_size:
        x = random.choice(j)
        y = random.choice(i)
        if (x, y) in points:
            continue
        else:
            points.append((x, y))
    return points


def main():
    """
    Main function that initializes and runs the video surveillance system.
    """
    global id_defined, roi_defined, phone_roi_defined, SERIAL_ID, roi_points, phone_roi_points, paused, frame_count, camera_tampered, camera_tampered_start_time, conf_phone, rtsp_link, tampering_index, points, temp_images, close_time, open_time, no_person_time, phone_time, vs, show_frame, intrusion_detected_start_time, previous_tampering_detected

    # Load YOLO models for person and phone detection
    model, model2 = load_model()
    load_configuration()

    # Initialize the video stream
    vs = init_video_stream(rtsp_link)
    if show_frame:
        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', select_roi)

    def signal_handler(sig, frame):
        """
        Handles termination signals to clean up resources.
        """
        global thread_stop, vs, db_handler
        thread_stop = True
        if vs:
            vs.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

    paused = False
    person_not_detected_start_time = None
    phone_detected_start_time = None
    intrusion_detected_start_time = None
    camera_tampered_start_time = None
    frame_interval = 1  # Process every 1st frame
    frame_count = 0

    while True:
        try:
            if not paused:
                ret, frame = vs.read()
                frame_count += 1

                if frame_count % frame_interval != 0:
                    continue

                if not ret or frame is None:
                    print("Failed to grab frame. Reconnecting...")
                    vs.stop()
                    time.sleep(1)
                    vs = init_video_stream(rtsp_link)
                    continue
                display_frame = frame.copy()
                if conf_phone is None or rtsp_link is None or tampering_index is None or no_person_time is None or phone_time is None or close_time is None or open_time is None:
                    print('PLEASE EDIT THE CONFIGURATION FILE')
                elif not id_defined:
                    SERIAL_ID = get_user_id(display_frame)
                    id_defined = True
                    save_configuration()
                elif not roi_defined:
                    box_text(display_frame, (120, 100), "Please select the four points of the polygon to detect person")
                    for i, point in enumerate(roi_points):
                        cv2.circle(display_frame, point, 10, (0, 255, 0), -1)
                        if i > 0:
                            cv2.line(display_frame, roi_points[i - 1], point, (0, 255, 0), 2)
                elif not phone_roi_defined:
                    box_text(display_frame, (120, 100), "Please select the four points of the polygon to detect phone")
                    for i, point in enumerate(phone_roi_points):
                        cv2.circle(display_frame, point, 10, (0, 255, 0), -1)
                        if i > 0:
                            cv2.line(display_frame, phone_roi_points[i - 1], point, (255, 0, 0), 2)
                else:
                    # Get the current hour
                    current_hour = datetime.datetime.now().hour
                    if current_hour >= open_time[0] or current_hour < open_time[1]:
                        process_frame(model, model2, frame, display_frame, SERIAL_ID)
                        if not temp_images:
                            points = get_random_boxes(500, frame)
                            temp_images = get_templates(frame, points, 20)
                            save_configuration()
                        frame_template = get_templates(frame, points, 20)
                        index = match(frame_template, temp_images) / 2
                        if index < tampering_index:
                            camera_tampered = True
                            print("Tampered")
                        else:
                            camera_tampered = False
                        handle_tamper(frame)

                if show_frame:
                    cv2.imshow('Video', display_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    paused = True
                    print("Stream paused. Press 's' to resume.")
                elif key == ord('s'):
                    paused = False
                    print("Stream resumed.")
                elif key == ord('r'):
                    reset_configuration()
                    load_configuration()
                    SERIAL_ID = get_user_id(display_frame)
                    id_defined = True
                    save_configuration()
                    roi_defined = False
                    phone_roi_defined = False

        except Exception as e:
            print(f"An error occurred: {e}")

    vs.stop()
    cv2.destroyAllWindows()


def process_frame(model, model2, frame, display_frame, SERIAL_ID):
    """
    Processes the frame to detect both persons and phones.
    """
    global person_not_detected_start_time, phone_detected_start_time, phone_roi_points, roi_points

    mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
    person_mask = mask.copy()
    phone_mask = mask.copy()
    roi_corners = np.array([roi_points], dtype=np.int32)
    phone_roi_corners = np.array([phone_roi_points], dtype=np.int32)
    cv2.fillPoly(person_mask, roi_corners, 255)
    cv2.fillPoly(phone_mask, phone_roi_corners, 255)

    person_roi_frame = cv2.bitwise_and(display_frame, display_frame, mask=person_mask)
    phone_roi_frame = cv2.bitwise_and(display_frame, display_frame, mask=phone_mask)

    cv2.polylines(display_frame, [roi_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(display_frame, [phone_roi_corners], isClosed=True, color=(255, 0, 0), thickness=2)

    person_detected = [False]
    phone_detected = [False]
    person_thread = Thread(target=detect_person, args=(model, person_roi_frame, person_detected))
    phone_thread = Thread(target=detect_phone, args=(model2, phone_roi_frame, phone_detected))

    person_thread.start()
    phone_thread.start()
    person_thread.join()
    phone_thread.join()
    display_detections(display_frame, person_detected[0], phone_detected[0], roi_corners, phone_roi_corners)
    handle_detections(frame, SERIAL_ID, person_detected[0], phone_detected[0])


def detect_person(model, person_roi_frame, person_detected):
    """
    Detects persons in the given frame.
    """
    global thread_stop
    results_person = model.track(person_roi_frame, persist=True)
    if results_person[0].boxes is not None:
        class_ids_person = results_person[0].boxes.cls.int().cpu().tolist()
        person_class_id = 0
        person_detected[0] = person_class_id in class_ids_person


def detect_phone(model2, phone_roi_frame, phone_detected):
    """
    Detects phones in the given frame.
    """
    global conf_phone
    results_phone = model2.track(phone_roi_frame, persist=True, conf=conf_phone, classes=67)
    if results_phone[0].boxes is not None:
        class_ids_phone = results_phone[0].boxes.cls.int().cpu().tolist()
        phone_class_id = 67
        phone_detected[0] = phone_class_id in class_ids_phone
        print(results_phone[0].boxes.conf)


def display_detections(display_frame, person_detected, phone_detected, roi_corners, phone_roi_corners):
    """
    Displays detections for both persons and phones.
    """
    global roi_points, phone_roi_points
    color = (0, 255, 0)
    cv2.polylines(display_frame, [roi_corners], isClosed=True, color=color, thickness=2)
    cv2.polylines(display_frame, [phone_roi_corners], isClosed=True, color=color, thickness=2)

    label_lines = ['Person Detected' if person_detected else 'No Person']
    for i, line in enumerate(label_lines):
        y_pos = roi_points[0][1] - 10 - i * 20
        cv2.putText(display_frame, line, (roi_points[0][0], y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    label_lines = ['Phone Detected' if phone_detected else 'No Phone']
    for i, line in enumerate(label_lines):
        y_pos = phone_roi_points[0][1] - 10 - i * 20
        cv2.putText(display_frame, line, (phone_roi_points[0][0], y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def handle_detections(frame, SERIAL_ID, person_detected, phone_detected):
    """
    Handles detections for both persons and phones.
    """
    global person_not_detected_start_time, phone_detected_start_time, previous_phone_detected, no_person_time, phone_time
    if time.time() - previous_phone_detected > 300:
        if phone_detected:
            if phone_detected_start_time is None:
                phone_detected_start_time = time.time()
            elif time.time() - phone_detected_start_time >= phone_time:
                print(f"Phone detected for {phone_time} seconds, capturing screenshot...")
                capture_screenshot(frame, "Phone_Images", "phone_detected", 2, SERIAL_ID)
                phone_detected_start_time = None
                previous_phone_detected = time.time()
        else:
            phone_detected_start_time = None

    if person_detected:
        person_not_detected_start_time = None
    else:
        if person_not_detected_start_time is None:
            person_not_detected_start_time = time.time()
        elif time.time() - person_not_detected_start_time > no_person_time:
            print(f"No Person detected for {no_person_time} seconds, capturing screenshot...")
            capture_screenshot(frame, "Person_Images", "staff_absent", 3, SERIAL_ID)
            person_not_detected_start_time = None


def capture_screenshot(frame, folder, prefix, status, SERIAL_ID):
    """
    Captures a screenshot and saves it to the specified folder.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    location = os.path.join(os.getcwd(), folder, f"{prefix}_{timestamp}.png")
    box_text(frame, (400, 100), prefix)
    cv2.imwrite(location, frame)
    datetimes = datetime.datetime.now()
    file_extension = str(mimetypes.guess_type(location)[0])
    data = ("dvi", "dvi", "dviKamlaNagar", status, location, file_extension, datetimes, "N")
    db_handler.insert_data(data)
    manage_folder(folder)


def reset_configuration(config_file='configurations.json'):
    """
    Resets the configuration to its default state.
    """
    global roi_points, roi_defined, phone_roi_points, phone_roi_defined, person_not_detected_start_time, intrusion_detected_start_time, id_defined, previous_phone_detected, conf_phone, rtsp_link, tampering_index, no_person_time, phone_time, close_time, open_time
    roi_defined = False
    phone_roi_defined = False
    phone_roi_points = []
    roi_points = []
    person_not_detected_start_time = None
    intrusion_detected_start_time = None
    previous_phone_detected = time.time() - 300
    previous_intrusion_detected = time.time() - 300
    previous_tampering_detected = time.time() - 1800
    id_defined = False
    data = {"UNIQUE_NUMBER": None, 'MAIN_ROI': None, 'PHONE_ROI': None, "PHONE_CONFIDENCE": conf_phone,
            "RTSP_LINK": rtsp_link, "TAMPERING_INDEX": tampering_index, "NO_PERSON_TIME": no_person_time,
            "PHONE_TIME": phone_time, "CLOSE_TIME": close_time, "OPEN_TIME": open_time, "SHOW_FRAME": True,
            "TAMPERING_TEMP_IMAGES": None}
    with open(config_file, 'w') as file:
        json.dump(data, file, indent=4)
    print("Select the ROI by clicking on 4 points.")


def handle_tamper(frame):
    """
    Handles camera tampering detection.
    """
    global camera_tampered, camera_tampered_start_time, points, temp_images, previous_tampering_detected
    if time.time() - previous_tampering_detected > 1800:
        if camera_tampered:
            if camera_tampered_start_time is None:
                camera_tampered_start_time = time.time()
            elif time.time() - camera_tampered_start_time > 2:
                print("Camera Tampering detected for 2 seconds, capturing screenshot...")
                capture_screenshot(frame, "Tampering_Images", "Camera_tampered", 4, SERIAL_ID)
                camera_tampered_start_time = None
                # points = get_random_boxes(500, frame)
                # temp_images = get_templates(frame, points, 20)
                previous_tampering_detected = time.time()
        else:
            camera_tampered_start_time = None


if __name__ == '__main__':
    main()
