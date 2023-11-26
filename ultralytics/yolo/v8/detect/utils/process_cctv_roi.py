import cv2
import numpy as np
import sys
import ast

if __name__ == '__main__':
    # get video source path, destination path, src points from argv
    if len(sys.argv) != 4:
        print('Usage: python road_cctv_roi.py [video_src_path] [video_dst_path] [src_points]')
        sys.exit(1)

    VIDEO_SRC_PATH = sys.argv[1]
    VIDEO_DST_PATH = sys.argv[2]
    SRC_POINTS = ast.literal_eval(sys.argv[3])

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_SRC_PATH)

    # Get video information
    FPS = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_DST_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # fill black background except for roi
        mask = np.zeros_like(frame)
        roi_corners = [np.array(SRC_POINTS, dtype=np.int32)]
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        masked_image = cv2.bitwise_and(frame, mask)

        # Save video
        out.write(masked_image)

# Release everything if job is finished
cap.release()
out.release()