import cv2
import pandas as pd
import numpy as np

VIDEO_PATH = "/home/hoya/road-cctv/vsl_231106_074901/road_1_6_sample.mp4"
POINT_PATH = "/home/hoya/road-cctv/vsl_231106_074901/road_1_6_sample_result.csv"
SRC_POINTS = [[376, 300], [506, 300], [465, 364], [649, 364]]
DST_POINTS = [[0, 0], [72, 0], [0, 160], [72, 160]]
OUTPUT_WIDTH = 72
OUTPUT_HEIGHT = 160
SHOW = True

if __name__ == '__main__':
    
    src_points = np.array(SRC_POINTS, dtype=np.float32)
    dst_points = np.array(DST_POINTS, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    point_df = pd.read_csv(POINT_PATH)
    point_df['src'] = list(zip(point_df['x'], point_df['y']))
    print(point_df['src'])
    point_df['dst'] = point_df['src'].apply(lambda x: cv2.perspectiveTransform(np.array([[x]], dtype=np.float32), matrix)[0][0])
    point_df.to_csv('road_transform_output.csv', index=False)

    capture = cv2.VideoCapture(VIDEO_PATH)
    fps = capture.get(cv2.CAP_PROP_FPS)

    video_size = (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(f'road_transform_output.mp4', fourcc, fps, video_size)

    while capture.isOpened():
        
        ret, frame = capture.read()

        if ret == False:
            break
        
        persp = cv2.warpPerspective(frame, matrix, video_size)
        writer.write(persp)

        if (SHOW):
            cv2.imshow('frame', frame)
            cv2.imshow('persp', persp)
            
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()