import datetime
import cv2
import pandas as pd
import numpy as np

VIDEO_PATH = "/home/hoya/road-cctv/vsl_231106_074901/road_1_6_sample.mp4"
POINT_PATH = "/home/hoya/road-cctv/vsl_231106_074901/road_1_6_sample_result.csv"
SRC_POINTS = [[465, 364], [649, 364], [506, 300], [376, 300]]
DST_POINTS = [[0, 0], [72, 0], [72, 160], [0, 160]]
ROI_WIDTH = 72
ROI_HEIGHT = 160
SHOW = True

def perspective_transform_tracking_data(
        tracking_data: pd.DataFrame,
        persp_m: np.ndarray,
        roi: tuple[int, int]) -> pd.DataFrame:
    
    df = tracking_data.copy()

    # perspective transform
    df['_src'] = list(zip(df['x'], df['y']))
    df['_dst'] = df['_src'].apply(
        lambda x: cv2.perspectiveTransform(np.array([[x]], dtype=np.float32), persp_m)[0][0])

    # filter out of range
    df['pos_x'] = df['_dst'].apply(lambda x: x[0])
    df['pos_y'] = df['_dst'].apply(lambda x: x[1])
    df = df.query(f'pos_x >= 0 and pos_x <= {roi[0]} and pos_y >= 0 and pos_y <= {roi[1]}')

    # drop temp columns
    df = df.drop(['_src', '_dst'], axis=1)
    
    return df

def remove_outlier_tracking_data(tracking_data: pd.DataFrame) -> pd.DataFrame:
    temp_df = tracking_data.sort_values(by=['id', 'frame'], ascending=[True, True])

    prev_id = -1
    prev_y_pos = -1
    valid_index = []

    for index, row in temp_df.iterrows():
        id = row['id']
        pos_y = row['pos_y']

        if id > prev_id: # new id
            prev_id = id
            prev_y_pos = pos_y
            valid_index.append(index)
            continue

        if pos_y >= prev_y_pos: # y position validation
            prev_y_pos = pos_y
            valid_index.append(index)

    return temp_df[temp_df.index.isin(valid_index)]

def interpolate_tracking_data(tracking_data: pd.DataFrame) -> pd.DataFrame:
    
    # create new dataframe
    df = pd.DataFrame()

    # interpolate missing frame
    for id in tracking_data['id'].unique():
        temp_df: pd.DataFrame = tracking_data[tracking_data['id'] == id]
        frames = temp_df['frame'].sort_values().values

        col_frame = np.arange(frames[0], frames[-1] + 1) # min, max + 1
        df = pd.concat([df, pd.DataFrame({ 'id': id, 'frame': col_frame })], ignore_index=True)

    # join tracking data
    df = df.join(tracking_data.set_index(['id', 'frame']), on=['id', 'frame'], how='left')

    # interpolate missing class with the same class
    df['class'] = df.groupby('id')['class'].ffill().bfill()

    # interpolate missing x, y with linear interpolation
    df['x'] = df.groupby('id')['x'].apply(lambda x: x.interpolate(method='linear')).reset_index(drop=True)
    df['y'] = df.groupby('id')['y'].apply(lambda x: x.interpolate(method='linear')).reset_index(drop=True)

    # interpolate missing pos_x, pox_y with linear interpolation
    df['pos_x'] = df.groupby('id')['pos_x'].apply(lambda x: x.interpolate(method='linear')).reset_index(drop=True)
    df['pos_y'] = df.groupby('id')['pos_y'].apply(lambda x: x.interpolate(method='linear')).reset_index(drop=True)

    return df

# timestamp logging
def print_timestamp(message):
    print(f'[{datetime.datetime.now()}] {message}')

if __name__ == '__main__':
    
    # calculate perspective transform matrix
    src_points = np.array(SRC_POINTS, dtype=np.float32)
    dst_points = np.array(DST_POINTS, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # preprocessing tracking data
    print_timestamp('read tracking data')
    tracking_data = pd.read_csv(POINT_PATH)
    print_timestamp('perspective transform tracking data')
    tracking_data = perspective_transform_tracking_data(tracking_data, matrix, (ROI_WIDTH, ROI_HEIGHT))
    print_timestamp('remove outlier tracking data')
    tracking_data = remove_outlier_tracking_data(tracking_data)
    print_timestamp('interpolate tracking data')
    tracking_data = interpolate_tracking_data(tracking_data)

    tracking_data.to_csv('road_transform_output.csv', index=False)

    capture = cv2.VideoCapture(VIDEO_PATH)
    fps = capture.get(cv2.CAP_PROP_FPS)

    video_size = (ROI_WIDTH, ROI_HEIGHT)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(f'road_transform_output.mp4', fourcc, fps, video_size)

    trail_history = {}
    frame_idx = 0

    while capture.isOpened():
        
        ret, frame = capture.read()

        if ret == False:
            break
        
        # points
        frame_points = tracking_data[tracking_data['frame'] == frame_idx]
        identities = frame_points['id'].values
        # print(identities)

        # remove tracked point from buffer if object is lost
        for key in list(trail_history.keys()):
            if key not in identities:
                trail_history.pop(key)

        # perspective transform
        persp = cv2.warpPerspective(frame, matrix, video_size)

        # draw trails
        for _, _row in frame_points.iterrows():
            _id = _row['id']
            _x = int(_row['x'])
            _y = int(_row['y'])
            _pos_x = int(_row['pos_x'])
            _pos_y = int(_row['pos_y'])

            trail: list = trail_history.get(_id, [])
            trail.append([(_x, _y), (_pos_x, _pos_y)])
            trail_history[_id] = trail

            # draw trail
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1][0], trail[i][0], (0, 255, 0), 1)
                cv2.line(persp, trail[i - 1][1], trail[i][1], (0, 255, 0), 1)

        persp = cv2.flip(persp, 0)
        writer.write(persp)
        frame_idx += 1

        if (SHOW):
            cv2.imshow('frame', frame)
            cv2.imshow('persp', persp)
            cv2.waitKey(33)


    capture.release()
    writer.release()
    cv2.destroyAllWindows()