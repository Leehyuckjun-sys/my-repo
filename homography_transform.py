import cv2
import numpy as np

def px_to_flat_coords(yolo_df, anchor_markers):
    src_pts = np.array([am['px'] for am in anchor_markers])
    dst_pts = np.array([am['floor'] for am in anchor_markers])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    px_coords = yolo_df[['px_x','px_y']].values
    px_coords_hom = cv2.perspectiveTransform(px_coords.reshape(-1,1,2), H).reshape(-1,2)
    yolo_df['flat_x'] = px_coords_hom[:,0]
    yolo_df['flat_y'] = px_coords_hom[:,1]
    return yolo_df
