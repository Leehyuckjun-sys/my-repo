import sync_and_resample
import homography_transform
import feature_engineering
import gpr_model
import online_correction

# 1. 데이터 불러오기 (YOLO, UWB, 앵커 등)
yolo_data, uwb_data, anchor_data = sync_and_resample.load_data('yolo.csv', 'uwb.csv', 'anchor.csv')

# 2. 시간 동기화 및 리샘플링 (공통 등간격 grid)
yolo_sync, uwb_sync = sync_and_resample.time_sync_and_resample(yolo_data, uwb_data)

# 3. YOLO 픽셀 → 평면 좌표 변환
yolo_flat = homography_transform.px_to_flat_coords(yolo_sync, anchor_data)

# 4. feature 및 라벨 생성
features, labels = feature_engineering.make_features_and_labels(yolo_flat, uwb_sync, anchor_data)

# 5. GPR 모델 학습 및 검증 (cross-validation 포함)
gpr_x, gpr_y, x_scaler, y_scaler, metrics = gpr_model.train_and_validate_gpr(features, labels)

# 6. 실시간 보정 및 융합 적용
results = online_correction.run_online(yolo_flat, uwb_sync, features, gpr_x, gpr_y, x_scaler, y_scaler)

print(results)