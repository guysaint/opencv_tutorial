# yolo_ball_only.py
from ultralytics import YOLO
import cv2
from collections import defaultdict

VIDEO_PATH = "../img/son.mp4"
MODEL_PATH = "yolo11s.pt"   # 필요시 yolo11m.pt로 교체
OUT_PATH = "son_ball_out.mp4"

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

    # 통계
    total_frames = 0
    frames_with_ball = 0
    total_ball_detections = 0

    # ByteTrack으로 트랙 ID 카운트(중복 제거용)
    seen_track_ids = set()

    # track 모드: classes=[32] => sports ball only
    # conf는 살짝 낮추고, imgsz 키워서 공 검출률↑
    tracker_cfg = "bytetrack.yaml"
    for result in model.track(
        source=VIDEO_PATH,
        classes=[32],
        conf=0.25,
        iou=0.6,
        imgsz=960,
        tracker=tracker_cfg,
        stream=True,
        verbose=False
    ):
        total_frames += 1

        frame = result.orig_img.copy()
        has_ball_this_frame = False

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0].item()) if box.cls is not None else -1
                if cls != 32:
                    continue

                # 통계
                has_ball_this_frame = True
                total_ball_detections += 1

                # 트랙 ID 추적(있을 경우에만)
                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id[0].item())
                    seen_track_ids.add(track_id)

                # 시각화
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "ball" if track_id is None else f"ball #{track_id}"
                cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if has_ball_this_frame:
            frames_with_ball += 1

        # 오버레이 정보
        overlay = f"Frames: {total_frames} | Frames with ball: {frames_with_ball} | Dets: {total_ball_detections} | Unique IDs: {len(seen_track_ids)}"
        cv2.putText(frame, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    print("=== Summary ===")
    print(f"Total frames: {total_frames}")
    print(f"Frames with ball: {frames_with_ball}")
    print(f"Total ball detections: {total_ball_detections}")
    print(f"Unique tracked balls (IDs): {len(seen_track_ids)}")

if __name__ == "__main__":
    main()