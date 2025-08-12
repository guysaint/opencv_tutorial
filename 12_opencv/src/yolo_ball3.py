# yolo_ball.py  (클래스 구조 유지 + 최소 수정)
from ultralytics import YOLO
import cv2

class TrafficMonitor:
    def __init__(self, model_path='yolo11s.pt'):
        # 모델 로드
        self.model = YOLO(model_path)
        # 공만 추출 (sports ball: 32)
        self.traffic_classes = {
            32: 'sports_ball'
        }
        # 통계
        self.stats = {
            'total_detections': 0,     # 박스 총 검출 수(중복 포함)
            'by_class': {},            # 클래스별 검출 수
            'total_frames': 0,         # 전체 프레임 수
            'frames_with_ball': 0,     # 공이 검출된 프레임 수
            'unique_ids': set()        # 트랙 ID 중복 제거용
        }

    def analyze_frame(self, result, frame):
        """
        track() 한 프레임의 result와 원본 frame을 받아서
        박스 그리기, 통계 업데이트를 수행.
        """
        has_ball = False

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0].item()) if box.cls is not None else -1
                if cls != 32:
                    continue

                has_ball = True
                self.stats['total_detections'] += 1
                class_name = self.traffic_classes.get(cls, str(cls))
                self.stats['by_class'][class_name] = self.stats['by_class'].get(class_name, 0) + 1

                # 트랙 ID 수집(있으면)
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0].item())
                    self.stats['unique_ids'].add(track_id)
                    label = f"ball #{track_id}"
                else:
                    label = "ball"

                # 시각화
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if has_ball:
            self.stats['frames_with_ball'] += 1

        # 오버레이 정보
        overlay = (
            f"Frames: {self.stats['total_frames']}  "
            f"FramesWithBall: {self.stats['frames_with_ball']}  "
            f"Dets: {self.stats['total_detections']}  "
            f"UniqueIDs: {len(self.stats['unique_ids'])}"
        )
        cv2.putText(frame, overlay, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return frame

    def run_live_monitoring(self):
        """
        (원래 구조 유지용 더미)
        필요하면 웹캠 로직으로 연결할 수 있지만, 이번 실습은 파일 재생을 권장.
        """
        raise NotImplementedError("이번 실습에서는 run_on_video()를 사용하세요.")

    def run_on_video(self, src_path="../img/son.mp4", out_path="son_ball_out.mp4"):
        """
        비디오 파일에서 공만(track) 추적하고, 결과를 비디오로 저장.
        """
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # track()로 스트리밍 처리
        for result in self.model.track(
            source=src_path,
            classes=[32],             # 공만
            conf=0.25,                # recall↑
            iou=0.6,                  # NMS 살짝 관대
            imgsz=960,                # 해상도 업
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
            device='cpu'              # M1 Pro지만 요청대로 CPU 고정
        ):
            self.stats['total_frames'] += 1
            frame = result.orig_img.copy()
            frame = self.analyze_frame(result, frame)
            writer.write(frame)

        writer.release()
        self.show_final_stats()
        print(f"Saved: {out_path}")

    def show_final_stats(self):
        print("\n=== Summary ===")
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Frames with ball: {self.stats['frames_with_ball']}")
        print(f"Total detections (boxes): {self.stats['total_detections']}")
        print(f"Unique tracked balls (IDs): {len(self.stats['unique_ids'])}")
        print("Detections by class:")
        for class_name, count in self.stats['by_class'].items():
            print(f"  {class_name}: {count}")

# 실행부
if __name__ == "__main__":
    monitor = TrafficMonitor()
    monitor.run_on_video(src_path="../img/son.mp4", out_path="son_ball_out.mp4")