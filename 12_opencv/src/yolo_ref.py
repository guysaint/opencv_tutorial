# yolo_referee.py  (네 클래스 구조 유지, referee only tracking)
from ultralytics import YOLO
import cv2
import numpy as np

def in_range(x, lo, hi):
    return lo <= x <= hi
def _center_of(self, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def _crop_upper_roi(self, frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h = max(1, y2 - y1)
    y_top = y1 + max(int(h * 0.05), 1)
    y_end = y1 + max(int(h * 0.40), 2)      # 상단 35~40% 정도
    x1 = max(0, x1); y_top = max(0, y_top)
    x2 = min(frame.shape[1]-1, x2); y_end = min(frame.shape[0]-1, y_end)
    if x2 <= x1 or y_end <= y_top:
        return None
    return frame[y_top:y_end, x1:x2]

def _compute_hist(self, roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # H,S 2D 히스토그램 (H:30 bins, S:32 bins) → 정규화 후 1D 벡터화
    hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180, 0,256])
    hist = cv2.normalize(hist, hist).flatten()
    # L2 정규화(코사인 유사도 계산 용이)
    norm = np.linalg.norm(hist) + 1e-6
    return hist / norm

def _sim_hist(self, h1, h2):
    # 코사인 유사도
    return float(np.dot(h1, h2))

def _is_redwhite(self, roi):
    """ 빨간 상의/하의, 하얀색(밝고 채도낮음) 비중이 높은지 검사 → True면 배제 """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # red: H∈[0,10] ∪ [170,179], S>80, V>80
    red1 = (H <= 10) & (S > 80) & (V > 80)
    red2 = (H >= 170) & (S > 80) & (V > 80)
    red_mask = red1 | red2

    # white: V>200 & S<30 (밝고 거의 무채색)
    white_mask = (V > 200) & (S < 30)

    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return False
    red_ratio = float(red_mask.sum()) / total
    white_ratio = float(white_mask.sum()) / total

    # 경기 영상에 맞춰 0.20~0.35 사이 조정
    return (red_ratio > 0.25) or (white_ratio > 0.35)

def _near_last(self, center, frame_shape):
    if self.last_center is None:
        return True
    (x, y) = center
    (lx, ly) = self.last_center
    h, w = frame_shape[:2]
    diag = (w**2 + h**2) ** 0.5
    dist = ((x - lx)**2 + (y - ly)**2) ** 0.5
    return (dist / diag) <= self.reacquire_dist

class TrafficMonitor:
    def __init__(self, model_path='yolo11s.pt', target_color='black', manual_pick=False):
        self.model = YOLO(model_path)
        self.traffic_classes = {0: 'person'}
        self.stats = {'total_frames': 0, 'frames_with_ref': 0}

        self.target_color = target_color
        self.manual_pick = manual_pick
        self.locked_track_id = None
        self.click_xy = None

        # 새로 추가: 잠금 유지/재획득 관련
        self.ref_hist = None          # 잠금 시 상의 HSV 히스토그램 템플릿
        self.last_center = None       # 마지막 박스 중심 (x,y)
        self.miss_streak = 0
        self.max_miss = 20            # 이 프레임 수까지는 재획득 시도하고, 다른 사람으로 갈아타지 않음
        self.sim_thresh = 0.85        # 히스토그램 코사인 유사도 임계
        self.reacquire_dist = 0.15    # 프레임 대각선의 15% 이내면 근처로 간주
        self.negative_redwhite = True # 빨강/백 유니폼 배제

    # ---------------- 핵심: 심판 후보 색상 필터 ----------------
    def looks_like_referee(self, frame, box_xyxy):
        x1, y1, x2, y2 = map(int, box_xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        # 상의 중심부만 ROI (가로 40%, 세로 35%)
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        roi_w = max(int(bw * 0.40), 4)
        roi_h = max(int(bh * 0.35), 4)
        rx1 = max(cx - roi_w // 2, x1)
        rx2 = min(cx + roi_w // 2, x2)
        ry1 = max(y1 + int(bh * 0.05), y1)     # 목/하이라이트 살짝 피함
        ry2 = min(ry1 + roi_h, y2)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 1) 잔디(초록) 제외: H 35~85 범위 제거
        green_mask = (H >= 35) & (H <= 85)

        # 2) "어두운 검정" 후보: V<70, S>30 (너무 회색/하양/하이라이트 제외)
        dark_mask = (V < 60) & (S > 30)

        valid = (~green_mask)
        dark_valid = dark_mask & valid

        total = int(valid.sum())
        if total < 50:  # ROI가 너무 작거나 유효 픽셀이 거의 없으면 패스
            return False

        ratio = float(dark_valid.sum()) / float(total)

        # 임계치: 경기/조명에 따라 0.25~0.40 사이에서 조정
        return ratio >= 0.25

    # -------------- manual pick: 마우스 클릭해서 대상 고정 --------------
    def on_mouse(self, event, x, y, flags, param):
        if self.manual_pick and event == cv2.EVENT_LBUTTONDOWN:
            self.click_xy = (x, y)

    def analyze_frame(self, result, frame):
        """
        track() 결과에서 사람 박스 중 '심판'만 시각화 & 통계 업데이트
        """
        has_ref = False
        target_box = None
        target_track = None

        if result.boxes is not None and len(result.boxes) > 0:
            # 1) 이미 특정 트랙 ID가 잠겼으면 그 ID만 표시
            if self.locked_track_id is not None:
                
                for box in result.boxes:
                    cls = int(box.cls[0].item()) if box.cls is not None else -1
                    if cls != 0:
                        continue
                    if hasattr(box, 'id') and box.id is not None:
                        tid = int(box.id[0].item())
                        if tid == self.locked_track_id:
                            target_box = box
                            target_track = tid
                            break
            else:
                # 2) 잠기지 않았다면, manual_pick이면 클릭 근처 박스를 우선
                candidates = []
                for box in result.boxes:
                    cls = int(box.cls[0].item()) if box.cls is not None else -1
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # 수동 선택이라면 클릭점이 박스 안에 있나 확인
                    if self.manual_pick and self.click_xy is not None:
                        cx, cy = self.click_xy
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            candidates = [(box, 'manual')]
                            break
                    else:
                        # 자동 색상 필터
                        if self.looks_like_referee(frame, (x1, y1, x2, y2)):
                            candidates.append((box, 'color'))

                if candidates:
                    # 우선순위: manual > color
                    if any(tag == 'manual' for _, tag in candidates):
                        target_box, _ = next((b, t) for b,t in candidates if t == 'manual')
                    else:
                        target_box, _ = candidates[0]

                    if hasattr(target_box, 'id') and target_box.id is not None:
                        self.locked_track_id = int(target_box.id[0].item())
                        target_track = self.locked_track_id

        # 시각화 + 통계
        if target_box is not None:
            has_ref = True
            x1, y1, x2, y2 = map(int, target_box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = "Referee" if target_track is None else f"Referee #{target_track}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if has_ref:
            self.stats['frames_with_ref'] += 1

        overlay = (
            f"Frames: {self.stats['total_frames']}  "
            f"FramesWithRef: {self.stats['frames_with_ref']}  "
            f"LockedID: {self.locked_track_id if self.locked_track_id is not None else '-'}"
        )
        cv2.putText(frame, overlay, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return frame

    def run_on_video(self, src_path="../img/son.mp4", out_path="son_referee_out.mp4"):
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        win = "referee_track"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, int(960 * h / w))
        cv2.setMouseCallback(win, self.on_mouse)

        # 사람만 추적, CPU 모드 (원하면 device='mps'로 가속 가능)
        for result in self.model.track(
            source=src_path,
            classes=[0],              # person only
            conf=0.22,
            iou=0.6,
            imgsz=1280,
            #tracker="bytetrack.yaml",
            tracker="botsort.yaml",
            stream=True,
            verbose=False,
            device='mps'
        ):
            self.stats['total_frames'] += 1
            frame = result.orig_img.copy()
            frame = self.analyze_frame(result, frame)

            cv2.imshow(win, frame)
            writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC to quit
                break
            if key == ord('r'):  # reset lock
                self.locked_track_id = None
                self.click_xy = None

        writer.release()
        cv2.destroyAllWindows()

        print("\n=== Summary ===")
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Frames with referee: {self.stats['frames_with_ref']}")
        print(f"Locked track id: {self.locked_track_id}")
        print(f"Saved: {out_path}")

# 실행
if __name__ == "__main__":
    # target_color: 경기 영상에 맞춰 'black' 또는 'yellow' 선택
    monitor = TrafficMonitor(model_path='yolo11s.pt', target_color='black', manual_pick=False)
    monitor.run_on_video(src_path="../img/son.mp4", out_path="son_referee_out.mp4")