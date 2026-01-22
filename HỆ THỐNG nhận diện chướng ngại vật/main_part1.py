import cv2
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from yolo_detector import YOLOObstacleDetector


class IntroPage:
    def __init__(self, root, on_next_callback):
        self.root = root
        self.on_next_callback = on_next_callback

        self.frame = tk.Frame(root, bg="#020617")
        self.frame.pack(fill="both", expand=True)

        self.alpha = 0
        self.root.attributes("-alpha", 0)

        self.build_ui()
        self.fade_in()

    # ================= UI =================
    def build_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.frame, bg="#020617")
        header.pack(fill="x", pady=45)

        tk.Label(
            header,
            text="🚗 OBSTACLE DETECTION & ANALYSIS SYSTEM",
            fg="white",
            bg="#020617",
            font=("Segoe UI", 38, "bold")
        ).pack()

        tk.Label(
            header,
            text="Next-Generation Real-Time Vision System",
            fg="#94a3b8",
            bg="#020617",
            font=("Segoe UI", 16)
        ).pack(pady=6)

        # ===== DASHBOARD =====
        dashboard = tk.Frame(self.frame, bg="#020617")
        dashboard.pack(expand=True)

        self.card(dashboard, "🧠 AI OBJECT DETECTION",
                  "YOLOv8 deep learning\nphát hiện chính xác\nchướng ngại vật",
                  "#0ea5e9", 0, 0)

        self.card(dashboard, "📍 SMART ROI ANALYSIS",
                  "Tập trung vùng nguy hiểm\nphía trước xe\nloại nhiễu",
                  "#a855f7", 0, 1)

        self.card(dashboard, "⏱ OBJECT TRACKING",
                  "Theo dõi thời gian tồn tại\n> 3s đánh giá\nmức nguy hiểm",
                  "#22c55e", 1, 0)

        self.card(dashboard, "🚨 WARNING SYSTEM",
                  "Cảnh báo trực quan\nFlash đỏ khi vật thể\ngần & nguy hiểm",
                  "#ef4444", 1, 1)

        # ===== CTA =====
        btn = tk.Button(
            self.frame,
            text="▶ KHỞI ĐỘNG HỆ THỐNG",
            font=("Segoe UI", 18, "bold"),
            bg="#f59e0b",
            fg="#020617",
            activebackground="#fbbf24",
            bd=0,
            height=2,
            width=26,
            command=self.start_app
        )
        btn.pack(pady=40)

        btn.bind("<Enter>", lambda e: btn.config(bg="#fbbf24"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#f59e0b"))

        # ===== FOOTER =====
        tk.Label(
            self.frame,
            text="YOLOv8 • Computer Vision • Real-Time Analysis • Tkinter UI",
            fg="#64748b",
            bg="#020617",
            font=("Segoe UI", 11)
        ).pack(pady=10)

    # ================= CARD =================
    def card(self, parent, title, desc, accent, row, col):
        card = tk.Frame(
            parent,
            bg="#020617",
            width=360,
            height=200,
            highlightbackground=accent,
            highlightthickness=3
        )
        card.grid(row=row, column=col, padx=40, pady=30)
        card.grid_propagate(False)

        card.bind("<Enter>", lambda e: card.config(highlightthickness=5))
        card.bind("<Leave>", lambda e: card.config(highlightthickness=3))

        tk.Label(
            card,
            text=title,
            fg=accent,
            bg="#020617",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=20)

        tk.Label(
            card,
            text=desc,
            fg="white",
            bg="#020617",
            font=("Segoe UI", 13),
            justify="center"
        ).pack()

    # ================= EFFECT =================
    def fade_in(self):
        if self.alpha < 1:
            self.alpha += 0.04
            self.root.attributes("-alpha", self.alpha)
            self.root.after(25, self.fade_in)

    def start_app(self):
        self.frame.destroy()
        self.on_next_callback()

# =========================
# VIDEO HANDLER
# =========================
class VideoHandler:
    def __init__(self):
        self.cap = None
        self.last_time = time.time()
        self.fps = 0
        self.detector = YOLOObstacleDetector()
        self.detections = []
        self.detections_in_roi = []
        self.tracking_result = None
        self.frame_count = 0
        self.video_fps = 30  # Lưu video FPS
        self.time_seconds = 0  # Lưu thời gian video
        self.nearest_obstacle = None  # Vật thể gần nhất
        self.flash_frames = 0  # Counter cho flash countdown
        
        # ===== CẤU HÌNH ROI (Sẽ tự động điều chỉnh khi mở video) =====
        # Các tham số tỷ lệ phần trăm để ROI tự động scale
        self.roi_width_percent = 0.8         # 80% chiều rộng frame (rộng hơn)
        self.roi_height_top_percent = 0.1    # 10% từ trên (cao hơn)
        self.roi_height_bottom_percent = 0.85 # 85% từ trên (dài hơn)
        

    def open(self, path):
        self.cap = cv2.VideoCapture(path)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Reset tracking khi mở video mới
        self.detector.reset_tracking()
        
        # Lấy FPS từ video
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30
        print(f"[INFO] Video FPS: {self.video_fps}")
        
        # ===== TỰ ĐỘNG ĐIỀU CHỈNH ROI THEO KÍCH THƯỚC VIDEO =====
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Frame size: {frame_width}x{frame_height}")
        
        # Tính toán ROI dựa trên tỷ lệ phần trăm
        # Vùng giữa frame: chiều rộng 60%, chiều cao 40-80%
        roi_width_percent = 0.8
        roi_height_top_percent = 0.1   # 10% từ trên
        roi_height_bottom_percent = 0.85 # 85% từ trên
        
        # Tính tọa độ thực tế
        roi_x1 = int(frame_width * (1 - roi_width_percent) / 2)  # Căn giữa
        roi_x2 = int(frame_width * (1 + roi_width_percent) / 2)
        roi_y1 = int(frame_height * roi_height_top_percent)
        roi_y2 = int(frame_height * roi_height_bottom_percent)
        
        # Tạo ROI dạng hình vuông (4 góc)
        roi_points = [
            (roi_x1, roi_y1),    # Góc trên trái
            (roi_x2, roi_y1),    # Góc trên phải
            (roi_x2, roi_y2),    # Góc dưới phải
            (roi_x1, roi_y2)     # Góc dưới trái
        ]
        
        self.detector.set_roi_polygon(roi_points)
        print(f"[INFO] ROI adjusted: {roi_points}")

    def read(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # ===== PHÁT HIỆN CHƯỚNG NGẠI VẬT =====
        frame, self.detections = self.detector.detect(frame, conf=0.4)
        
        # ===== KIỂM TRA VẬT CẢN TRONG ROI =====
        self.detections_in_roi = self.detector.get_detections_in_roi(self.detections)
        
        # ===== UPDATE TRACKING & KIỂM TRA QUY TẮC 3 GIÂY =====
        self.tracking_result = self.detector.update_tracking(
            self.detections_in_roi, 
            self.frame_count, 
            fps=self.video_fps
        )
        
        # ===== TÌM VẬT THỂ GẦN NHẤT TỪ TRACKED OBJECTS =====
        h_frame = frame.shape[0]
        self.nearest_obstacle = self.detector.get_nearest_obstacle_with_tracking_id(h_frame)
        
        # ===== VẼ ROI =====
        frame = self.detector.draw_roi(frame, color=(0, 255, 0), thickness=3, alpha=0.2)
        
        # ===== VẼ THÔNG TIN TRACKING & CẢNH BÁO =====
        frame = self.detector.draw_tracking_info(frame, self.tracking_result)
        
        # ===== VẼ BOUNDING BOX ĐỎ CHO VẬT NGUY HIỂM (>3s) =====
        frame = self.detector.draw_dangerous_obstacles(frame, self.tracking_result)
        
        # ===== VẼ INDICATOR CHO VẬT THỂ GẦN NHẤT =====
        frame = self.detector.draw_nearest_indicator(frame, self.nearest_obstacle)
        
        # ===== KIỂM TRA VÀ CHỚP ĐỎ NẾU VẬT THỂ GẦN + NGUY HIỂM =====
        # Điều kiện: proximity_score > 30000 (gần ~2m) VÀ vật thể > 3s
        if (self.nearest_obstacle and 
            self.nearest_obstacle.get('proximity_score', 0) > 30000):
            
            # Kiểm tra xem vật thể gần nhất có trong dangerous_objects không (bằng object_id)
            is_dangerous = False
            if self.tracking_result:
                for danger_obj in self.tracking_result['dangerous_objects']:
                    # So sánh bằng object_id để chắc chắn là cùng vật thể
                    if danger_obj['object_id'] == self.nearest_obstacle.get('object_id'):
                        is_dangerous = True
                        break
            
            # Nếu là chướng ngại vật (>3s) + gần (<2m), bắt đầu flash
            if is_dangerous:
                self.flash_frames = 6  # Chớp 2 lần (3 frame mỗi lần)
        
        # Vẽ flash nếu flash_frames > 0
        if self.flash_frames > 0:
            # Chớp alternating (lần 1: frame 6-4, lần 2: frame 2-0)
            frame = self.detector.draw_red_flash(frame, flash_intensity=0.5)
            self.flash_frames -= 1
        
        # ===== VẼ FPS =====
        now = time.time()
        current_fps = 1 / (now - self.last_time)
        self.fps = int(0.85 * self.fps + 0.15 * current_fps)
        self.last_time = now

        cv2.putText(frame, f"FPS: {self.fps}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        
        # ===== HIỂN THỊ THỜI GIAN VIDEO =====
        self.time_seconds = self.frame_count / self.video_fps
        cv2.putText(frame, f"Time: {self.time_seconds:.2f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        
        self.frame_count += 1
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
        self.detector.reset_tracking()


# =========================
# MAIN GUI
# =========================
class ObstacleDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Obstacle Detection & Analysis System")
        self.root.geometry("1450x850")
        self.root.configure(bg="#1e272e")

        self.video = VideoHandler()
        self.video_path = None
        self.running = False

        self.build_header()
        self.build_layout()

    # ===== HEADER =====
    def build_header(self):
        header = tk.Frame(self.root, bg="#2f3640", height=70)
        header.pack(fill="x")

        tk.Label(header,
                 text="🚗 OBSTACLE DETECTION & ANALYSIS SYSTEM",
                 fg="white", bg="#2f3640",
                 font=("Segoe UI", 22, "bold")).pack(pady=8)

        tk.Label(header,
                 text="Computer Vision | AI | Real-time Video Processing",
                 fg="#dcdde1", bg="#2f3640",
                 font=("Segoe UI", 10)).pack()

    # ===== LAYOUT =====
    def build_layout(self):
        main = tk.Frame(self.root, bg="#1e272e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        main.columnconfigure(0, weight=7)  # 87.5% video
        main.columnconfigure(1, weight=1)  # 12.5% control
        main.rowconfigure(0, weight=1)

        # VIDEO PANEL
        video_card = tk.Frame(main, bg="#2f3640")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(video_card,
                 text="📺 VIDEO DISPLAY",
                 fg="white", bg="#2f3640",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=5)

        self.video_label = tk.Label(video_card, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # CONTROL PANEL
        control = tk.Frame(main, bg="#2f3640")
        control.grid(row=0, column=1, sticky="nsew")

        tk.Label(control,
                 text="🎛 CONTROL PANEL",
                 fg="white", bg="#2f3640",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=10)

        self.build_buttons(control)
        self.build_info(control)

    # ===== BUTTONS =====
    def build_buttons(self, parent):
        style = dict(font=("Segoe UI", 11, "bold"), height=2)

        tk.Button(parent, text="📂 CHỌN VIDEO",
                  bg="#0984e3", fg="white",
                  command=self.choose_video, **style).pack(fill="x", padx=10, pady=6)

        tk.Button(parent, text="▶ BẮT ĐẦU",
                  bg="#00b894", fg="white",
                  command=self.start, **style).pack(fill="x", padx=10, pady=6)

        tk.Button(parent, text="⏹ DỪNG",
                  bg="#d63031", fg="white",
                  command=self.stop, **style).pack(fill="x", padx=10, pady=6)

        tk.Button(parent, text="🔄 RESET",
                  bg="#fdcb6e", fg="black",
                  command=self.reset, **style).pack(fill="x", padx=10, pady=6)

    # ===== INFO =====
    def build_info(self, parent):
        info = tk.LabelFrame(parent, text="📊 THÔNG TIN REAL-TIME",
                             fg="white", bg="#2f3640",
                             font=("Segoe UI", 11, "bold"))
        info.pack(fill="x", padx=10, pady=15)

        # Status bar
        self.lbl_status = tk.Label(info, text="Trạng thái: Chưa chạy",
                                   fg="#00ff00", bg="#2f3640")
        self.lbl_status.pack(anchor="w", padx=10, pady=3)

        # FPS + Video info
        self.lbl_fps = tk.Label(info, text="FPS: 0 | Time: 0.00s",
                                fg="#00ff00", bg="#2f3640", font=("Segoe UI", 10, "bold"))
        self.lbl_fps.pack(anchor="w", padx=10, pady=3)

        self.lbl_video = tk.Label(info, text="Video: ---",
                                  fg="#dcdde1", bg="#2f3640")
        self.lbl_video.pack(anchor="w", padx=10, pady=3)

        # Separator
        tk.Frame(info, height=1, bg="#555555").pack(fill="x", padx=10, pady=8)

        # Summary counters
        self.lbl_summary = tk.Label(info, text="",
                                    fg="#00ff00", bg="#2f3640", 
                                    font=("Segoe UI", 9, "bold"), justify="left")
        self.lbl_summary.pack(anchor="w", padx=10, pady=3)

        # Separator
        tk.Frame(info, height=1, bg="#555555").pack(fill="x", padx=10, pady=8)

        # Detections list
        info_frame = tk.Frame(info, bg="#2f3640")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(info_frame)
        scrollbar.pack(side="right", fill="y")

        self.lbl_detections_list = tk.Listbox(info_frame, 
                                              bg="#34495e", fg="#00ff00",
                                              font=("Consolas", 8),
                                              yscrollcommand=scrollbar.set,
                                              height=10)
        self.lbl_detections_list.pack(fill="both", expand=True)
        scrollbar.config(command=self.lbl_detections_list.yview)

    # ===== LOGIC =====
    def choose_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.lbl_video.config(text=f"Video: {os.path.basename(self.video_path)}")

    def start(self):
        if not self.video_path:
            self.lbl_status.config(text="⚠ Chưa chọn video!")
            return
        self.video.open(self.video_path)
        self.running = True
        self.lbl_status.config(text="🟢 Đang xử lý...")
        self.update_frame()

    def stop(self):
        self.running = False
        self.video.release()
        self.lbl_status.config(text="⏹ Đã dừng")

    def reset(self):
        self.stop()
        self.video_label.config(image="")
        self.lbl_fps.config(text="FPS: 0 | Time: 0.00s")
        self.lbl_video.config(text="Video: ---")
        self.lbl_status.config(text="🔄 Đã reset")
        self.lbl_summary.config(text="")
        try:
            self.lbl_detections_list.delete(0, 'end')
        except:
            pass

    # ===== CORE FIX: KEEP ASPECT RATIO =====
    def update_frame(self):
        if not self.running:
            return

        frame = self.video.read()
        if frame is None:
            self.stop()
            return

        h_frame, w_frame = frame.shape[:2]
        h_label = self.video_label.winfo_height()
        w_label = self.video_label.winfo_width()

        if h_label <= 1 or w_label <= 1:
            self.root.after(30, self.update_frame)
            return

        scale = min(w_label / w_frame, h_label / h_frame)
        new_w = int(w_frame * scale)
        new_h = int(h_frame * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((h_label, w_label, 3), dtype=np.uint8)

        x = (w_label - new_w) // 2
        y = (h_label - new_h) // 2
        canvas[y:y+new_h, x:x+new_w] = resized

        self.lbl_fps.config(text=f"FPS: {self.video.fps} | Time: {self.video.time_seconds:.2f}s")

        # Cập nhật danh sách detections
        detections = self.video.detections
        detections_in_roi = self.video.detections_in_roi
        
        # Cập nhật số vật nguy hiểm (>3s)
        dangerous_count = 0
        if self.video.tracking_result:
            dangerous_count = len(self.video.tracking_result['dangerous_objects'])
        
        # Tính tóm tắt
        normal_count = len(detections) - len(detections_in_roi)
        roi_count = len(detections_in_roi)
        
        summary_text = (
            f"📍 Total: {len(detections)} | "
            f"✓ Normal: {normal_count} | "
            f"⚠️ In ROI: {roi_count} | "
            f"🚨 Danger: {dangerous_count}"
        )
        self.lbl_summary.config(text=summary_text)
        
        # ===== THÊM THÔNG TIN VẬT GẦN NHẤT =====
        nearest_info = ""
        if self.video.nearest_obstacle:
            nearest_det = self.video.nearest_obstacle
            nearest_class = nearest_det['class_name']
            # Lấy confidence từ last_detection vì nearest_det từ tracked_objects
            nearest_conf = nearest_det['last_detection']['confidence']
            nearest_score = nearest_det.get('proximity_score', 0)
            nearest_info = f"\n NEAREST: {nearest_class} ({nearest_conf:.0%}) - Score: {nearest_score:.0f}"
        
        self.lbl_summary.config(text=summary_text + nearest_info)
        
        # Hiển thị danh sách detections + thông tin ROI + tracking
        self.lbl_detections_list.delete(0, 'end')
        
        if not detections:
            self.lbl_detections_list.insert('end', "")
            self.lbl_detections_list.insert('end', "   ✓ Không phát hiện chướng ngại vật")
            self.lbl_detections_list.insert('end', "")
            return
        
        # ===== PHẦN 1: VẬT THỂ NGUY HIỂM (>3s) =====
        if dangerous_count > 0:
            self.lbl_detections_list.insert('end', "🚨 OBSTACLE (>3s):")
            self.lbl_detections_list.itemconfig('end', {'fg': '#d63031'})
            self.lbl_detections_list.insert('end', "─" * 38)
            
            dangerous_objects = self.video.tracking_result['dangerous_objects']
            for danger_obj in dangerous_objects:
                detection = danger_obj['last_detection']
                time_elapsed = danger_obj['time_elapsed']
                conf = detection['confidence']
                class_name = detection['class_name']
                
                line = f"  🔴 {class_name:15} {conf:5.0%} {time_elapsed:6.1f}s"
                self.lbl_detections_list.insert('end', line)
                self.lbl_detections_list.itemconfig('end', {'fg': '#d63031'})
            
            self.lbl_detections_list.insert('end', "")
        
        # ===== PHẦN 2: VẬT THỂ TRONG ROI (0-3s) =====
        if roi_count > 0:
            self.lbl_detections_list.insert('end', "⚠️  IN ROI (<3s):")
            self.lbl_detections_list.itemconfig('end', {'fg': '#fdcb6e'})
            self.lbl_detections_list.insert('end', "─" * 38)
            
            time_elapsed = self.video.tracking_result['time_elapsed'] if self.video.tracking_result else {}
            
            for detection in detections_in_roi:
                class_name = detection['class_name']
                conf = detection['confidence']
                
                # Cari tracked info
                tracked_time = "0.0s"
                for obj_id, elapsed in time_elapsed.items():
                    if elapsed <= 3.0:  # Chỉ hiển thị những trong 3s
                        obj_info = self.video.detector.tracked_objects.get(obj_id)
                        if obj_info and obj_info['class_name'] == class_name:
                            tracked_time = f"{elapsed:.1f}s"
                            break
                
                line = f"  🟡 {class_name:15} {conf:5.0%} {tracked_time:6}"
                self.lbl_detections_list.insert('end', line)
                self.lbl_detections_list.itemconfig('end', {'fg': '#fdcb6e'})
            
            self.lbl_detections_list.insert('end', "")
        
        # ===== PHẦN 3: VẬT THỂ BÌNH THƯỜNG (NGOÀI ROI) =====
        if normal_count > 0:
            self.lbl_detections_list.insert('end', "✓ OTHER OBJECTS:")
            self.lbl_detections_list.itemconfig('end', {'fg': '#00ff00'})
            self.lbl_detections_list.insert('end', "─" * 38)
            
            for detection in detections:
                # Bỏ qua những đã hiển thị ở roi
                is_in_roi = any(
                    d['class_name'] == detection['class_name'] 
                    for d in detections_in_roi
                )
                
                if not is_in_roi:
                    class_name = detection['class_name']
                    conf = detection['confidence']
                    
                    line = f"  🟢 {class_name:15} {conf:5.0%}"
                    self.lbl_detections_list.insert('end', line)
                    self.lbl_detections_list.itemconfig('end', {'fg': '#00ff00'})

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(canvas))
        self.video_label.imgtk = img
        self.video_label.config(image=img)

        self.root.after(30, self.update_frame)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Obstacle Detection System")
    root.geometry("1450x850")

    def start_main_app():
        ObstacleDetectionUI(root)

    IntroPage(root, start_main_app)
    root.mainloop()

