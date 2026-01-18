import cv2
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from yolo_detector import YOLOObstacleDetector

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

    def open(self, path):
        self.cap = cv2.VideoCapture(path)
        self.last_time = time.time()

    def read(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Phát hiện chướng ngại vật
        frame, self.detections = self.detector.detect(frame, conf=0.4)

        now = time.time()
        current_fps = 1 / (now - self.last_time)
        self.fps = int(0.85 * self.fps + 0.15 * current_fps)
        self.last_time = now

        cv2.putText(frame, f"FPS: {self.fps}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        return frame

    def release(self):
        if self.cap:
            self.cap.release()


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
                 fg="#00cec9", bg="#2f3640",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=5)

        self.video_label = tk.Label(video_card, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # CONTROL PANEL
        control = tk.Frame(main, bg="#2f3640")
        control.grid(row=0, column=1, sticky="nsew")

        tk.Label(control,
                 text="🎛 CONTROL PANEL",
                 fg="#00cec9", bg="#2f3640",
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
        info = tk.LabelFrame(parent, text="📊 THÔNG TIN",
                             fg="white", bg="#2f3640",
                             font=("Segoe UI", 11, "bold"))
        info.pack(fill="x", padx=10, pady=15)

        self.lbl_status = tk.Label(info, text="Trạng thái: Chưa chạy",
                                   fg="#00cec9", bg="#2f3640")
        self.lbl_status.pack(anchor="w", padx=10)

        self.lbl_fps = tk.Label(info, text="FPS: 0",
                                fg="#00cec9", bg="#2f3640")
        self.lbl_fps.pack(anchor="w", padx=10)

        self.lbl_video = tk.Label(info, text="Video: ---",
                                  fg="#dcdde1", bg="#2f3640")
        self.lbl_video.pack(anchor="w", padx=10)

        self.lbl_detection_count = tk.Label(info, text="Detections: 0",
                                            fg="#00b894", bg="#2f3640")
        self.lbl_detection_count.pack(anchor="w", padx=10)

        info_frame = tk.Frame(info, bg="#2f3640")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(info_frame)
        scrollbar.pack(side="right", fill="y")

        self.lbl_detections_list = tk.Listbox(info_frame, 
                                              bg="#34495e", fg="#00cec9",
                                              font=("Segoe UI", 9),
                                              yscrollcommand=scrollbar.set,
                                              height=6)
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
        self.lbl_fps.config(text="FPS: 0")
        self.lbl_video.config(text="Video: ---")
        self.lbl_status.config(text="🔄 Đã reset")
        self.lbl_detection_count.config(text="Detections: 0")
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

        self.lbl_fps.config(text=f"FPS: {self.video.fps}")

        # Cập nhật danh sách detections
        detections = self.video.detections
        self.lbl_detection_count.config(text=f"Detections: {len(detections)}")
        
        # Hiển thị danh sách detections
        self.lbl_detections_list.delete(0, 'end')
        if detections:
            for i, det in enumerate(detections, 1):
                info = f"{i}. {det['class_name']} {det['confidence']:.0%}"
                self.lbl_detections_list.insert('end', info)
        else:
            self.lbl_detections_list.insert('end', "Chưa phát hiện")

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
    ObstacleDetectionUI(root)
    root.mainloop()
