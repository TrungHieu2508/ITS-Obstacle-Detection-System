# core/enhancer.py - Nâng cấp xử lý ảnh thông minh
import cv2
import numpy as np
from scipy import ndimage

class ImageEnhancer:
    def __init__(self, config):
        self.config = config
        self.use_gamma = config.USE_GAMMA
        self.gamma = config.GAMMA_VALUE
        self.use_hist = config.USE_HIST_EQ
        self.use_adaptive_gain = config.USE_ADAPTIVE_GAIN
        
        # Adaptive parameters
        self.current_light_level = 0
        self.adaptive_gamma = 1.0
        self.adaptive_gain = 1.0
        
        # Image quality metrics
        self.quality_history = []
        self.max_history = 20
        
        # Initialize LUTs
        self.gamma_luts = {}
        self._init_gamma_luts()
    
    def _init_gamma_luts(self):
        """Khởi tạo LUTs cho gamma correction"""
        gamma_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        for gamma in gamma_values:
            inv = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv) * 255
                for i in np.arange(256)
            ]).astype("uint8")
            self.gamma_luts[gamma] = table
    
    def estimate_light_level(self, image):
        """Ước tính mức độ ánh sáng của ảnh"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sử dụng mean và std để ước tính ánh sáng
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Kết hợp mean và std để có đánh giá tốt hơn
        # std cao có nghĩa là có độ tương phản tốt (không quá tối hay quá sáng đều)
        light_score = mean_brightness + std_brightness * 0.5
        
        self.current_light_level = light_score
        return light_score
    
    def adaptive_gamma_correction(self, image, light_level=None):
        """Gamma correction thích ứng theo điều kiện ánh sáng"""
        if light_level is None:
            light_level = self.current_light_level
        
        # Chọn gamma dựa trên mức ánh sáng
        if light_level < 50:  # Rất tối
            gamma = 1.6
        elif light_level < 100:  # Tối
            gamma = 1.4
        elif light_level < 180:  # Bình thường
            gamma = 1.2
        elif light_level < 220:  # Sáng
            gamma = 0.9
        else:  # Rất sáng
            gamma = 0.8
        
        self.adaptive_gamma = gamma
        
        # Sử dụng LUT đã tính trước
        if gamma in self.gamma_luts:
            return cv2.LUT(image, self.gamma_luts[gamma])
        else:
            return self.gamma_correction(image, gamma)
    
    def gamma_correction(self, image, gamma=None):
        """Gamma correction cơ bản"""
        if gamma is None:
            gamma = self.gamma
        
        inv = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv) * 255
            for i in np.arange(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)
    
    def histogram_equalization(self, image):
        """Histogram equalization cho ảnh màu"""
        if len(image.shape) == 2:  # Grayscale
            return cv2.equalizeHist(image)
        
        # Với ảnh màu, dùng YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
    def adaptive_gain_control(self, image):
        """Điều chỉnh gain tự động"""
        if not self.use_adaptive_gain:
            return image
        
        light_level = self.estimate_light_level(image)
        
        # Tính toán gain dựa trên mức ánh sáng
        if light_level < self.config.MIN_LIGHT_THRESHOLD:
            # Quá tối, tăng gain
            gain = 1.5 + (self.config.MIN_LIGHT_THRESHOLD - light_level) / 100
            gain = min(gain, 2.5)  # Giới hạn gain tối đa
        elif light_level > 200:
            # Quá sáng, giảm gain
            gain = 0.8 - (light_level - 200) / 400
            gain = max(gain, 0.5)  # Giới hạn gain tối thiểu
        else:
            # Bình thường
            gain = 1.0
        
        self.adaptive_gain = gain
        
        # Áp dụng gain
        result = image.astype(np.float32) * gain
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def sharpen_image(self, image, strength=0.5):
        """Làm sắc nét ảnh"""
        if strength <= 0:
            return image
        
        # Kernel làm sắc nét
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        # Áp dụng convolution
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend với ảnh gốc
        result = cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)
        
        return result
    
    def denoise_image(self, image, strength=3):
        """Khử nhiễu ảnh"""
        if strength <= 0:
            return image
        
        # Sử dụng Non-local Means Denoising
        if len(image.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, 7, 21
            )
        else:
            result = cv2.fastNlMeansDenoising(
                image, None, strength, 7, 21
            )
        
        return result
    
    def enhance_contrast(self, image, clip_limit=2.0):
        """Tăng cường độ tương phản cục bộ"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Áp dụng CLAHE trên kênh L
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            
            lab_eq = cv2.merge((l_eq, a, b))
            result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            result = clahe.apply(image)
        
        return result
    
    def adjust_color_balance(self, image, temperature=6500):
        """Cân bằng màu sắc theo nhiệt độ màu"""
        # Chuyển đổi temperature (Kelvin) thành màu sắc
        # Đơn giản hóa: temperature cao -> lạnh (xanh), thấp -> ấm (đỏ)
        
        if temperature < 0:
            return image
        
        # Normalize temperature (3000K-9000K -> 0.0-1.0)
        temp_norm = (temperature - 3000) / 6000
        temp_norm = np.clip(temp_norm, 0, 1)
        
        # Tạo adjustment matrix
        # Ấm (đỏ/vàng) khi temp_norm thấp, lạnh (xanh) khi temp_norm cao
        b_gain = 1.0 + (temp_norm - 0.5) * 0.2  # Xanh
        g_gain = 1.0  # Giữ nguyên xanh lá
        r_gain = 1.0 + (0.5 - temp_norm) * 0.2  # Đỏ
        
        # Áp dụng gain
        result = image.astype(np.float32)
        result[:, :, 0] *= b_gain  # Blue
        result[:, :, 1] *= g_gain  # Green
        result[:, :, 2] *= r_gain  # Red
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def measure_image_quality(self, image):
        """Đo chất lượng ảnh (độ sắc nét, độ tương phản)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Độ sắc nét (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Độ tương phản (std của histogram)
        contrast = np.std(gray)
        
        # 3. Độ sáng trung bình
        brightness = np.mean(gray)
        
        # 4. Signal-to-Noise Ratio (ước tính)
        noise = np.std(gray - cv2.medianBlur(gray, 3))
        snr = contrast / (noise + 1e-6)
        
        quality_score = (sharpness * 0.4 + contrast * 0.3 + 
                        min(snr, 50) * 0.2 + min(abs(brightness - 128), 128) * 0.1)
        
        # Lưu vào history
        self.quality_history.append(quality_score)
        if len(self.quality_history) > self.max_history:
            self.quality_history.pop(0)
        
        return {
            'sharpness': sharpness,
            'contrast': contrast,
            'brightness': brightness,
            'snr': snr,
            'overall': quality_score,
            'light_level': self.current_light_level
        }
    
    def auto_enhance(self, image):
        """Tự động áp dụng các kỹ thuật enhancement dựa trên chất lượng ảnh"""
        quality = self.measure_image_quality(image)
        result = image.copy()
        
        # Quyết định kỹ thuật enhancement dựa trên chất lượng
        if quality['light_level'] < 50:
            # Rất tối: tăng gain và gamma
            result = self.adaptive_gain_control(result)
            result = self.adaptive_gamma_correction(result, quality['light_level'])
            result = self.enhance_contrast(result, 3.0)
            
        elif quality['light_level'] > 200:
            # Rất sáng: giảm gamma, cân bằng màu
            result = self.adaptive_gamma_correction(result, quality['light_level'])
            result = self.adjust_color_balance(result, 7500)  # Lạnh hơn
            
        else:
            # Bình thường: áp dụng cài đặt mặc định
            if self.use_gamma:
                result = self.gamma_correction(result)
            
            if self.use_hist:
                result = self.histogram_equalization(result)
            
            if self.use_adaptive_gain:
                result = self.adaptive_gain_control(result)
        
        # Luôn làm sắc nét nhẹ
        result = self.sharpen_image(result, 0.3)
        
        # Khử nhiễu nếu SNR thấp
        if quality['snr'] < 10:
            result = self.denoise_image(result, 5)
        elif quality['snr'] < 20:
            result = self.denoise_image(result, 3)
        
        return result, quality
    
    def process(self, frame):
        """Xử lý frame với các enhancement được chọn"""
        if frame is None:
            return None
        
        # Ước tính mức ánh sáng
        self.estimate_light_level(frame)
        
        # Tự động enhance
        enhanced_frame, quality_metrics = self.auto_enhance(frame)
        
        return enhanced_frame
    
    def get_status(self):
        """Lấy trạng thái hiện tại của enhancer"""
        return {
            'use_gamma': self.use_gamma,
            'gamma': self.gamma,
            'adaptive_gamma': self.adaptive_gamma,
            'use_hist': self.use_hist,
            'use_adaptive_gain': self.use_adaptive_gain,
            'adaptive_gain': self.adaptive_gain,
            'current_light_level': self.current_light_level,
            'avg_quality': np.mean(self.quality_history) if self.quality_history else 0
        }
    
    def toggle_gamma(self):
        """Bật/tắt gamma correction"""
        self.use_gamma = not self.use_gamma
        return self.use_gamma
    
    def toggle_hist(self):
        """Bật/tắt histogram equalization"""
        self.use_hist = not self.use_hist
        return self.use_hist
    
    def toggle_adaptive_gain(self):
        """Bật/tắt adaptive gain"""
        self.use_adaptive_gain = not self.use_adaptive_gain
        return self.use_adaptive_gain
    
    def adjust_gamma(self, delta):
        """Điều chỉnh gamma value"""
        self.gamma = max(0.1, min(3.0, self.gamma + delta))
        return self.gamma