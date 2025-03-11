import cv2
import time
from collections import deque

class CameraHandler:
    """Clase para gestionar la cámara y captura de frames"""
    
    def __init__(self, config):
        """
        Inicializa el gestor de cámara
        
        Args:
            config: Configuración con parámetros de cámara
        """
        self.config = config["camera"]
        self.camera_index = self.config["index"]
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.fps = self.config["fps"]
        self.buffer_size = self.config.get("buffer_size", 1)
        
        # Inicializar cámara
        self.cap = None
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
    def initialize(self):
        """
        Inicializa la cámara con varios métodos
        
        Returns:
            bool: True si la cámara se inicializó correctamente
        """
        # Intento 1: DirectShow (Windows)
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                print(f"[INFO] Cámara abierta con DirectShow en índice {self.camera_index}")
                self._configure_camera()
                return True
            else:
                self.cap = None
                print("[WARNING] No se pudo abrir la cámara con DirectShow")
        except Exception as e:
            print(f"[WARNING] Error con DirectShow: {e}")
            self.cap = None
        
        # Intento 2: Apertura estándar
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                print(f"[INFO] Cámara abierta en modo estándar en índice {self.camera_index}")
                self._configure_camera()
                return True
            else:
                print(f"[ERROR] No se pudo abrir la cámara en índice {self.camera_index}")
                # Probar con cámara 0 (por defecto)
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    print("[INFO] Cámara abierta en índice 0")
                    self._configure_camera()
                    return True
        except Exception as e:
            print(f"[ERROR] Error inicializando cámara: {e}")
            
        return False
    
    def _configure_camera(self):
        """Configura parámetros de la cámara (resolución, FPS)"""
        if self.cap is not None:
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Mostrar información de la cámara
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Resolución de cámara: {actual_width}x{actual_height}, {actual_fps} FPS")
    
    def read_frame(self):
        """
        Lee un frame de la cámara
        
        Returns:
            frame: Frame capturado o None si hay error
            success: True si se leyó correctamente
        """
        if self.cap is None or not self.cap.isOpened():
            return None, False
        
        # Capturar frame
        success, frame = self.cap.read()
        
        if success:
            # Añadir al buffer si se configuró para estabilidad
            if self.buffer_size > 1:
                self.frame_buffer.append(frame)
                # Si tenemos suficientes frames en el buffer, devolver el promedio
                if len(self.frame_buffer) == self.buffer_size:
                    # Simple implementación: devolver el último frame
                    # Para más estabilidad, se podría implementar un promedio de frames
                    return self.frame_buffer[-1], True
            
            return frame, True
        else:
            print("[ERROR] Error al capturar el frame")
            return None, False
    
    def release(self):
        """Libera los recursos de la cámara"""
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Cámara liberada")
            self.cap = None