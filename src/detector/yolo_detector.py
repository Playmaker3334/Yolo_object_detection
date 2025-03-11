from ultralytics import YOLO
import time
import numpy as np

class YOLODetector:
    """Detector de objetos basado en YOLOv8"""
    
    def __init__(self, config):
        """
        Inicializa el detector YOLO
        
        Args:
            config: Configuración con parámetros del detector
        """
        self.config = config
        self.detector_config = config["detector"]
        self.model_name = self.detector_config["model"]
        self.confidence = self.detector_config["confidence"]
        self.iou_threshold = self.detector_config.get("iou_threshold", 0.45)
        self.max_det = self.detector_config.get("max_det", 100)
        
        # Cargar modelo
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo YOLO usando Ultralytics"""
        print(f"[INFO] Cargando modelo {self.model_name}...")
        
        try:
            self.model = YOLO(f"{self.model_name}.pt")
            print(f"[INFO] Modelo {self.model_name} cargado correctamente")
        except Exception as e:
            print(f"[ERROR] Error cargando modelo: {e}")
            print("[INFO] Intentando descargar modelo...")
            # La primera vez que se usa, Ultralytics descargará el modelo automáticamente
            self.model = YOLO(f"{self.model_name}.pt")
    
    def detect(self, frame):
        """
        Detecta objetos en un frame utilizando YOLO
        
        Args:
            frame: Imagen a procesar
        
        Returns:
            detections: Lista de detecciones con clase, confianza y coordenadas
            object_counts: Diccionario con conteo de objetos por clase
        """
        if frame is None:
            print("[ERROR] Frame nulo recibido")
            return [], {}
        
        # Ejecutar detección con YOLOv8
        results = self.model(
            frame, 
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_det
        )
        
        # Extraer detecciones
        detections = []
        
        # Contadores por clase
        object_counts = {}
        
        # Para cada detección en el primer frame
        for det in results[0].boxes:
            # Extraer clase
            class_id = int(det.cls.item())
            class_name = self.model.names[class_id]
            
            # Extraer confianza
            confidence = det.conf.item()
            
            # Extraer coordenadas
            xyxy = det.xyxy.cpu().numpy()[0]  # xmin, ymin, xmax, ymax
            x, y, x2, y2 = map(int, xyxy)
            w = x2 - x
            h = y2 - y
            
            # Contar por clase
            if class_name not in object_counts:
                object_counts[class_name] = 1
            else:
                object_counts[class_name] += 1
            
            # ID único
            object_id = f"{class_name}_{object_counts[class_name]}"
            
            # Añadir a lista de detecciones
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": (x, y, w, h),
                "object_id": object_id
            })
        
        return detections, object_counts