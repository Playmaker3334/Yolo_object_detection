from collections import deque
import numpy as np
import cv2
import json
import os

class DistanceCalculator:
    """Clase para cálculo de distancias a objetos detectados"""
    
    def __init__(self, config):
        """
        Inicializa el calculador de distancia
        
        Args:
            config: Configuración con parámetros de distancia y tamaños de objetos
        """
        self.config = config
        self.distance_history = {}
        self.focal_length = config["distance"]["focal_length"]
        self.smooth_frames = config["distance"]["smooth_frames"]
        self.object_sizes = config["object_sizes"]
        self.max_distance = config["distance"].get("max_distance", 500)
        self.min_size_px = config["distance"].get("min_size_px", 20)
        
        # Cargar calibración si existe
        self.calibration_data = self._load_calibration()
    
    def calculate_distance(self, class_name, width_px, height_px, x, y, frame_height, object_id):
        """
        Calcula la distancia a un objeto basado en su tamaño conocido
        
        Args:
            class_name: Nombre de la clase del objeto
            width_px: Ancho en píxeles
            height_px: Alto en píxeles
            x, y: Coordenadas del borde superior izquierdo 
            frame_height: Altura total del frame
            object_id: ID único del objeto para seguimiento
            
        Returns:
            distance: Distancia estimada en centímetros, None si no se puede calcular
        """
        # Verificar que la clase esté en nuestros tamaños conocidos
        if class_name not in self.object_sizes:
            return None
            
        # Verificar tamaño mínimo para evitar inestabilidad con objetos muy pequeños
        if width_px < self.min_size_px or height_px < self.min_size_px:
            return None
        
        # Usar cálculo especializado para personas
        if class_name == "person":
            return self.calculate_person_distance(width_px, height_px, x, y, frame_height, object_id)
                
        # Para otros objetos, usar el cálculo estándar
        obj_info = self.object_sizes[class_name]
        real_width = obj_info["width"]
        real_height = obj_info["height"]
        
        # Usar focal_length calibrado para esta clase si existe
        focal_length = self.focal_length
        if class_name in self.calibration_data and "focal_length" in self.calibration_data[class_name]:
            focal_length = self.calibration_data[class_name]["focal_length"]
        
        # Determinar qué dimensión usar
        if "reference" in obj_info and obj_info["reference"] == "height":
            # Usar altura para el cálculo
            distance = (real_height * focal_length) / height_px
        else:
            # Usar ancho para el cálculo
            distance = (real_width * focal_length) / width_px
        
        # Aplicar factor de corrección específico para la clase
        correction_factor = 1.0
        
        # Factor de la configuración
        if "correction_factor" in obj_info:
            correction_factor *= obj_info["correction_factor"]
            
        # Factor de calibración
        if class_name in self.calibration_data and "correction_factor" in self.calibration_data[class_name]:
            correction_factor *= self.calibration_data[class_name]["correction_factor"]
        
        distance *= correction_factor
            
        # Limitar a la distancia máxima configurable
        distance = min(distance, self.max_distance)
        
        # Aplicar suavizado temporal
        smoothed_distance = self._apply_smoothing(distance, object_id)
        
        return smoothed_distance
    
    def calculate_person_distance(self, width_px, height_px, x, y, frame_height, object_id):
        """
        Cálculo de distancia especializado para personas, considerando si están parcialmente visibles
        
        Args:
            width_px: Ancho de la persona en píxeles
            height_px: Alto de la persona en píxeles
            x, y: Coordenadas del borde superior izquierdo
            frame_height: Altura total del frame
            object_id: ID único para tracking
            
        Returns:
            distance: Distancia estimada en cm
        """
        # Obtener dimensiones de referencia para personas
        person_info = self.object_sizes["person"]
        real_height = person_info["height"]  # Altura real en cm
        
        # Estimación de qué porcentaje de la persona está visible
        # Verificar si el bounding box toca el borde inferior del frame
        touches_bottom = (y + height_px >= frame_height - 10)
        
        # Si la persona toca el borde inferior pero no el superior, podemos asumir
        # que vemos la parte superior del cuerpo (aproximadamente 75%)
        if touches_bottom and y > 10:
            visible_portion = 0.75
        # Si no toca ningún borde, puede ser una vista parcial (aproximadamente 60%)
        elif y > 10 and y + height_px < frame_height - 10:
            visible_portion = 0.6
        # Si toca ambos bordes o está muy cerca, asumimos que vemos la persona completa
        else:
            visible_portion = 1.0
        
        # Calcular la altura estimada total de la persona
        estimated_full_height = height_px / visible_portion
        
        # Calcular distancia basada en el tamaño estimado total
        focal_length = self.focal_length
        if "person" in self.calibration_data and "focal_length" in self.calibration_data["person"]:
            focal_length = self.calibration_data["person"]["focal_length"]
        
        distance = (real_height * focal_length) / estimated_full_height
        
        # Aplicar factor de corrección si existe
        correction_factor = 1.0
        if "correction_factor" in person_info:
            correction_factor *= person_info["correction_factor"]
        if "person" in self.calibration_data and "correction_factor" in self.calibration_data["person"]:
            correction_factor *= self.calibration_data["person"]["correction_factor"]
        
        distance *= correction_factor
        
        # Limitar a distancia máxima
        distance = min(distance, self.max_distance)
        
        # Aplicar suavizado temporal
        return self._apply_smoothing(distance, object_id)
        
    def _apply_smoothing(self, distance, object_id):
        """
        Aplica suavizado temporal a las mediciones de distancia
        
        Args:
            distance: Distancia calculada actual
            object_id: ID único del objeto
            
        Returns:
            smoothed_distance: Distancia suavizada
        """
        # Inicializar historial si no existe para este objeto
        if object_id not in self.distance_history:
            self.distance_history[object_id] = deque(maxlen=self.smooth_frames)
            
        # Añadir medición actual al historial
        self.distance_history[object_id].append(distance)
        
        # Para estabilidad, necesitamos al menos 3 mediciones
        if len(self.distance_history[object_id]) < 3:
            return distance
        
        # Convertir a array para operaciones estadísticas
        distance_array = np.array(self.distance_history[object_id])
        
        # MEJORA: Usar filtro de mediana para eliminar valores atípicos
        # Esto es más robusto que un promedio simple
        median_distance = np.median(distance_array)
        mad = np.median(np.abs(distance_array - median_distance))  # Median Absolute Deviation
        
        # Filtrar distancias que están a más de 2 MAD de la mediana
        if mad > 0:  # Evitar división por cero
            filtered_distances = distance_array[np.abs(distance_array - median_distance) <= 2 * mad]
            if len(filtered_distances) > 0:
                # MEJORA: Usar promedio ponderado dando más peso a mediciones recientes
                weights = np.linspace(0.5, 1.0, len(filtered_distances))
                return np.average(filtered_distances, weights=weights)
        
        # Si no se pudo aplicar filtrado avanzado, usar mediana
        return median_distance
        
    def reset_tracking(self):
        """Resetea el historial de distancias"""
        self.distance_history = {}
    
    def calibrate(self, class_name, real_distance, pixel_size, is_height=False):
        """
        Calibra el cálculo de distancia para una clase específica
        
        Args:
            class_name: Nombre de la clase a calibrar
            real_distance: Distancia real conocida en cm
            pixel_size: Tamaño en píxeles (ancho o alto)
            is_height: True si pixel_size es altura, False si es ancho
            
        Returns:
            success: True si se calibró correctamente
        """
        if class_name not in self.object_sizes:
            print(f"[ERROR] Clase '{class_name}' no encontrada en la configuración")
            return False
        
        # Obtener dimensiones reales del objeto
        obj_info = self.object_sizes[class_name]
        real_size = obj_info["height"] if is_height else obj_info["width"]
        
        # Calcular focal_length específico para este objeto
        focal_length = (pixel_size * real_distance) / real_size
        
        # Inicializar datos de calibración para esta clase si no existen
        if class_name not in self.calibration_data:
            self.calibration_data[class_name] = {}
        
        # Guardar focal_length calibrado
        self.calibration_data[class_name]["focal_length"] = focal_length
        
        # Calcular factor de corrección comparando con distancia estimada usando focal_length normal
        estimated_distance = (real_size * self.focal_length) / pixel_size
        correction_factor = real_distance / estimated_distance
        self.calibration_data[class_name]["correction_factor"] = correction_factor
        
        # Guardar calibración
        self._save_calibration()
        
        print(f"[INFO] Calibración para '{class_name}' guardada: focal_length={focal_length:.2f}, factor={correction_factor:.2f}")
        return True
    
    def _load_calibration(self):
        """
        Carga datos de calibración desde archivo
        
        Returns:
            calibration_data: Diccionario con datos de calibración
        """
        calibration_file = "config/calibration.json"
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error cargando calibración: {e}")
        
        return {}
    
    def _save_calibration(self):
        """Guarda datos de calibración en archivo"""
        calibration_file = "config/calibration.json"
        try:
            os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
            with open(calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Error guardando calibración: {e}")