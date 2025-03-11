import cv2
import numpy as np
import time

class DetectionVisualizer:
    """Clase para visualizar detecciones y distancias"""
    
    def __init__(self, config):
        """
        Inicializa el visualizador
        
        Args:
            config: Configuración con parámetros de visualización
        """
        self.config = config
        self.display_config = config["display"]
        
        # Inicializar contador FPS
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Paleta de colores consistente para clases
        self.color_palette = {}
    
    def visualize_detections(self, frame, detections, object_counts):
        """
        Visualiza detecciones en el frame
        
        Args:
            frame: Frame original
            detections: Lista de detecciones
            object_counts: Conteo de objetos por clase
            
        Returns:
            frame_viz: Frame con visualizaciones
        """
        # Actualizar FPS
        self._update_fps()
        
        # Crear copia del frame para no modificar el original
        frame_viz = frame.copy()
        
        # Mostrar cada detección
        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]
            x, y, w, h = det["box"]
            distance = det["distance"]
            
            # Determinar color (basado en distancia si está disponible)
            if distance is not None and self.display_config["show_distance"]:
                color = self._get_distance_color(distance)
                
                # Dibujar rectángulo
                cv2.rectangle(
                    frame_viz, 
                    (x, y), 
                    (x + w, y + h), 
                    color, 
                    self.display_config["line_thickness"]
                )
                
                # Mostrar etiqueta con distancia
                if self.display_config["show_labels"]:
                    # Determinar unidad de distancia
                    if self.display_config.get("distance_unit", "cm") == "m" and distance > 100:
                        # Convertir a metros si es mayor a 1 metro y está configurado
                        label = f"{class_name}: {distance/100:.2f}m"
                    else:
                        label = f"{class_name}: {distance:.1f}cm"
                    
                    # Dibujar fondo semi-transparente para texto
                    self._draw_text_with_background(
                        frame_viz, 
                        label, 
                        (x, y - 10), 
                        color
                    )
            else:
                # Color consistente para esta clase
                color = self._get_class_color(class_name)
                
                # Dibujar rectángulo
                cv2.rectangle(
                    frame_viz, 
                    (x, y), 
                    (x + w, y + h), 
                    color, 
                    self.display_config["line_thickness"]
                )
                
                # Mostrar etiqueta
                if self.display_config["show_labels"]:
                    label = f"{class_name}: {confidence:.2f}"
                    self._draw_text_with_background(
                        frame_viz, 
                        label, 
                        (x, y - 10), 
                        color
                    )
        
        # Añadir información adicional al frame
        self._add_info_overlay(frame_viz, object_counts)
        
        return frame_viz
    
    def _draw_text_with_background(self, frame, text, position, color):
        """
        Dibuja texto con fondo semi-transparente para mejor legibilidad
        
        Args:
            frame: Frame donde dibujar
            text: Texto a mostrar
            position: Posición (x, y) del texto
            color: Color del texto
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.display_config["font_scale"]
        thickness = self.display_config["line_thickness"]
        opacity = self.display_config.get("text_bg_opacity", 0.7)
        
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Coordenadas del rectángulo de fondo
        x, y = position
        bg_rect = (
            x, y - text_height - baseline,
            x + text_width + 10, y + baseline
        )
        
        # Crear capa de fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (bg_rect[0], bg_rect[1]), 
            (bg_rect[2], bg_rect[3]), 
            (0, 0, 0), 
            -1  # Relleno
        )
        
        # Combinar con el frame original
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        
        # Dibujar texto
        cv2.putText(
            frame,
            text,
            (x + 5, y - 5),  # Añadir padding
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    def _add_info_overlay(self, frame, object_counts):
        """
        Añade información general al frame (FPS, conteo de objetos)
        
        Args:
            frame: Frame donde añadir información
            object_counts: Conteo de objetos por clase
        """
        # Información de objetos detectados
        summary = ", ".join([f"{count} {obj}" for obj, count in object_counts.items()])
        summary_text = f"Objetos: {summary}"
        
        # Añadir fondo semi-transparente en la parte superior
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], 40),
            (0, 0, 0),
            -1
        )
        
        # Combinar con el frame original
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Mostrar resumen de objetos
        cv2.putText(
            frame,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.display_config["font_scale"],
            (0, 0, 255),  # Rojo
            self.display_config["line_thickness"],
            cv2.LINE_AA
        )
        
        # Mostrar FPS en la esquina inferior
        if self.display_config["show_fps"]:
            fps_text = f"FPS: {self.fps}"
            
            # Añadir fondo para FPS
            text_size = cv2.getTextSize(
                fps_text, 
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config["font_scale"],
                self.display_config["line_thickness"]
            )[0]
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (5, frame.shape[0] - 10 - text_size[1] - 10),
                (15 + text_size[0], frame.shape[0] - 5),
                (0, 0, 0),
                -1
            )
            
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(
                frame,
                fps_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config["font_scale"],
                (0, 255, 255),  # Amarillo
                self.display_config["line_thickness"],
                cv2.LINE_AA
            )
    
    def _update_fps(self):
        """Actualiza el contador de FPS"""
        self.fps_counter += 1
        if (time.time() - self.fps_start_time) > 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def _get_distance_color(self, distance):
        """
        Genera un color basado en la distancia
        
        Args:
            distance: Distancia en cm
            
        Returns:
            color: Tupla BGR para OpenCV
        """
        # Normalizar entre 0 y 1 (máximo configurado)
        max_dist = self.config["distance"].get("max_distance", 500)
        normalized = min(distance / max_dist, 1.0)
        
        colormap = self.display_config["distance_colormap"]
        
        if colormap == "GREEN_TO_RED":
            # Verde cercano a rojo lejano (formato BGR para OpenCV)
            return (
                0,                              # B
                int(255 * (1 - normalized)),    # G disminuye con distancia
                int(255 * normalized)           # R aumenta con distancia
            )
        elif colormap == "RED_TO_GREEN":
            # Rojo cercano a verde lejano
            return (
                0,                              # B
                int(255 * normalized),          # G aumenta con distancia
                int(255 * (1 - normalized))     # R disminuye con distancia
            )
        else:
            # Por defecto, azul a rojo
            if normalized < 0.5:
                # Azul a magenta
                ratio = normalized * 2
                return (
                    255,                        # B siempre 255
                    0,                          # G siempre 0
                    int(255 * ratio)            # R aumenta
                )
            else:
                # Magenta a rojo
                ratio = (normalized - 0.5) * 2
                return (
                    int(255 * (1 - ratio)),     # B disminuye
                    0,                          # G siempre 0
                    255                         # R siempre 255
                )
    
    def _get_class_color(self, class_name):
        """
        Obtiene un color consistente para una clase
        
        Args:
            class_name: Nombre de la clase
            
        Returns:
            color: Tupla BGR para OpenCV
        """
        if class_name not in self.color_palette:
            # Generar color aleatorio pero consistente
            color_hash = sum(ord(c) for c in class_name) % 100
            np.random.seed(color_hash)
            color = tuple(map(int, np.random.randint(100, 255, size=3)))
            np.random.seed(None)  # Resetear semilla
            
            self.color_palette[class_name] = color
            
        return self.color_palette[class_name]