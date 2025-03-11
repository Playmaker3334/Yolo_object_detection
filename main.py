import cv2
import time
import os
import sys

# Añadir directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos del proyecto
from utils.config_loader import ConfigLoader
from camera.camera_utils import CameraHandler
from src.detector.yolo_detector import YOLODetector
from src.detector.distance_calc import DistanceCalculator
from visualization.visualizer import DetectionVisualizer

def main():
    """Función principal de la aplicación"""
    print("=== YOLO Distance Detector ===")
    print("Comandos:")
    print("  'q' - Salir")
    print("  'r' - Reiniciar tracking")
    print("  'c' - Modo calibración")
    print("  's' - Guardar calibración actual")
    
    # Ruta de configuración
    config_path = "config/config.yml"
    
    # Verificar si existe el archivo de configuración, y crearlo si no existe
    if not os.path.exists(config_path):
        print("[INFO] Creando archivo de configuración por defecto...")
        try:
            ConfigLoader.create_default_config(config_path)
            print("[INFO] Archivo de configuración creado correctamente.")
        except Exception as e:
            print(f"[ERROR] No se pudo crear configuración: {e}")
            return
    
    # Cargar configuración
    try:
        config = ConfigLoader.load_config(config_path)
        print("[INFO] Configuración cargada correctamente.")
    except Exception as e:
        print(f"[ERROR] Error cargando configuración: {e}")
        return
    
    # Inicializar componentes
    try:
        # 1. Inicializar cámara
        camera = CameraHandler(config)
        if not camera.initialize():
            print("[ERROR] No se pudo inicializar la cámara.")
            return
        
        # 2. Inicializar detector YOLO
        detector = YOLODetector(config)
        
        # 3. Inicializar calculador de distancia
        distance_calculator = DistanceCalculator(config)
        
        # 4. Inicializar visualizador
        visualizer = DetectionVisualizer(config)
        
        print("[INFO] Sistema inicializado. Iniciando bucle de detección...")
        
        # Variables para calibración
        calibration_mode = config["distance"].get("calibration_mode", False)
        calibration_object = None
        calibration_distance = 100  # cm por defecto
        
        # Bucle principal
        while True:
            # Capturar frame
            frame, success = camera.read_frame()
            
            if not success:
                print("[ERROR] Error al capturar el frame. Reintentando...")
                time.sleep(0.5)
                continue
            
            try:
                # Detectar objetos
                detections, object_counts = detector.detect(frame)
                
                # Calcular distancias para cada detección
                for det in detections:
                    class_name = det["class_name"]
                    x, y, w, h = det["box"]
                    object_id = det["object_id"]
                    
                    # Calcular distancia si la clase está en los tamaños conocidos
                    # Pasar información adicional (coordenadas y altura del frame)
                    distance = distance_calculator.calculate_distance(
                        class_name, 
                        w, h, 
                        x, y,
                        frame.shape[0],  # Altura del frame
                        object_id
                    )
                    det["distance"] = distance
                    
                    # Si estamos en modo calibración y este objeto es el seleccionado
                    if calibration_mode and calibration_object == class_name:
                        # Dibujar información de calibración
                        label = f"CALIBRANDO: {class_name} a {calibration_distance}cm"
                        cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "Presiona 's' para guardar, '+'/'-' para ajustar distancia", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Visualizar resultados
                processed_frame = visualizer.visualize_detections(frame, detections, object_counts)
                
                # Mostrar información adicional en modo calibración
                if calibration_mode:
                    cv2.putText(processed_frame, "MODO CALIBRACIÓN", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Mostrar objetos disponibles para calibrar
                    available_objects = [obj for obj in object_counts.keys() if obj in config["object_sizes"]]
                    if available_objects:
                        cv2.putText(processed_frame, f"Objetos detectados: {', '.join(available_objects)}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        if not calibration_object:
                            cv2.putText(processed_frame, "Presiona número para seleccionar objeto:", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                            
                            for i, obj in enumerate(available_objects):
                                cv2.putText(processed_frame, f"{i+1}: {obj}", 
                                           (20, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Mostrar frame procesado
                cv2.imshow("YOLO Distance Detector", processed_frame)
                
                # Capturar tecla
                key = cv2.waitKey(1) & 0xFF
                
                # Procesar teclas
                if key == ord('q'):  # Salir
                    break
                elif key == ord('r'):  # Reiniciar tracking
                    distance_calculator.reset_tracking()
                    print("[INFO] Tracking reiniciado")
                elif key == ord('c'):  # Activar/desactivar modo calibración
                    calibration_mode = not calibration_mode
                    calibration_object = None
                    print(f"[INFO] Modo calibración: {'ACTIVADO' if calibration_mode else 'DESACTIVADO'}")
                
                # Teclas para modo calibración
                if calibration_mode:
                    if key == ord('s') and calibration_object:  # Guardar calibración
                        # Buscar la detección del objeto seleccionado
                        for det in detections:
                            if det["class_name"] == calibration_object:
                                _, _, w, h = det["box"]
                                # Obtener referencia de configuración
                                ref = config["object_sizes"][calibration_object].get("reference", "width")
                                is_height = (ref == "height")
                                size_px = h if is_height else w
                                
                                # Calibrar
                                distance_calculator.calibrate(
                                    calibration_object, 
                                    calibration_distance, 
                                    size_px, 
                                    is_height
                                )
                                break
                    elif key == ord('+') or key == ord('='):  # Aumentar distancia calibración
                        calibration_distance += 5
                        print(f"[INFO] Distancia de calibración: {calibration_distance}cm")
                    elif key == ord('-') or key == ord('_'):  # Disminuir distancia calibración
                        calibration_distance = max(5, calibration_distance - 5)
                        print(f"[INFO] Distancia de calibración: {calibration_distance}cm")
                    elif ord('1') <= key <= ord('9'):  # Seleccionar objeto por número
                        idx = key - ord('1')
                        available_objects = [obj for obj in object_counts.keys() if obj in config["object_sizes"]]
                        if idx < len(available_objects):
                            calibration_object = available_objects[idx]
                            print(f"[INFO] Objeto seleccionado para calibración: {calibration_object}")
            
            except Exception as e:
                print(f"[ERROR] Error en procesamiento: {e}")
                # Mostrar el frame original en caso de error
                cv2.imshow("YOLO Distance Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Liberar recursos
        camera.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        # Intentar liberar recursos
        try:
            if 'camera' in locals():
                camera.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    os.makedirs("config", exist_ok=True)
    os.makedirs("src/detector", exist_ok=True)
    os.makedirs("camera", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)
    
    # Ejecutar la aplicación
    main()