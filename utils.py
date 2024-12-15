import cv2
import os
import numpy as np
from PIL import Image
import logging
import face_recognition
from datetime import datetime
from werkzeug.utils import secure_filename
from typing import List, Tuple, Optional

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_required_folders():
    """Create necessary folders for file uploads and face storage if they don't exist"""
    folders = ['static/uploads', 'static/faces']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o755, exist_ok=True)
            logging.debug(f"Created folder: {folder}")
        else:
            logging.debug(f"Folder already exists: {folder}")

class FaceDetector:
    def __init__(self):
        logging.debug("Initializing FaceDetector")
        try:
            # Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                logging.error("Error loading Haar cascade classifier")
                raise RuntimeError("Failed to load Haar cascade classifier")
            logging.debug("Haar cascade classifier loaded successfully")
            
            # DNN
            self.dnn_model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            self.dnn_config_file = "models/deploy.prototxt"
            self.initialize_dnn_model()
            logging.debug("DNN model initialized successfully")
            
            # MTCNN is initialized lazily when needed
            self._mtcnn_detector = None
            
        except Exception as e:
            logging.error(f"Error initializing FaceDetector: {str(e)}", exc_info=True)
            raise
            
    def detect_faces(self, image, algorithm):
        """Detect faces using the specified algorithm and return faces, color, and metadata."""
        metadata = {
            'algorithm_used': algorithm,
            'fallback_used': False,
            'confidence_scores': []
        }
        
        try:
            if algorithm == 'haar':
                faces = self.detect_faces_haar(image)
                color = (0, 255, 0)  # Green for Haar
                metadata['confidence_scores'] = [1.0] * len(faces)  # Haar doesn't provide confidence
                
            elif algorithm == 'dnn':
                faces, confidences = self.detect_faces_dnn(image)
                color = (255, 0, 0)  # Blue for DNN
                metadata['confidence_scores'] = confidences
                
            elif algorithm == 'mtcnn':
                try:
                    faces, confidences = self.detect_faces_mtcnn(image)
                    color = (0, 0, 255)  # Red for MTCNN
                    metadata['confidence_scores'] = confidences
                except Exception as e:
                    logging.error(f"MTCNN detection failed: {str(e)}")
                    logging.info("Falling back to DNN detector")
                    faces, confidences = self.detect_faces_dnn(image)
                    color = (255, 0, 0)  # Blue for DNN
                    metadata['algorithm_used'] = 'dnn'
                    metadata['fallback_used'] = True
                    metadata['confidence_scores'] = confidences
                    metadata['error'] = str(e)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            logging.info(f"Face detection completed using {metadata['algorithm_used']}")
            logging.info(f"Found {len(faces)} faces with confidence scores: {metadata['confidence_scores']}")
            
            return faces, color, metadata
            
        except Exception as e:
            logging.error(f"Face detection failed: {str(e)}", exc_info=True)
            raise

    @property
    def mtcnn_detector(self):
        if self._mtcnn_detector is None:
            try:
                from mtcnn import MTCNN
                self._mtcnn_detector = MTCNN()
                logging.debug("MTCNN detector initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing MTCNN: {str(e)}")
                raise RuntimeError("Failed to initialize MTCNN detector. Please try another algorithm.")
        return self._mtcnn_detector
    
    def initialize_dnn_model(self):
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Download DNN model files if they don't exist
        if not os.path.exists(self.dnn_model_file):
            logging.info("Downloading DNN model files...")
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            
            import urllib.request
            urllib.request.urlretrieve(model_url, self.dnn_model_file)
            urllib.request.urlretrieve(config_url, self.dnn_config_file)
        
        self.dnn_net = cv2.dnn.readNet(self.dnn_model_file, self.dnn_config_file)

    def detect_faces_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]

    def detect_faces_dnn(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        faces = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2-x1, y2-y1))
                confidences.append(float(confidence))
        
        return faces, confidences
        
    def detect_faces_mtcnn(self, image):
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detect faces
        detections = self.mtcnn_detector.detect_faces(rgb_image)
        faces = []
        confidences = []
        
        for face in detections:
            box = face['box']
            confidence = face['confidence']
            faces.append((
                int(box[0]), int(box[1]),
                int(box[2]), int(box[3])
            ))
            confidences.append(float(confidence))
            
        return faces, confidences

def get_face_embedding(face_img) -> Optional[List[float]]:
    """Generate face embedding using face_recognition library"""
    try:
        # Convert BGR to RGB for face_recognition
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Get face embedding
        face_encodings = face_recognition.face_encodings(rgb_img)
        if face_encodings:
            # Convert numpy array to list for JSON storage
            return face_encodings[0].tolist()
        return None
    except Exception as e:
        logging.error(f"Error generating face embedding: {str(e)}")
        return None

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute similarity between two face embeddings"""
    try:
        # Convert lists back to numpy arrays
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        # Compute distance
        distance = np.linalg.norm(e1 - e2)
        # Convert distance to similarity score (0 to 1)
        similarity = 1 / (1 + distance)
        return similarity
    except Exception as e:
        logging.error(f"Error computing similarity: {str(e)}")
        return 0.0

def find_similar_faces(embedding: List[float], faces: List['Face'], threshold: float = 0.55) -> List[Tuple['Face', float]]:
    """Find similar faces based on embedding similarity
    
    Args:
        embedding: Face embedding to compare
        faces: List of Face objects to compare against
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        List of tuples containing (Face, similarity_score) sorted by similarity
    """
    similar_faces = []
    try:
        # Convert input embedding to numpy array once
        embedding_array = np.array(embedding)
        
        # Process faces in batches for efficiency
        batch_size = 50
        for i in range(0, len(faces), batch_size):
            batch = faces[i:i + batch_size]
            # Filter faces with embeddings
            valid_faces = [(face, np.array(face.embedding)) 
                          for face in batch if face.embedding]
            
            if not valid_faces:
                continue
                
            # Calculate similarities in bulk
            batch_faces, batch_embeddings = zip(*valid_faces)
            distances = np.linalg.norm(
                np.array(batch_embeddings) - embedding_array, axis=1)
            similarities = 1 / (1 + distances)
            
            # Add faces that meet threshold
            for face, similarity in zip(batch_faces, similarities):
                if similarity > threshold:
                    similar_faces.append((face, float(similarity)))
        
        # Sort by similarity score
        similar_faces.sort(key=lambda x: x[1], reverse=True)
        logging.debug(f"Found {len(similar_faces)} similar faces above threshold {threshold}")
        return similar_faces
        
    except Exception as e:
        logging.error(f"Error in find_similar_faces: {str(e)}")
        return []

def save_detected_faces(image, faces, original_filename, faces_folder):
    saved_faces = []
    # Use a single timestamp for all faces from the same image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create faces folder if it doesn't exist
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder, mode=0o755, exist_ok=True)
        logging.debug(f"Created or verified faces folder: {faces_folder}")
    
    # Clean filename once for all faces
    clean_original = secure_filename(original_filename)
    
    for i, (x, y, w, h) in enumerate(faces):
        try:
            # Add padding to the face
            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_img = image[y1:y2, x1:x2]
            if face_img.size == 0:
                logging.warning(f"Skipping empty face region for face {i}")
                continue
            
            # Consistent filename format: face_YYYYMMDD_HHMMSS_index_originalname
            face_filename = f"face_{timestamp}_{i}_{clean_original}"
            face_path = os.path.join(faces_folder, face_filename)
            
            # Ensure the face image is in BGR format for cv2.imwrite
            if len(face_img.shape) == 2:  # If grayscale
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            
            # Generate face embedding
            face_embedding = get_face_embedding(face_img)
            
            success = cv2.imwrite(face_path, face_img)
            if success:
                saved_faces.append((face_filename, face_embedding))
                logging.info(f"Successfully saved face {i} to {face_path}")
            else:
                logging.error(f"Failed to save face {i} to {face_path}")
                
        except Exception as e:
            logging.error(f"Error saving face {i}: {str(e)}", exc_info=True)
            continue
    
    if not saved_faces:
        logging.warning(f"No faces were saved from image: {original_filename}")
    else:
        logging.info(f"Successfully saved {len(saved_faces)} faces from {original_filename}")
    
    return saved_faces

def process_image(filepath, algorithm='haar', faces_folder=None):
    logging.debug(f"Processing image: {filepath} with algorithm: {algorithm}")
    try:
        detector = FaceDetector()
        
        # Read the image
        image = cv2.imread(filepath)
        if image is None:
            logging.error(f"Could not read image file: {filepath}")
            raise ValueError("Could not read image file")
        logging.debug("Image loaded successfully")
        
        # Store original image for face extraction
        original_image = image.copy()
        
        # Detect faces and get color and metadata for the algorithm
        faces, color, metadata = detector.detect_faces(image, algorithm)
        
        if not faces:
            logging.warning(f"No faces detected with {algorithm} algorithm")
            
        # Draw rectangle around faces with algorithm-specific color
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # Add algorithm label above the face rectangle
            label = algorithm.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(image, (x, y - label_h - 5), (x + label_w, y), color, -1)
            # Draw label text
            cv2.putText(image, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Save individual faces if faces_folder is provided
        saved_faces = []
        if faces_folder and faces:
            if not os.path.exists(faces_folder):
                os.makedirs(faces_folder)
                logging.debug(f"Created faces folder: {faces_folder}")
            
            filename = os.path.basename(filepath)
            saved_faces = save_detected_faces(original_image, faces, filename, faces_folder)
            logging.debug(f"Saved {len(saved_faces)} individual face images")
        
        # Save processed image
        filename = os.path.basename(filepath)
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join(os.path.dirname(filepath), processed_filename)
        success = cv2.imwrite(processed_filepath, image)
        
        if not success:
            logging.error(f"Failed to save processed image: {processed_filepath}")
            raise RuntimeError("Failed to save processed image")
            
        logging.debug(f"Saved processed image: {processed_filepath}")
        logging.debug(f"Detected {len(faces)} faces using {algorithm} algorithm")
        return processed_filename, saved_faces
        
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}", exc_info=True)
        raise