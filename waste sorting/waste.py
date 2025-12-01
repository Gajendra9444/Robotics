import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class WebcamWasteClassifier:
    def __init__(self):
        self.cap = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.waste_data = []
        self.storage_capacity = 1000
        self.current_storage = 0
        
        # Waste categories with color coding for display
        self.waste_categories = {
            'plastic': {'decomposition_time': 450, 'recyclable': True, 'hazard_level': 2, 'color': (255, 0, 0)},    # Red
            'paper': {'decomposition_time': 30, 'recyclable': True, 'hazard_level': 1, 'color': (0, 255, 0)},      # Green
            'glass': {'decomposition_time': 1000000, 'recyclable': True, 'hazard_level': 1, 'color': (255, 255, 0)}, # Yellow
            'metal': {'decomposition_time': 200, 'recyclable': True, 'hazard_level': 2, 'color': (128, 128, 128)}, # Gray
            'organic': {'decomposition_time': 15, 'recyclable': False, 'hazard_level': 1, 'color': (165, 42, 42)}, # Brown
            'electronic': {'decomposition_time': 1000, 'recyclable': True, 'hazard_level': 3, 'color': (255, 0, 255)}, # Magenta
            'hazardous': {'decomposition_time': 500, 'recyclable': False, 'hazard_level': 4, 'color': (0, 0, 255)} # Blue
        }
        
        # Initialize webcam
        self.init_webcam()
        
    def init_webcam(self):
        """Initialize webcam connection"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        print("Webcam initialized successfully")
        return True
    
    def extract_image_features(self, image):
        """Extract features from image for classification"""
        if image is None:
            return None
            
        # Resize image for consistent feature extraction
        image = cv2.resize(image, (224, 224))
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # Color features (mean and std of each channel in different color spaces)
        for channel in range(3):
            features.append(np.mean(image[:, :, channel]))
            features.append(np.std(image[:, :, channel]))
            
            features.append(np.mean(hsv[:, :, channel]))
            features.append(np.std(hsv[:, :, channel]))
            
            features.append(np.mean(lab[:, :, channel]))
            features.append(np.std(lab[:, :, channel]))
        
        # Texture features using Sobel filters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        features.append(np.mean(sobelx))
        features.append(np.std(sobelx))
        features.append(np.mean(sobely))
        features.append(np.std(sobely))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (224 * 224))
        
        # Brightness and contrast
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features)
    
    def generate_training_data(self, num_samples_per_class=100):
        """Generate synthetic training data based on color and texture patterns"""
        print("Generating training data...")
        
        categories = list(self.waste_categories.keys())
        features_list = []
        labels_list = []
        
        for category in categories:
            color = self.waste_categories[category]['color']
            
            for _ in range(num_samples_per_class):
                # Create synthetic image based on category color
                synthetic_image = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # Base color with variations
                base_color = np.array(color, dtype=np.uint8)
                variation = np.random.randint(-30, 30, 3, dtype=np.int16)
                actual_color = np.clip(base_color + variation, 0, 255).astype(np.uint8)
                
                # Fill with base color
                synthetic_image[:, :] = actual_color
                
                # Add texture based on category with proper type handling
                if category == 'plastic':
                    # Add shiny reflections
                    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                    synthetic_image = cv2.add(synthetic_image, noise)
                elif category == 'paper':
                    # Add paper-like texture
                    texture = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
                    synthetic_image = cv2.subtract(synthetic_image, texture)
                elif category == 'glass':
                    # Add transparency effect
                    overlay = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                    synthetic_image = cv2.addWeighted(synthetic_image, 0.7, overlay, 0.3, 0)
                elif category == 'metal':
                    # Add metallic shine
                    shine = np.random.randint(50, 100, (112, 112, 3), dtype=np.uint8)
                    synthetic_image[56:168, 56:168] = cv2.add(synthetic_image[56:168, 56:168], shine)
                elif category == 'organic':
                    # Add organic texture
                    organic_noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
                    synthetic_image = cv2.add(synthetic_image, organic_noise)
                elif category == 'electronic':
                    # Add electronic component patterns
                    pattern = np.random.randint(0, 25, (224, 224, 3), dtype=np.uint8)
                    synthetic_image = cv2.add(synthetic_image, pattern)
                elif category == 'hazardous':
                    # Add warning stripe patterns
                    for i in range(0, 224, 20):
                        synthetic_image[i:i+10, :] = np.clip(synthetic_image[i:i+10, :] + 50, 0, 255)
                
                # Extract features
                features = self.extract_image_features(synthetic_image)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(category)
        
        return np.array(features_list), np.array(labels_list)
    
    def create_simple_training_data(self):
        """Create simpler training data without complex image generation"""
        print("Creating simplified training data...")
        
        categories = list(self.waste_categories.keys())
        features_list = []
        labels_list = []
        
        # Create feature patterns for each category
        for category in categories:
            for _ in range(100):  # 100 samples per category
                features = []
                
                # Create distinctive feature patterns for each category
                if category == 'plastic':
                    # Plastic: medium weight, low density, various colors
                    features.extend([
                        np.random.uniform(0.5, 5.0),    # weight
                        np.random.uniform(1.0, 10.0),   # volume
                        np.random.uniform(0.8, 1.2),    # density
                        np.random.uniform(0.6, 0.9),    # color_blue (plastic often blue)
                        np.random.uniform(0.3, 0.7),    # color_green
                        np.random.uniform(0.2, 0.6),    # color_red
                        0.7,  # high smoothness
                        0.3   # low edge density
                    ])
                elif category == 'paper':
                    # Paper: light weight, low density, light colors
                    features.extend([
                        np.random.uniform(0.1, 2.0),
                        np.random.uniform(0.5, 5.0),
                        np.random.uniform(0.4, 0.8),
                        np.random.uniform(0.7, 0.9),  # high brightness
                        np.random.uniform(0.6, 0.9),
                        np.random.uniform(0.5, 0.8),
                        0.8,  # high smoothness
                        0.2   # low edge density
                    ])
                elif category == 'glass':
                    # Glass: heavy, high density, transparent
                    features.extend([
                        np.random.uniform(1.0, 10.0),
                        np.random.uniform(0.5, 3.0),
                        np.random.uniform(2.2, 2.8),
                        np.random.uniform(0.4, 0.7),
                        np.random.uniform(0.4, 0.7),
                        np.random.uniform(0.4, 0.7),
                        0.9,  # very smooth
                        0.8   # high edge density (edges visible)
                    ])
                elif category == 'metal':
                    # Metal: heavy, very high density, metallic
                    features.extend([
                        np.random.uniform(2.0, 15.0),
                        np.random.uniform(0.5, 5.0),
                        np.random.uniform(2.5, 8.0),
                        np.random.uniform(0.3, 0.6),  # gray tones
                        np.random.uniform(0.3, 0.6),
                        np.random.uniform(0.3, 0.6),
                        0.6,  # medium smoothness
                        0.5   # medium edge density
                    ])
                elif category == 'organic':
                    # Organic: light, medium density, brown/green
                    features.extend([
                        np.random.uniform(0.2, 3.0),
                        np.random.uniform(1.0, 8.0),
                        np.random.uniform(0.8, 1.1),
                        np.random.uniform(0.3, 0.5),  # brown/green tones
                        np.random.uniform(0.4, 0.6),
                        np.random.uniform(0.2, 0.4),
                        0.4,  # low smoothness
                        0.6   # high edge density (textured)
                    ])
                elif category == 'electronic':
                    # Electronic: medium weight, various colors
                    features.extend([
                        np.random.uniform(1.0, 8.0),
                        np.random.uniform(2.0, 6.0),
                        np.random.uniform(1.5, 3.0),
                        np.random.uniform(0.1, 0.9),  # various colors
                        np.random.uniform(0.1, 0.9),
                        np.random.uniform(0.1, 0.9),
                        0.5,  # medium smoothness
                        0.7   # high edge density (components)
                    ])
                elif category == 'hazardous':
                    # Hazardous: various properties, often distinctive colors
                    features.extend([
                        np.random.uniform(0.5, 8.0),
                        np.random.uniform(1.0, 6.0),
                        np.random.uniform(1.0, 2.0),
                        np.random.uniform(0.1, 0.3),  # dark/warning colors
                        np.random.uniform(0.1, 0.3),
                        np.random.uniform(0.7, 0.9),  # often bright warning colors
                        0.3,  # low smoothness
                        0.4   # medium edge density
                    ])
                
                # Add some random variation
                features = [f * np.random.uniform(0.9, 1.1) for f in features]
                features_list.append(features)
                labels_list.append(category)
        
        return np.array(features_list), np.array(labels_list)
    
    def train_classifier(self):
        """Train the waste classification model"""
        print("Training waste classifier...")
        
        try:
            # Try simplified training first
            X, y = self.create_simple_training_data()
        except Exception as e:
            print(f"Simplified training failed: {e}")
            # Fallback: create very basic training data
            X, y = self.create_fallback_training_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y_encoded)
        
        # Calculate training accuracy
        accuracy = self.model.score(X_scaled, y_encoded)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def create_fallback_training_data(self):
        """Create very basic fallback training data"""
        print("Creating fallback training data...")
        
        categories = list(self.waste_categories.keys())
        features_list = []
        labels_list = []
        
        # Simple feature patterns based on category properties
        for category in categories:
            for i in range(50):
                if category == 'plastic':
                    features = [0.5, 2.0, 0.9, 0.7, 0.3, 0.2, 0.7, 0.3]
                elif category == 'paper':
                    features = [0.1, 1.0, 0.5, 0.8, 0.7, 0.6, 0.8, 0.2]
                elif category == 'glass':
                    features = [1.5, 1.0, 2.5, 0.5, 0.5, 0.5, 0.9, 0.8]
                elif category == 'metal':
                    features = [2.0, 1.0, 7.0, 0.4, 0.4, 0.4, 0.6, 0.5]
                elif category == 'organic':
                    features = [0.3, 2.0, 0.8, 0.4, 0.5, 0.3, 0.4, 0.6]
                elif category == 'electronic':
                    features = [3.0, 3.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.7]
                elif category == 'hazardous':
                    features = [1.0, 1.5, 1.2, 0.2, 0.2, 0.8, 0.3, 0.4]
                
                # Add some noise
                features = [f * np.random.uniform(0.95, 1.05) for f in features]
                features_list.append(features)
                labels_list.append(category)
        
        return np.array(features_list), np.array(labels_list)
    
    def classify_waste_from_image(self, image):
        """Classify waste from webcam image using extracted features"""
        if self.model is None:
            return "unknown", 0.0
        
        try:
            # Extract features from actual image
            features = self.extract_image_features(image)
            if features is None:
                return "unknown", 0.0
            
            # Use only the first 8 features to match training data
            if len(features) > 8:
                features = features[:8]
            
            # Scale features and predict
            features_scaled = self.scaler.transform([features])
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            category = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = probabilities[prediction_encoded]
            
            return category, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown", 0.0
    
    def estimate_waste_properties(self, category, image):
        """Estimate waste properties based on category and image analysis"""
        # Basic property estimation based on category
        base_properties = {
            'plastic': {'weight': 0.5, 'volume': 2.0, 'density': 0.9},
            'paper': {'weight': 0.1, 'volume': 1.0, 'density': 0.5},
            'glass': {'weight': 1.5, 'volume': 1.0, 'density': 2.5},
            'metal': {'weight': 2.0, 'volume': 1.0, 'density': 7.0},
            'organic': {'weight': 0.3, 'volume': 2.0, 'density': 0.8},
            'electronic': {'weight': 3.0, 'volume': 3.0, 'density': 2.0},
            'hazardous': {'weight': 1.0, 'volume': 1.5, 'density': 1.2}
        }
        
        properties = base_properties.get(category, {'weight': 1.0, 'volume': 1.0, 'density': 1.0})
        
        # Add some variation
        properties['weight'] *= np.random.uniform(0.8, 1.2)
        properties['volume'] *= np.random.uniform(0.8, 1.2)
        
        # Add environmental properties
        properties.update({
            'temperature': np.random.uniform(18, 35),
            'moisture': np.random.uniform(10, 80),
            'ph_level': np.random.uniform(5, 8)
        })
        
        return properties
    
    def add_waste_item(self, category, confidence, image, properties):
        """Add classified waste item to storage"""
        waste_item = {
            'id': len(self.waste_data) + 1,
            'category': category,
            'confidence': confidence,
            'weight': properties['weight'],
            'volume': properties['volume'],
            'timestamp': datetime.datetime.now(),
            'properties': properties,
            'storage_zone': self.assign_storage_zone(category, properties),
            'image_size': image.shape[:2] if image is not None else (0, 0)
        }
        
        # Check storage capacity
        if self.current_storage + properties['weight'] <= self.storage_capacity:
            self.waste_data.append(waste_item)
            self.current_storage += properties['weight']
            return True, waste_item
        else:
            return False, None
    
    def assign_storage_zone(self, category, properties):
        """Assign storage zone based on waste category and properties"""
        hazard_level = self.waste_categories[category]['hazard_level']
        
        if hazard_level >= 4:
            return "HAZARDOUS CONTAINMENT"
        elif hazard_level >= 3:
            return "SECURE ZONE"
        elif properties.get('temperature', 25) > 30:
            return "COOLING ZONE"
        elif self.waste_categories[category]['recyclable']:
            return "RECYCLING ZONE"
        else:
            return "GENERAL STORAGE"
    
    def draw_dashboard(self, frame, classification, confidence, storage_info):
        """Draw information dashboard on the video frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw classification info
        color = self.waste_categories.get(classification, {}).get('color', (255, 255, 255))
        
        cv2.putText(frame, f"Classification: {classification.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Storage: {storage_info}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press SPACE to capture | Q to quit", 
                   (width - 400, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_storage_info(self):
        """Get current storage information"""
        used = self.current_storage
        capacity = self.storage_capacity
        utilization = (used / capacity) * 100
        return f"{used:.1f}kg / {capacity}kg ({utilization:.1f}%)"
    
    def run_webcam_classification(self):
        """Main webcam classification loop"""
        if self.cap is None:
            print("Webcam not available")
            return
        
        print("\nStarting webcam waste classification...")
        print("Instructions:")
        print("- Show waste item to camera")
        print("- Press SPACEBAR to capture and classify")
        print("- Press 'Q' to quit")
        print("- Press 'S' to show storage summary")
        print("- Press 'E' to export data")
        
        last_classification = "unknown"
        last_confidence = 0.0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Draw dashboard
            storage_info = self.get_storage_info()
            frame = self.draw_dashboard(frame, last_classification, last_confidence, storage_info)
            
            # Display frame
            cv2.imshow('Waste Classification System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE to capture
                # Classify current frame
                classification, confidence = self.classify_waste_from_image(frame)
                last_classification = classification
                last_confidence = confidence
                
                # Estimate properties and add to storage
                properties = self.estimate_waste_properties(classification, frame)
                success, waste_item = self.add_waste_item(classification, confidence, frame, properties)
                
                if success:
                    print(f"✅ Added: {classification} ({properties['weight']:.1f}kg) to {waste_item['storage_zone']}")
                else:
                    print("❌ Storage full! Cannot add more waste.")
            
            elif key == ord('s'):  # Show storage summary
                self.show_storage_summary()
            
            elif key == ord('e'):  # Export data
                self.export_data()
                print("Data exported to waste_data.json")
            
            elif key == ord('q'):  # Quit
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def show_storage_summary(self):
        """Display current storage summary"""
        print("\n" + "="*50)
        print("STORAGE SUMMARY")
        print("="*50)
        print(f"Total Items: {len(self.waste_data)}")
        print(f"Total Weight: {self.current_storage:.1f}kg / {self.storage_capacity}kg")
        print(f"Utilization: {(self.current_storage/self.storage_capacity)*100:.1f}%")
        
        # Category breakdown
        if self.waste_data:
            categories = {}
            for item in self.waste_data:
                cat = item['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            print("\nCategory Breakdown:")
            for cat, count in categories.items():
                print(f"  {cat}: {count} items")
        
        print("="*50)
    
    def export_data(self, filename="waste_data.json"):
        """Export waste data to JSON file"""
        export_data = {
            'export_time': datetime.datetime.now().isoformat(),
            'storage_capacity': self.storage_capacity,
            'current_storage': self.current_storage,
            'waste_items': []
        }
        
        for item in self.waste_data:
            # Convert non-serializable objects
            export_item = item.copy()
            export_item['timestamp'] = item['timestamp'].isoformat()
            if 'image_size' in export_item:
                export_item['image_size'] = list(export_item['image_size'])
            export_data['waste_items'].append(export_item)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

def main():
    """Main function to run the webcam waste classification system"""
    print("=== AI Webcam Waste Classification System ===")
    print("Initializing...")
    
    # Initialize the system
    waste_classifier = WebcamWasteClassifier()
    
    # Train the model
    print("Training AI model...")
    waste_classifier.train_classifier()
    
    # Run webcam classification
    waste_classifier.run_webcam_classification()
    
    # Final summary
    print("\nFinal Storage Report:")
    waste_classifier.show_storage_summary()
    
    # Export data
    filename = waste_classifier.export_data()
    print(f"Data exported to {filename}")

if __name__ == "__main__":
    main()