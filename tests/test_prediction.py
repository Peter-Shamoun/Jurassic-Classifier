import unittest
import os
import sys
from io import BytesIO
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after adding to path
from dino import binary_to_matrix, resize_image, softmax, get_model_info

class TestPredictionUtils(unittest.TestCase):
    """Tests for the prediction utility functions in dino.py"""
    
    def setUp(self):
        """Create test image data to use in tests"""
        # Create a simple red test image
        self.test_img = Image.new('RGB', (100, 100), color='red')
        self.img_bytes = BytesIO()
        self.test_img.save(self.img_bytes, format='JPEG')
        self.img_bytes = self.img_bytes.getvalue()
        
        # Create a large test image
        self.large_img = Image.new('RGB', (5000, 5000), color='blue')
        self.large_img_bytes = BytesIO()
        self.large_img.save(self.large_img_bytes, format='JPEG')
        self.large_img_bytes = self.large_img_bytes.getvalue()
    
    def test_binary_to_matrix(self):
        """Test conversion of binary image data to numpy matrix"""
        matrix = binary_to_matrix(self.img_bytes)
        
        # Verify shape and content
        self.assertEqual(matrix.shape, (100, 100, 3))
        self.assertTrue(np.all(matrix[:, :, 0] > 250))  # Red channel should be high
        
    def test_binary_to_matrix_large_image(self):
        """Test that oversized images are rejected"""
        with self.assertRaises(ValueError):
            binary_to_matrix(self.large_img_bytes)
    
    def test_resize_image(self):
        """Test image resizing"""
        # Convert binary to matrix first
        matrix = binary_to_matrix(self.img_bytes)
        
        # Resize the matrix
        resized = resize_image(matrix)
        
        # Check output dimensions
        self.assertEqual(resized.shape, (256, 256, 3))
        
    def test_softmax(self):
        """Test softmax implementation"""
        # Create sample logits
        logits = np.array([2.0, 1.0, 0.1])
        
        # Apply softmax
        probs = softmax(logits)
        
        # Check properties
        self.assertEqual(probs.shape, logits.shape)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.argmax(probs) == np.argmax(logits))
    
    def test_get_model_info(self):
        """Test model info function"""
        info = get_model_info()
        
        # Verify structure and content
        self.assertIn('version', info)
        self.assertIn('classes', info)
        self.assertIn('class_names', info)
        self.assertEqual(len(info['class_names']), 15)  # Should be 15 dinosaur classes

if __name__ == '__main__':
    unittest.main() 