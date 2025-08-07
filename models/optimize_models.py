#!/usr/bin/env python3
"""
Model Optimization Script for Raspberry Pi Deployment
Converts your existing .h5 model to optimized formats
"""

import os
import numpy as np
import cv2

def convert_to_tflite():
    """Convert H5 model to TensorFlow Lite for edge deployment"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print("Loading original model...")
        model = keras.models.load_model('model_file_30epochs.h5')
        
        print("Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Basic optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open('emotion_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("✅ TensorFlow Lite model saved as 'emotion_model.tflite'")
        
        # Test the converted model
        test_tflite_model()
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow Lite conversion failed: {e}")
        return False

def convert_to_quantized_tflite():
    """Convert to quantized TF Lite for maximum optimization"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print("Loading original model for quantization...")
        model = keras.models.load_model('model_file_30epochs.h5')
        
        # Representative dataset for quantization
        def representative_dataset_gen():
            # Load some sample images from your training data
            sample_dir = "data/train/happy"  # Use any emotion folder
            samples = []
            
            if os.path.exists(sample_dir):
                for img_file in os.listdir(sample_dir)[:100]:  # Use 100 samples
                    if img_file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(sample_dir, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            resized = cv2.resize(img, (48, 48))
                            normalized = resized.astype(np.float32) / 255.0
                            samples.append(normalized.reshape(1, 48, 48, 1))
            
            # If no training data, generate synthetic data
            if not samples:
                print("No training data found, using synthetic data for quantization")
                for _ in range(100):
                    synthetic = np.random.random((1, 48, 48, 1)).astype(np.float32)
                    samples.append(synthetic)
            
            for sample in samples:
                yield [sample]
        
        print("Converting to quantized TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Quantization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert
        quantized_tflite_model = converter.convert()
        
        # Save
        with open('emotion_model_quantized.tflite', 'wb') as f:
            f.write(quantized_tflite_model)
        
        print("✅ Quantized TensorFlow Lite model saved as 'emotion_model_quantized.tflite'")
        return True
        
    except Exception as e:
        print(f"❌ Quantized TensorFlow Lite conversion failed: {e}")
        return False

def convert_to_onnx():
    """Convert to ONNX format for cross-platform optimization"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        import tf2onnx
        
        print("Loading model for ONNX conversion...")
        model = keras.models.load_model('model_file_30epochs.h5')
        
        print("Converting to ONNX...")
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        
        # Save
        with open('emotion_model.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print("✅ ONNX model saved as 'emotion_model.onnx'")
        return True
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        print("Install tf2onnx: pip install tf2onnx")
        return False

def test_tflite_model():
    """Test the converted TensorFlow Lite model"""
    try:
        import tflite_runtime.interpreter as tflite
        
        # Load model
        interpreter = tflite.Interpreter(model_path="emotion_model.tflite")
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TF Lite model test successful")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Test with dummy data
        dummy_input = np.random.random((1, 48, 48, 1)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        predicted_emotion = emotions[np.argmax(output[0])]
        confidence = np.max(output[0])
        
        print(f"   Test prediction: {predicted_emotion} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"❌ TF Lite model test failed: {e}")

def benchmark_models():
    """Benchmark different model formats for performance comparison"""
    import time
    
    print("\n🚀 Benchmarking model performance...")
    
    # Test data
    test_input = np.random.random((1, 48, 48, 1)).astype(np.float32)
    num_runs = 100
    
    results = {}
    
    # Test original Keras model
    try:
        from tensorflow import keras
        model = keras.models.load_model('model_file_30epochs.h5')
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.predict(test_input, verbose=0)
        keras_time = (time.time() - start_time) / num_runs
        results['Keras'] = keras_time * 1000  # Convert to ms
        
    except Exception as e:
        print(f"Keras benchmark failed: {e}")
    
    # Test TensorFlow Lite
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path="emotion_model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        start_time = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        tflite_time = (time.time() - start_time) / num_runs
        results['TensorFlow Lite'] = tflite_time * 1000  # Convert to ms
        
    except Exception as e:
        print(f"TF Lite benchmark failed: {e}")
    
    # Test ONNX
    try:
        import onnxruntime as ort
        session = ort.InferenceSession("emotion_model.onnx")
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {'input_1': test_input})
        onnx_time = (time.time() - start_time) / num_runs
        results['ONNX'] = onnx_time * 1000  # Convert to ms
        
    except Exception as e:
        print(f"ONNX benchmark failed: {e}")
    
    # Print results
    print("\n📊 Performance Results (average inference time):")
    print("-" * 50)
    for model_type, time_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model_type:15}: {time_ms:.2f} ms")
    
    if results:
        fastest = min(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]:.2f} ms)")

def get_model_sizes():
    """Check file sizes of different model formats"""
    print("\n📏 Model File Sizes:")
    print("-" * 30)
    
    models = [
        ('Original H5', 'model_file_30epochs.h5'),
        ('TensorFlow Lite', 'emotion_model.tflite'),
        ('Quantized TF Lite', 'emotion_model_quantized.tflite'),
        ('ONNX', 'emotion_model.onnx')
    ]
    
    for name, filename in models:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"{name:15}: {size_mb:.2f} MB")
        else:
            print(f"{name:15}: Not found")

def main():
    print("🔧 Model Optimization for Raspberry Pi")
    print("=" * 50)
    
    if not os.path.exists('model_file_30epochs.h5'):
        print("❌ Original model 'model_file_30epochs.h5' not found!")
        return
    
    # Convert models
    print("\n1. Converting to TensorFlow Lite...")
    convert_to_tflite()
    
    print("\n2. Converting to Quantized TensorFlow Lite...")
    convert_to_quantized_tflite()
    
    print("\n3. Converting to ONNX...")
    convert_to_onnx()
    
    # Show file sizes
    get_model_sizes()
    
    # Benchmark performance
    benchmark_models()
    
    print("\n✅ Model optimization complete!")
    print("\nRecommendations for Raspberry Pi:")
    print("- Use TensorFlow Lite for best performance")
    print("- Use Quantized TF Lite for smallest size")
    print("- Use ONNX for cross-platform compatibility")

if __name__ == "__main__":
    main()
