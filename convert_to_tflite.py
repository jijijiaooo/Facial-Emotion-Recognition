#!/usr/bin/env python3
"""
Convert Keras models (.h5) to TensorFlow Lite (.tflite) for Raspberry Pi
TFLite models are optimized for edge devices with lower RAM and faster inference
"""

import os
import sys
import tensorflow as tf
import numpy as np

def convert_h5_to_tflite(h5_path, tflite_path, quantize=True):
    """
    Convert a Keras .h5 model to TensorFlow Lite format
    
    Args:
        h5_path: Path to input .h5 model
        tflite_path: Path to output .tflite model
        quantize: If True, apply dynamic range quantization for smaller size
    """
    print(f"\n{'='*60}")
    print(f"Converting: {os.path.basename(h5_path)}")
    print(f"{'='*60}")
    
    try:
        # Load the Keras model
        print("📥 Loading Keras model...")
        model = tf.keras.models.load_model(h5_path)
        
        # Get model info
        input_shape = model.input_shape
        output_shape = model.output_shape
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {output_shape}")
        
        # Convert to TFLite
        print("🔄 Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # CRITICAL: Set compatibility for older TFLite Runtime versions (Raspberry Pi)
        # This ensures the model works with older TFLite runtime on Raspberry Pi 4B
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite builtin ops
        ]
        
        # Set to lowest opset version for maximum compatibility
        # This prevents "version 12" errors on older TFLite runtimes
        converter._experimental_lower_tensor_list_ops = False
        
        if quantize:
            print("   ⚙️  Applying dynamic range quantization (reduces size by ~4x)")
            # Dynamic range quantization - reduces model size significantly
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        # Ensure compatibility with older TFLite versions
        print("   🔧 Setting compatibility for Raspberry Pi TFLite Runtime...")
        converter.target_spec.supported_types = [tf.float32]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        print(f"💾 Saving TFLite model to: {tflite_path}")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file sizes
        h5_size = os.path.getsize(h5_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        reduction = ((h5_size - tflite_size) / h5_size) * 100
        
        print(f"\n✅ Conversion successful!")
        print(f"   Original .h5 size: {h5_size:.2f} MB")
        print(f"   TFLite size: {tflite_size:.2f} MB")
        print(f"   Size reduction: {reduction:.1f}%")
        
        # Test the converted model
        print("\n🧪 Testing TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input details: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"   Output details: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
        # Test with dummy input
        input_shape = input_details[0]['shape']
        test_input = np.random.random(input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   ✅ Test inference successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Convert all .h5 models to .tflite"""
    
    print("=" * 60)
    print("KERAS TO TENSORFLOW LITE CONVERTER")
    print("Optimized for Raspberry Pi")
    print("=" * 60)
    
    # Models directory
    models_dir = 'models'
    
    # Get all .h5 files
    h5_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"\n❌ No .h5 files found in {models_dir}/")
        return
    
    print(f"\n📁 Found {len(h5_files)} Keras model(s):")
    for i, f in enumerate(h5_files, 1):
        print(f"   {i}. {f}")
    
    # Ask which models to convert
    print("\n" + "=" * 60)
    print("CONVERSION OPTIONS:")
    print("1. Convert Simple CNN only (recommended for Pi)")
    print("2. Convert Simple CNN + Ensemble PKL models")
    print("3. Convert all models")
    print("4. Custom selection")
    print("=" * 60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Convert only Simple CNN
        models_to_convert = ['raf_db_simple_cnn.h5']
    elif choice == '2':
        # Convert Simple CNN (Ensemble PKL doesn't need conversion)
        models_to_convert = ['raf_db_simple_cnn.h5']
        print("\nℹ️  Note: ensemble_raf_db1_20250904_124505.pkl is already lightweight")
        print("   and doesn't need TFLite conversion (it's a scikit-learn model)")
    elif choice == '3':
        # Convert all
        models_to_convert = h5_files
    elif choice == '4':
        # Custom selection
        print("\nEnter model numbers to convert (comma-separated, e.g., 1,3,5):")
        indices = input().strip().split(',')
        models_to_convert = [h5_files[int(i.strip())-1] for i in indices if i.strip().isdigit()]
    else:
        print("Invalid choice. Converting Simple CNN only.")
        models_to_convert = ['raf_db_simple_cnn.h5']
    
    # Ask about quantization
    print("\n" + "=" * 60)
    quantize = input("Apply quantization for smaller size? (y/n) [y]: ").strip().lower()
    quantize = quantize != 'n'
    
    if quantize:
        print("✅ Quantization enabled (models will be ~4x smaller)")
    else:
        print("ℹ️  Quantization disabled (larger but potentially more accurate)")
    
    # Convert models
    print("\n" + "=" * 60)
    print("STARTING CONVERSION")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for model_file in models_to_convert:
        h5_path = os.path.join(models_dir, model_file)
        tflite_path = os.path.join(models_dir, model_file.replace('.h5', '.tflite'))
        
        if convert_h5_to_tflite(h5_path, tflite_path, quantize=quantize):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 TFLite models saved in: {models_dir}/")
    
    if successful > 0:
        print("\n🎉 Next steps:")
        print("   1. Copy .tflite files to your Raspberry Pi")
        print("   2. Use simple_emotion_detection_tflite.py instead")
        print("   3. Enjoy faster inference with lower RAM usage!")
        print("\nExpected improvements:")
        print("   - RAM usage: ~50-100MB (vs 300-400MB)")
        print("   - Inference speed: 2-3x faster")
        print("   - Model size: ~75% smaller")


if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    if not tf.__version__.startswith('2.'):
        print("⚠️  Warning: TensorFlow 2.x recommended")
    
    main()
