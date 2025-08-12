import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import time
from collections import deque
import threading
import queue

# Same model architecture as training
class SimpleCNN(nn.Module):
    """Same model architecture - must match training code"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(13, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        
        return x

class LiveDigitRecognizer:
    """Real-time digit recognition using trained model"""
    
    def __init__(self, model_path='digit_model.pth'):
        # Same config as training
        self.sample_rate = 8000
        self.n_mfcc = 13
        self.max_length = 1.0
        
        # Load trained model
        print("🔄 Loading trained model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")
        
        # Audio streaming setup
        self.audio_queue = queue.Queue()
        self.recording_duration = 1.5  # Record for 1.5 seconds
        self.energy_threshold = 0.001  # Lower default threshold
        
        print("🎤 Live digit recognizer ready!")
        
        # Auto-calibrate microphone on startup
        self.calibrate_microphone()
    
    def extract_mfcc_features(self, audio_array):
        """Extract MFCC features (same as training)"""
        
        # Resample if needed
        if len(audio_array) == 0:
            return None
        
        # Extract MFCC using librosa (same parameters as training)
        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=160,
            n_fft=400
        )
        
        # Fixed length processing (same as training)
        max_frames = int(self.max_length * self.sample_rate / 160)
        if mfcc.shape[1] < max_frames:
            pad_width = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_frames]
        
        return mfcc
    
    def predict_digit(self, audio_array):
        """Predict digit from audio array"""
        
        # Extract features
        mfcc = self.extract_mfcc_features(audio_array)
        if mfcc is None:
            return None, 0.0, 0.0
        
        # Convert to tensor and add batch dimension
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        
        # Predict
        start_time = time.time()
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_digit = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # milliseconds
        
        return predicted_digit, confidence, inference_time
    
    def calibrate_microphone(self):
        """Auto-calibrate microphone sensitivity"""
        
        print("\n🎚️  Auto-calibrating microphone...")
        print("   Please speak a digit when prompted")
        
        input("   Press Enter to start calibration...")
        
        print("   Recording 2 seconds for calibration...")
        print("   📢 Speak 'one' or 'two' clearly now!")
        
        # Record calibration sample
        cal_recording = sd.rec(
            int(2.0 * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        cal_audio = cal_recording.flatten()
        cal_energy = np.mean(cal_audio ** 2)
        
        print(f"   Measured energy: {cal_energy:.6f}")
        
        if cal_energy > 0.00001:  # If we detected some audio
            # Set threshold to 30% of detected speech energy
            self.energy_threshold = cal_energy * 0.3
            print(f"   ✅ Auto-calibrated threshold to: {self.energy_threshold:.6f}")
        else:
            # Use very low threshold if no speech detected
            self.energy_threshold = 0.0001
            print(f"   ⚠️  No speech detected, using low threshold: {self.energy_threshold}")
        
        print(f"   🎤 Microphone calibration complete!")
    
    def detect_voice_activity(self, audio_chunk):
        """Simple energy-based voice activity detection"""
        energy = np.mean(audio_chunk ** 2)
        return energy > self.energy_threshold
    
    def record_and_predict(self):
        """Record audio and make prediction"""
        
        print(f"🎙️  Recording for {self.recording_duration} seconds...")
        print("   📢 Speak a digit (0-9) clearly into your microphone!")
        print("   📢 Say 'zero', 'one', 'two', etc.")
        
        # Countdown to help user
        for i in range(3, 0, -1):
            print(f"   Starting in {i}...")
            time.sleep(1)
        
        print("   🔴 RECORDING NOW!")
        
        # Record audio
        recording = sd.rec(
            int(self.recording_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        print("   ⏹️  Recording stopped")
        
        audio_array = recording.flatten()
        
        # Show audio level for debugging
        energy_level = np.mean(audio_array ** 2)
        max_amplitude = np.max(np.abs(audio_array))
        
        print(f"🔍 Audio analysis:")
        print(f"   Energy level: {energy_level:.6f} (threshold: {self.energy_threshold})")
        print(f"   Max amplitude: {max_amplitude:.3f}")
        print(f"   Audio length: {len(audio_array)} samples ({len(audio_array)/self.sample_rate:.2f}s)")
        
        # Check if speech was detected
        has_speech = self.detect_voice_activity(audio_array)
        
        if not has_speech:
            print("⚠️  No speech detected")
            print(f"   Your audio energy: {energy_level:.6f}")
            print(f"   Required threshold: {self.energy_threshold}")
            if energy_level > 0:
                suggested_threshold = energy_level * 0.5
                print(f"   💡 Try threshold: {suggested_threshold:.6f}")
            print(f"   💡 Or speak louder / closer to microphone")
            return None, 0.0, 0.0
        
        print("✅ Speech detected! Processing...")
        
        # Make prediction
        digit, confidence, inference_time = self.predict_digit(audio_array)
        
        return digit, confidence, inference_time
    
    def run_continuous_demo(self, num_predictions=5):
        """Run continuous prediction demo"""
        
        print(f"\n🎯 Continuous Recognition Demo")
        print(f"Will record {num_predictions} predictions")
        print("=" * 40)
        
        results = []
        
        for i in range(num_predictions):
            print(f"\n📍 Prediction {i+1}/{num_predictions}")
            
            # Wait for user
            input("Press Enter when ready to speak...")
            
            # Record and predict
            digit, confidence, inference_time = self.record_and_predict()
            
            if digit is not None:
                # Display result with confidence indicator
                if confidence > 0.9:
                    conf_icon = "🟢"  # High confidence
                elif confidence > 0.7:
                    conf_icon = "🟡"  # Medium confidence
                else:
                    conf_icon = "🔴"  # Low confidence
                
                print(f"\n🎯 RESULT: Digit {digit} {conf_icon}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Inference time: {inference_time:.1f}ms")
                
                results.append({
                    'digit': digit,
                    'confidence': confidence,
                    'inference_time': inference_time
                })
            else:
                print("❌ No prediction (audio processing failed)")
        
        # Summary
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_inference_time = np.mean([r['inference_time'] for r in results])
            
            print(f"\n📊 === DEMO SUMMARY ===")
            print(f"✅ Successful predictions: {len(results)}/{num_predictions}")
            print(f"✅ Average confidence: {avg_confidence:.3f}")
            print(f"✅ Average inference time: {avg_inference_time:.1f}ms")
            
            print(f"\n🔍 All predictions:")
            for i, result in enumerate(results, 1):
                conf_icon = "🟢" if result['confidence'] > 0.9 else "🟡" if result['confidence'] > 0.7 else "🔴"
                print(f"  {i}. Digit {result['digit']} {conf_icon} ({result['confidence']:.3f})")

def test_audio_setup():
    """Test microphone and audio setup"""
    
    print("🧪 Testing Audio Setup")
    print("=" * 30)
    
    try:
        # Check available audio devices
        print("🎤 Available audio devices:")
        print(sd.query_devices())
        
        # Test recording
        print(f"\n🎙️  Testing microphone...")
        print("   Recording 2 seconds of audio...")
        
        test_recording = sd.rec(
            int(2 * 8000),  # 2 seconds at 8kHz
            samplerate=8000,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        audio_level = np.mean(test_recording ** 2)
        print(f"✅ Recording successful!")
        print(f"   Audio energy level: {audio_level:.6f}")
        
        if audio_level < 0.001:
            print("⚠️  Audio level very low - check microphone")
        else:
            print("✅ Audio level good for recognition")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio setup failed: {e}")
        print("   Check microphone permissions and try again")
        return False

def main():
    """Main demo function"""
    
    print("🎤 Live Spoken Digit Recognition Demo")
    print("Using your trained model with 99.7% accuracy!")
    print("=" * 50)
    
    # Test audio setup first
    if not test_audio_setup():
        print("❌ Cannot proceed without working microphone")
        return
    
    # Load recognizer with your trained model
    try:
        recognizer = LiveDigitRecognizer('digit_model.pth')
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("Make sure 'digit_model.pth' exists in current directory")
        return
    
    while True:
        print(f"\n🎛️  Choose demo mode:")
        print("1. Single digit test")
        print("2. Continuous recognition (5 predictions)")
        print("3. Continuous recognition (10 predictions)")
        print("4. Adjust sensitivity")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
            
        elif choice == '1':
            print("\n🎯 Single Digit Test")
            print("📝 Instructions:")
            print("   1. Just press Enter (don't type anything)")
            print("   2. Wait for countdown")
            print("   3. Speak a digit like 'zero', 'one', 'two', etc.")
            print("   4. Speak clearly into your microphone")
            
            input("\n👆 Press Enter to start (don't type the digit)...")
            
            digit, confidence, inference_time = recognizer.record_and_predict()
            
            if digit is not None:
                conf_icon = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🔴"
                print(f"\n🎯 PREDICTED: {digit} {conf_icon}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Inference: {inference_time:.1f}ms")
                
                # Ask user if it was correct
                try:
                    actual = input(f"\n❓ Was the prediction correct? What digit did you say? (0-9 or 'skip'): ").strip()
                    if actual.isdigit() and 0 <= int(actual) <= 9:
                        actual_digit = int(actual)
                        correct = actual_digit == digit
                        print(f"   {'✅ Correct!' if correct else '❌ Incorrect'}")
                        if not correct:
                            print(f"   Actual: {actual_digit}, Predicted: {digit}")
                except:
                    pass
            else:
                print("❌ Try speaking louder or adjusting sensitivity (option 4)")
            
        elif choice == '2':
            recognizer.run_continuous_demo(5)
            
        elif choice == '3':
            recognizer.run_continuous_demo(10)
            
        elif choice == '4':
            current = recognizer.energy_threshold
            print(f"Current threshold: {current}")
            try:
                new_threshold = float(input("New threshold (0.001-0.1): "))
                if 0.001 <= new_threshold <= 0.1:
                    recognizer.energy_threshold = new_threshold
                    print(f"✅ Updated to {new_threshold}")
                else:
                    print("❌ Must be between 0.001 and 0.1")
            except:
                print("❌ Invalid number")
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()