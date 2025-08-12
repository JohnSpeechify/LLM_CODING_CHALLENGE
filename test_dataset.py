import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import librosa

def test_direct_download():
    """Test downloading FSDD dataset directly from GitHub"""
    
    print("🧪 Testing Direct FSDD Dataset Download")
    print("=" * 45)
    
    # Setup paths
    zip_filename = "fsdd_test.zip"
    extract_folder = "fsdd_test_data"
    github_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    
    # Download dataset
    recordings_path = Path(extract_folder) / "free-spoken-digit-dataset-master" / "recordings"
    
    if not recordings_path.exists():
        print("🌐 Downloading dataset from GitHub...")
        
        try:
            response = requests.get(github_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"📦 Download size: {total_size / (1024*1024):.1f} MB")
            
            # Download with progress
            with open(zip_filename, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  📈 Progress: {percent:.1f}%", end="", flush=True)
            
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            
            # Clean up zip
            os.remove(zip_filename)
            print("✅ Download complete!")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return [], []
    else:
        print("📁 Dataset already downloaded!")
    
    # Test loading audio files
    print(f"\n🔄 Testing audio file loading...")
    
    wav_files = list(recordings_path.glob("*.wav"))
    print(f"📊 Found {len(wav_files)} WAV files")
    
    if not wav_files:
        print(f"❌ No WAV files found in {recordings_path}")
        return [], []
    
    # Test loading first few files
    audio_data = []
    labels = []
    
    print(f"🧪 Testing first 20 audio files...")
    
    for i, wav_file in enumerate(wav_files[:20]):
        try:
            # Parse filename
            filename_parts = wav_file.stem.split('_')
            digit = int(filename_parts[0])
            speaker = filename_parts[1] if len(filename_parts) > 1 else "unknown"
            
            print(f"  📄 File: {wav_file.name}")
            print(f"      Digit: {digit}, Speaker: {speaker}")
            
            if 0 <= digit <= 9:
                # Load with librosa
                audio_array, sample_rate = librosa.load(str(wav_file), sr=None)
                
                print(f"      Audio: {len(audio_array)} samples at {sample_rate}Hz")
                print(f"      Duration: {len(audio_array)/sample_rate:.2f}s")
                
                # Validate
                if len(audio_array) > 0:
                    audio_data.append((audio_array, sample_rate))
                    labels.append(digit)
                    print(f"      ✅ Valid")
                else:
                    print(f"      ❌ Empty audio")
            else:
                print(f"      ❌ Invalid digit: {digit}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error with {wav_file.name}: {e}")
            continue
    
    print(f"✅ Successfully tested {len(audio_data)}/20 samples")
    
    # Show what we loaded
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Loaded digit distribution:")
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples")
    
    print(f"\n🎵 Sample audio info:")
    if audio_data:
        sample_audio, sample_sr = audio_data[0]
        print(f"  Shape: {sample_audio.shape}")
        print(f"  Sample rate: {sample_sr} Hz")
        print(f"  Duration: {len(sample_audio)/sample_sr:.2f} seconds")
        print(f"  Data type: {sample_audio.dtype}")
        print(f"  Value range: [{sample_audio.min():.3f}, {sample_audio.max():.3f}]")
    
    return audio_data, labels

def test_mfcc_extraction():
    """Test MFCC feature extraction on loaded audio"""
    
    print(f"\n🧪 Testing MFCC Feature Extraction")
    print("=" * 40)
    
    # Load some test data first
    audio_data, labels = test_direct_download()
    
    if not audio_data:
        print("❌ No audio data to test MFCC extraction")
        return
    
    # Test MFCC extraction on first sample
    sample_audio, sample_sr = audio_data[0]
    sample_label = labels[0]
    
    print(f"🎵 Testing MFCC on digit {sample_label}...")
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=sample_audio,
        sr=sample_sr,
        n_mfcc=13,
        hop_length=160,
        n_fft=400
    )
    
    print(f"✅ MFCC extraction successful!")
    print(f"   MFCC shape: {mfcc.shape}")
    print(f"   MFCC range: [{mfcc.min():.3f}, {mfcc.max():.3f}]")
    
    # Test fixed length processing
    max_frames = int(1.0 * 8000 / 160)  # 1 second at 8kHz
    print(f"   Target frames: {max_frames}")
    
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        print(f"   Padded to: {mfcc_padded.shape}")
    else:
        mfcc_padded = mfcc[:, :max_frames]
        print(f"   Truncated to: {mfcc_padded.shape}")
    
    print(f"✅ Feature extraction pipeline working!")
    
    return True

if __name__ == "__main__":
    print("🎯 FSDD Dataset Loading Test (Direct Download)")
    print("=" * 55)
    
    # Test dataset download and loading
    audio_data, labels = test_direct_download()
    
    if audio_data:
        print(f"\n🎉 Dataset loading successful!")
        print(f"✅ Ready for training with {len(audio_data)} samples")
        
        # Test MFCC extraction
        test_mfcc_extraction()
        
        print(f"\n🚀 All tests passed! You can now use the main training code.")
        
    else:
        print(f"\n❌ Dataset loading failed")
        print(f"Check your internet connection and try again")