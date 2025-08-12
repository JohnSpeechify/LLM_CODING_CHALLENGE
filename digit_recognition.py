import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
import requests
import zipfile
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Simple configuration
class Config:
    SAMPLE_RATE = 8000
    N_MFCC = 13
    MAX_LENGTH = 1.0  # seconds
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30

class SimpleDigitDataset(Dataset):
    """Dataset using only librosa - completely avoiding torch audio issues"""
    
    def __init__(self, audio_data, labels):
        self.audio_data = audio_data  # List of (audio_array, sample_rate)
        self.labels = labels
        self.config = Config()
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio_array, sr = self.audio_data[idx]
        label = self.labels[idx]
        
        # Resample if needed using librosa
        if sr != self.config.SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
        
        # Extract MFCC features using librosa
        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=self.config.SAMPLE_RATE,
            n_mfcc=self.config.N_MFCC,
            hop_length=160,
            n_fft=400
        )
        
        # Fixed length processing (pad or truncate)
        max_frames = int(self.config.MAX_LENGTH * self.config.SAMPLE_RATE / 160)
        if mfcc.shape[1] < max_frames:
            # Pad with zeros
            pad_width = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfcc = mfcc[:, :max_frames]
        
        # Convert to tensor
        mfcc_tensor = torch.FloatTensor(mfcc)
        
        return mfcc_tensor, label

class SimpleCNN(nn.Module):
    """Simple 1D CNN for digit recognition"""
    
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

def download_fsdd_dataset():
    """Download FSDD dataset directly from GitHub - NO torch dependencies"""
    
    print("📥 Downloading FSDD dataset directly from GitHub...")
    
    # Setup paths
    zip_filename = "fsdd_dataset.zip"
    extract_folder = "fsdd_data"
    github_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    
    # Download if not already present
    recordings_path = Path(extract_folder) / "free-spoken-digit-dataset-master" / "recordings"
    
    if not recordings_path.exists():
        print("🌐 Downloading from GitHub...")
        
        try:
            response = requests.get(github_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"📦 Download size: {total_size / (1024*1024):.1f} MB")
            
            with open(zip_filename, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
            
            print("\n📦 Extracting files...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            
            # Clean up
            os.remove(zip_filename)
            print("✅ Download and extraction complete!")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return [], []
    else:
        print("📁 Dataset already exists!")
    
    # Load audio files using librosa
    print("🔄 Loading audio files with librosa...")
    
    audio_data = []
    labels = []
    
    wav_files = list(recordings_path.glob("*.wav"))
    print(f"📊 Found {len(wav_files)} WAV files")
    
    for i, wav_file in enumerate(wav_files):
        try:
            # Parse filename: digit_speaker_instance.wav
            filename_parts = wav_file.stem.split('_')
            digit = int(filename_parts[0])
            
            if 0 <= digit <= 9:
                # Load audio with librosa (no torch dependencies)
                audio_array, sample_rate = librosa.load(str(wav_file), sr=None)
                
                # Validate audio
                if len(audio_array) > 0:
                    audio_data.append((audio_array, sample_rate))
                    labels.append(digit)
                
                if (i + 1) % 200 == 0:
                    print(f"  ✅ Loaded {i + 1}/{len(wav_files)} files")
            
        except Exception as e:
            print(f"  ⚠️ Skipping {wav_file.name}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(audio_data)} audio samples")
    
    # Show distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 Digit distribution:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} samples")
    
    # Sample info
    if audio_data:
        sample_audio, sample_sr = audio_data[0]
        print(f"\n🎵 Sample info:")
        print(f"  Sample rate: {sample_sr} Hz")
        print(f"  Duration: {len(sample_audio)/sample_sr:.2f} seconds")
        print(f"  Audio range: [{sample_audio.min():.3f}, {sample_audio.max():.3f}]")
    
    return audio_data, labels

def create_data_loaders(audio_data, labels, test_size=0.2):
    """Create train/test data loaders"""
    
    if len(audio_data) == 0:
        print("❌ No audio data to create loaders")
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        audio_data, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SimpleDigitDataset(X_train, y_train)
    test_dataset = SimpleDigitDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"✅ Train set: {len(train_dataset)} samples")
    print(f"✅ Test set: {len(test_dataset)} samples")
    
    return train_loader, test_loader

def train_model(model, train_loader, device):
    """Train the model"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"\n🚀 Starting training on {device}...")
    
    model.train()
    
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1:2d}/{Config.NUM_EPOCHS}: '
                  f'Loss: {avg_loss:.4f}, '
                  f'Accuracy: {accuracy:.4f}')
    
    print("✅ Training completed!")
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    
    model.eval()
    all_preds = []
    all_targets = []
    inference_times = []
    
    print("📊 Evaluating model...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            inference_times.append((end_time - start_time) / data.size(0))
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    avg_inference_time = np.mean(inference_times) * 1000
    
    print(f"\n🎉 === RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Inference Time: {avg_inference_time:.1f} ms per sample")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(all_targets, all_preds, 
                              target_names=[f"Digit {i}" for i in range(10)]))
    
    return accuracy, avg_inference_time

def predict_single_audio(model, audio_array, sample_rate, device):
    """Predict digit from audio array"""
    
    config = Config()
    
    # Resample if needed
    if sample_rate != config.SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=config.SAMPLE_RATE)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio_array,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        hop_length=160,
        n_fft=400
    )
    
    # Fixed length processing
    max_frames = int(config.MAX_LENGTH * config.SAMPLE_RATE / 160)
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    
    # Convert to tensor and predict
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(mfcc_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_digit = output.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    return predicted_digit, confidence

def main():
    """Main function using direct download only"""
    
    print("🎯 Simple Spoken Digit Recognition (Direct Download)")
    print("=" * 55)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download dataset directly (no HuggingFace dependencies)
    audio_data, labels = download_fsdd_dataset()
    
    if not audio_data:
        print("❌ Failed to load dataset")
        return
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(audio_data, labels)
    
    if train_loader is None:
        return
    
    # Create and train model
    model = SimpleCNN()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {param_count:,} parameters")
    
    # Train
    start_time = time.time()
    model = train_model(model, train_loader, device)
    training_time = time.time() - start_time
    
    # Save model
    torch.save(model.state_dict(), 'digit_model.pth')
    print(f"💾 Model saved as 'digit_model.pth'")
    
    # Evaluate
    accuracy, inference_time = evaluate_model(model, test_loader, device)
    
    # Test single prediction
    if audio_data:
        sample_audio, sample_sr = audio_data[0]
        sample_label = labels[0]
        predicted_digit, confidence = predict_single_audio(model, sample_audio, sample_sr, device)
        
        print(f"\n🧪 Sample test:")
        print(f"True: {sample_label}, Predicted: {predicted_digit}, Confidence: {confidence:.3f}")
    
    print(f"\n🎉 === SUMMARY ===")
    print(f"✅ Training time: {training_time/60:.1f} minutes")
    print(f"✅ Test accuracy: {accuracy:.4f}")
    print(f"✅ Inference time: {inference_time:.1f}ms")

if __name__ == "__main__":
    main()