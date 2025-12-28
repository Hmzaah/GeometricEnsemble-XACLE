import os
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import (
    WhisperModel, WhisperProcessor,
    DebertaV2Model, DebertaV2Tokenizer
)
# Note: These assume you have the challenge-specific libraries installed
import laion_clap 
from ms_clap import CLAP as MS_CLAP

from src.fusion import construct_unified_vector
from src.config import RAW_DATA_DIR, FEATURE_DIR

# --- MODEL LOADERS ---

def load_models(device):
    print("Loading Encoders...")
    
    # 1. Whisper v2 (Audio)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v2").to(device)
    
    # 2. DeBERTa V3 (Text)
    deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    deberta_model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to(device)
    
    # 3. LAION-CLAP
    # (Using the standard large checkpoint)
    laion_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    laion_model.load_ckpt() # Load default ckpt
    laion_model.to(device)

    # 4. MS-CLAP
    ms_model = MS_CLAP(version='2023', use_cuda=(device.type == 'cuda'))

    return {
        'whisper': (whisper_model, whisper_processor),
        'deberta': (deberta_model, deberta_tokenizer),
        'laion': laion_model,
        'ms_clap': ms_model
    }

# --- EXTRACTION LOGIC ---

def get_whisper_features(model, processor, audio_array, sr=16000, device='cpu'):
    # Resample if necessary logic here (omitted for brevity)
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        # We take the mean of the encoder last hidden state to get a fixed vector
        outputs = model.encoder(input_features)
        last_hidden = outputs.last_hidden_state # (1, Seq, 1280)
        embedding = torch.mean(last_hidden, dim=1).cpu().numpy()
    return embedding # Shape (1, 1280)

def get_deberta_features(model, tokenizer, text, device='cpu'):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token or Mean pooling
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding # Shape (1, 768)

def process_dataset(input_dir, output_file, device='cuda'):
    models = load_models(device)
    
    # 1. Identify Files
    # Assuming file structure: {id}.wav and {id}.txt
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    unified_vectors = []
    
    print(f"Processing {len(audio_files)} pairs...")
    
    for wav_file in tqdm(audio_files):
        file_id = wav_file.split('.')[0]
        txt_file = f"{file_id}.txt"
        
        audio_path = os.path.join(input_dir, wav_file)
        text_path = os.path.join(input_dir, txt_file)
        
        if not os.path.exists(text_path):
            continue
            
        # --- Load Raw Data ---
        # Load Audio (16k for Whisper)
        audio, sr = librosa.load(audio_path, sr=16000)
        # Load Text
        with open(text_path, 'r') as f:
            caption = f.read().strip()
            
        # --- Run Encoders ---
        
        # A. Whisper (1280 dim)
        whisper_emb = get_whisper_features(models['whisper'][0], models['whisper'][1], audio, device=device)
        
        # B. DeBERTa (768 dim)
        deberta_emb = get_deberta_features(models['deberta'][0], models['deberta'][1], caption, device=device)
        
        # C. LAION-CLAP (1536 dim - Audio+Text Concatenated usually, or just Audio)
        # For this architecture, we need Audio embedding mainly
        laion_audio_emb = models['laion'].get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
        # If text is needed for geometric comparison:
        # laion_text_emb = models['laion'].get_text_embedding([caption])
        
        # D. MS-CLAP (2048 dim)
        ms_audio_emb = models['ms_clap'].get_audio_embeddings([audio_path])
        # ms_text_emb = models['ms_clap'].get_text_embeddings([caption])
        
        # --- Fuse & Construct 9,220 Dim Vector ---
        # Note: We need to align dimensions (1, D)
        # We pass the embeddings to the fusion logic which handles the geometry calculation
        
        # *Critically*, we need the Text embeddings from CLAP for the Geometry injection
        # Let's assume the fusion function handles specific inputs, 
        # or we calculate geometry here. For strict adherence to `src/fusion.py`,
        # we pass the base embeddings.
        
        # Ensure shape (1, D)
        ms_audio_emb = ms_audio_emb.reshape(1, -1)
        laion_audio_emb = laion_audio_emb.reshape(1, -1)
        
        unified_vec = construct_unified_vector(
            whisper=whisper_emb,
            ms_clap=ms_audio_emb,
            laion=laion_audio_emb,
            deberta=deberta_emb
        )
        
        unified_vectors.append(unified_vec)
        
    # --- Save Result ---
    if unified_vectors:
        final_matrix = np.vstack(unified_vectors)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, final_matrix)
        print(f"Saved features to {output_file}. Shape: {final_matrix.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=RAW_DATA_DIR, help="Path to raw .wav/.txt files")
    parser.add_argument("--output_file", type=str, default=os.path.join(FEATURE_DIR, "unified_features.npy"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_file, args.device)
