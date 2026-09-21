import os
import wfdb
import numpy as np
from tqdm import tqdm

# AAMI Mapping based on physionet annotation symbols
# N: Normal beat
# S: Supraventricular ectopic beat
# V: Ventricular ectopic beat
# F: Fusion beat
# Q: Unknown beat
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # SVEB
    'V': 2, 'E': 2,                          # VEB
    'F': 3,                                  # Fusion
    '/': 4, 'f': 4, 'Q': 4                   # Unknown
}

# Patient-independent splits (DS1 and DS2)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 104, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234]

RAW_DATA_DIR = 'mit-bih-arrhythmia-database-1.0.0'
PROCESSED_DATA_DIR = 'data/processed'

# Beat segment parameters
BEFORE = 90  # samples before R-peak
AFTER = 90   # samples after R-peak
SEQ_LEN = BEFORE + AFTER

def load_and_segment_beats(records, data_dir):
    X, y, RR = [], [], []
    for record_id in tqdm(records, desc="Processing Records"):
        record_path = os.path.join(data_dir, str(record_id))
        
        if not os.path.exists(record_path + '.dat'):
            print(f"Skipping {record_id} - not found.")
            continue
            
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        signal = record.p_signal[:, 0]  # Use first channel (usually MLII)
        
        # Get R-peaks and labels
        r_peaks = annotation.sample
        symbols = annotation.symbol
        
        for i, (peak, symbol) in enumerate(zip(r_peaks, symbols)):
            if symbol in AAMI_MAPPING:
                # Check boundaries
                if peak - BEFORE >= 0 and peak + AFTER < len(signal):
                    segment = signal[peak - BEFORE : peak + AFTER]
                    X.append(segment)
                    y.append(AAMI_MAPPING[symbol])
                    
                    # RR features
                    pre_rr = (peak - r_peaks[i-1]) / 360.0 if i > 0 else 0.0
                    post_rr = (r_peaks[i+1] - peak) / 360.0 if i < len(r_peaks) - 1 else 0.0
                    
                    # Local average RR (last 10)
                    start_idx = max(0, i - 10)
                    if i > start_idx:
                        local_rr = (peak - r_peaks[start_idx]) / (i - start_idx) / 360.0
                    else:
                        local_rr = 0.0
                    
                    RR.append([pre_rr, post_rr, local_rr])
                    
    return np.array(X), np.array(y), np.array(RR)

if __name__ == '__main__':
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    print("Extracting Train Set (DS1)...")
    X_train, y_train, RR_train = load_and_segment_beats(DS1, RAW_DATA_DIR)
    
    print("Extracting Test Set (DS2)...")
    X_test, y_test, RR_test = load_and_segment_beats(DS2, RAW_DATA_DIR)
    
    print(f"Train set: X shape {X_train.shape}, y shape {y_train.shape}, RR shape {RR_train.shape}")
    print(f"Test set: X shape {X_test.shape}, y shape {y_test.shape}, RR shape {RR_test.shape}")
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'RR_train.npy'), RR_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'RR_test.npy'), RR_test)
    
    print(f"Data saved to {PROCESSED_DATA_DIR}")
