# Speech Recognition Dataset Augmentation Tool

This tool creates a speech recognition benchmark dataset with various audio augmentations.

## Installation

```bash
# Clone repository
git clone [repository-url]
cd [repository-name]

# Install requirements
pip install -r requirements.txt
```

## Authentication Setup

Before pushing datasets to Hugging Face Hub, you need to setup authentication:

```bash
# Configure git credentials
git config --global credential.helper store

# Login to Hugging Face
huggingface-cli login
```
When prompted, enter your Hugging Face token (can be found at https://huggingface.co/settings/tokens)

## Configuration

The tool uses a YAML configuration file. Here's a description of each section:

### General Settings
```yaml
general:
  seed: 42                     # Random seed for reproducibility
  target_sampling_rate: 16000  # Target audio sampling rate
  samples_per_dataset: 200     # Number of samples to take from each dataset
  max_duration_seconds: 30     # Maximum allowed audio duration
  aug_size: 100               # Number of samples to augment
```

### Dataset Configuration
```yaml
datasets:
  - path: "google/fleurs"      # HF dataset path
    name: "ky_kg"             # Dataset name (if required)
    split: "train"            # Dataset split

  - path: "custom/dataset"     
    split: "train"
```

### Noise Configuration
```yaml
noise:
  - name: 'white_noise'        # Noise type name
    path: '/path/to/noise.wav' # Path to noise file
    snr_levels: [0.1, 0.4, 0.6, 0.8, 1.0]  # SNR levels for mixing

  - name: 'ambient'
    path: '/path/to/ambient.wav'
    snr_levels: [0.5, 1.0, 2.0, 5.0, 10.0]
```

### Save Configuration (Optional)
```yaml
save_dataset:
  repo_dataset_name: 'username/dataset-name'  # HF repository name
  private: false                              # Repository visibility
```

## Running the Tool

Basic usage:
```bash
python main.py
```

With custom config:
```bash
python main.py --config path/to/config.yaml
```

## Output

The tool creates a dataset with the following features:
- Original audio samples
- Augmented versions with different noise types
- Phone effect simulation
- Additional metadata (SNR levels, augmentation types)

### Dataset Structure
```python
Dataset({
    features: [
        'id': int64,
        'raw_transcription': string,
        'transcription': string,
        'dataset_name': string,
        'augmentation_type': string,
        'snr': float64,
        'audio': {
            'array': float32[],
            'path': string,
            'sampling_rate': int64
        }
    ]
})
```

## Notes

- The tool first creates a benchmark dataset from the specified sources
- Then applies augmentations to a subset of samples (specified by aug_size)
- Augmentations include noise mixing and phone effect simulation
- If save_dataset configuration is present, pushes the result to Hugging Face Hub

## Troubleshooting

Common issues:
1. Authentication errors:
   - Make sure you've run the login command
   - Check your token permissions

2. Memory issues:
   - Reduce samples_per_dataset
   - Reduce aug_size
   - Processing is done in chunks to manage memory

3. Audio file issues:
   - Ensure noise files exist at specified paths
   - Check supported audio formats (wav, m4a)

For more help, please create an issue in the repository.
