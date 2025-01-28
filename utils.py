from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence
import re
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import random
import yaml
import librosa
from scipy import signal


class BenchmarkDataset:

    """
    A class for preparing multilingual speech recognition benchmark datasets.

    This class handles the loading, preprocessing, and validation of speech datasets. 
    It performs the following key tasks:
    - Loads datasets from a YAML configuration file
    - Filters out samples containing digits or exceeding maximum duration
    - Resamples audio to target sampling rate
    - Combines multiple datasets into a single benchmark dataset

    Args:
      config_path (str): Path to YAML configuration file containing:
          - general settings (seed, sampling rate, etc.)
          - dataset paths and parameters
          - preprocessing options

    Attributes:
      config (dict): Loaded configuration settings
      SEED (int): Random seed for reproducibility
      TARGET_SR (int): Target sampling rate for audio
      samples_per_dataset (int): Number of samples to select from each dataset
      max_duration_seconds (float): Maximum allowed audio duration
      datasets_config (list): Configuration for each dataset
      final_dataset (Dataset): Processed and combined final dataset
      datasets (dict): Dictionary of loaded datasets

    Methods:
      load_config(config_path: str) -> dict:
          Loads and returns configuration from YAML file.

      get_datasets() -> None:
          Loads all datasets specified in configuration into self.datasets.

      resample_audio(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
          Resamples audio to target sampling rate if needed.

      has_digits(text: str) -> bool:
          Checks if text contains any digits.

      get_audio_duration(audio_array: np.ndarray, sampling_rate: int) -> float:
          Calculates duration of audio in seconds.

      prepare_dataset(dataset: Dataset, dataset_name: str) -> Dataset:
          Processes single dataset by filtering and preprocessing samples.

      prepare_all_datasets() -> None:
          Processes and combines all datasets into final shuffled dataset.

    Example:
      >>> benchmark = BenchmarkDataset('config.yaml')
      >>> print(benchmark.final_dataset)
      Dataset({
          features: ['id', 'raw_transcription', 'transcription', 'audio', 'dataset_name'],
          num_rows: 600
      })
    """

    def __init__(self, config_path:str):
        self.config = self.load_config(config_path)
        self.SEED = self.config['general']['seed']
        self.TARGET_SR = self.config['general']['target_sampling_rate']
        self.samples_per_dataset = self.config['general']['samples_per_dataset']
        self.max_duration_seconds = self.config['general']['max_duration_seconds']
        self.datasets_config = self.config['datasets']
        self.final_dataset = None
        self.datasets = {}
        self.get_datasets()
        self.prepare_all_datasets()

    def load_config(self, config_path:str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_datasets(self):
        for data in tqdm(self.datasets_config, desc='Dataset List from config'):
            name = data['path']
            self.datasets[name] = load_dataset(**data)

    def resample_audio(self, waveform, orig_sr):
        if orig_sr != self.TARGET_SR:
            waveform = torch.tensor(waveform, dtype=torch.float32)
            resampler = torchaudio.transforms.Resample(orig_sr, self.TARGET_SR)
            waveform = resampler(waveform)
            return waveform.numpy().astype(np.float32)
        return waveform.astype(np.float32)

    def has_digits(self, text):
        return any(char.isdigit() for char in text)

    def get_audio_duration(self, audio_array, sampling_rate):
        return len(audio_array) / sampling_rate

    
    def prepare_dataset(self, dataset, dataset_name):
        batch_size = min(self.samples_per_dataset * 4, len(dataset))
        check_indices = np.random.choice(len(dataset), batch_size, replace=False).astype(np.int32)
        all_valid_indices = []
        
        for idx in tqdm(check_indices, desc=f"Checking samples from {dataset_name}"):
            sample = dataset[int(idx)]
            
            if self.has_digits(sample['transcription']):
                continue
                
            audio_duration = self.get_audio_duration(
                sample['audio']['array'], 
                sample['audio']['sampling_rate']
            )
            if audio_duration > self.max_duration_seconds:
                continue
                
            all_valid_indices.append(idx)

            if len(all_valid_indices) >= self.samples_per_dataset:
                break
                
        selected_samples = []
        
        for idx in tqdm(all_valid_indices, desc=f"Processing samples from {dataset_name}"):
            sample = dataset[int(idx)]
            audio_data = sample['audio']
            
            audio_array = np.array(audio_data['array'], dtype=np.float32)
            
            if audio_data['sampling_rate'] != self.TARGET_SR:
                audio_array = self.resample_audio(
                    audio_array,
                    audio_data['sampling_rate']
                )
            
            audio_data = {
                'array': audio_array,
                'path': audio_data['path'],
                'sampling_rate': self.TARGET_SR
            }
            
            new_sample = {
                'id': sample['id'],
                'raw_transcription': sample['raw_transcription'],
                'transcription': sample['transcription'],
                'audio': audio_data,
                'dataset_name': dataset_name
            }
            selected_samples.append(new_sample)
        
        return Dataset.from_list(selected_samples)


    def prepare_all_datasets(self):
        prepared_datasets = []
        
        for name, dataset in self.datasets.items():
            prepared_dataset = self.prepare_dataset(dataset, name)
            prepared_datasets.append(prepared_dataset)
        
        final_dataset = concatenate_datasets(prepared_datasets)
        self.final_dataset = final_dataset.shuffle(seed=self.SEED)


class AugmentedBenchmarkDataset(BenchmarkDataset):

    """
    A class for creating an augmented speech recognition benchmark dataset.
    Inherits from BenchmarkDataset and adds various audio augmentation capabilities.

    This class extends BenchmarkDataset by adding:
    - Noise mixing with different SNR levels
    - Phone effect simulation
    - Parallel processing of augmentations
    - Pushing augmented dataset to Hugging Face Hub

    Args:
      config_path (str): Path to YAML configuration file containing:
          - All BenchmarkDataset configurations
          - Noise files configuration
          - Augmentation parameters
          - Save settings for Hugging Face Hub

    Attributes:
      Inherits all attributes from BenchmarkDataset plus:
      noises (list): List of noise configurations and arrays
      dataset_length (int): Length of original dataset
      aug_final_dataset (Dataset): Final augmented dataset
      to_aug_indices (np.ndarray): Indices of samples to augment

    Methods:
      load_noise_files() -> None:
          Loads noise files specified in config and stores them in memory.

      mix_with_noise(audio_array: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
          Mixes audio with noise at specified SNR level.

      apply_phone_filter(audio: np.ndarray) -> np.ndarray:
          Applies bandpass filter to simulate phone audio.

      add_phone_compression(audio: np.ndarray) -> np.ndarray:
          Adds compression effect to simulate phone audio.

      add_phone_noise(audio: np.ndarray) -> np.ndarray:
          Adds characteristic phone noise.

      apply_phone_effect(audio: np.ndarray) -> np.ndarray:
          Combines all phone effects into one transformation.

      augment_dataset(dataset: Dataset) -> None:
          Creates augmented versions of selected samples.

      push_HF() -> None:
          Pushes augmented dataset to Hugging Face Hub.

    Example:
      >>> aug_benchmark = AugmentedBenchmarkDataset('config.yaml')
      >>> aug_benchmark.push_HF()  # Pushes to Hugging Face Hub

    Notes:
      - The class expects noise files and configuration in the YAML file
      - Augmentations include both noise mixing and phone effect simulation
      - Augmentation is applied only to a subset of samples specified by aug_size
    """


    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.noises = self.config['noise']
        self.load_noise_files()
        self.dataset_lenght = len(self.final_dataset)
        self.aug_final_dataset = self.final_dataset.add_column("augmentation_type", 
                                                                ["original"] * self.dataset_lenght)
        self.aug_final_dataset = self.aug_final_dataset.add_column("snr", [0.0] * self.dataset_lenght)
        self.to_aug_indices = np.random.choice(self.dataset_lenght,
                                               self.config['general']['aug_size'],
                                               replace=False).astype(np.int32)

        self.augment_dataset(self.final_dataset)

    def load_noise_files(self)->None:
        print('________ Loading Noise Files _______')
        for file_ in self.noises:
            noise, _ = librosa.load(file_['path'], sr=self.TARGET_SR)
            file_['array'] = noise

    def mix_with_noise(self, audio_array: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
        audio_length = len(audio_array)
        
        while len(noise) < audio_length:
            noise = np.concatenate([noise, noise])
        
        max_start = len(noise) - audio_length - 1
        start_idx = np.random.randint(0, max_start)
        noise_segment = noise[start_idx:start_idx + audio_length]
        
        audio_energy = np.linalg.norm(audio_array)
        noise_energy = np.linalg.norm(noise)
        alpha = audio_energy / noise_energy

        mixed = audio_array + snr * alpha * noise_segment
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val 
        return mixed

    def apply_phone_filter(self, audio: np.ndarray)-> np.ndarray:
        nyquist = self.TARGET_SR / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            filtered_audio = filtered_audio / max_val
        return filtered_audio

    def add_phone_compression(self, audio: np.ndarray) -> np.ndarray:
        threshold = 0.5
        ratio = 1.0
        
        mask = np.abs(audio) > threshold
        audio[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio * np.sign(audio[mask])
        return audio

    def add_phone_noise(self, audio: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 0.005, len(audio))
        filtered_noise = self.apply_phone_filter(noise)
        return audio + filtered_noise

    def apply_phone_effect(self, audio: np.ndarray) -> np.ndarray:
        audio = self.apply_phone_filter(audio)
        audio = self.add_phone_compression(audio)
        audio = self.add_phone_noise(audio)
        audio = audio / np.max(np.abs(audio))
        return audio


    def augment_dataset(self, dataset):

        features = Features({
            'id': Value(dtype='int64'),
            'raw_transcription': Value(dtype='string'),
            'transcription': Value(dtype='string'),
            'dataset_name': Value(dtype='string'),
            'augmentation_type': Value(dtype='string'),
            'snr': Value(dtype='float64'),
            'audio': {
                'array': Sequence(feature=Value(dtype='float32'), length=-1),
                'path': Value(dtype='string'),
                'sampling_rate': Value(dtype='int64')
            }
        })

        for ind_ in tqdm(self.to_aug_indices, desc="Augmentating original samples"):
            sample = dataset[int(ind_)]
            augmented_samples = []
            
            for noise_data in self.noises:
                noise_array = noise_data['array']
                noise_name = noise_data['name']
                
                for snr in noise_data['snr_levels']:
                    aug_sample = {
                        'id': sample['id'],
                        'raw_transcription': sample['raw_transcription'],
                        'transcription': sample['transcription'],
                        'dataset_name': sample['dataset_name'],
                        'augmentation_type': noise_name,
                        'snr': snr,
                        'audio': {
                            'array': self.mix_with_noise(sample['audio']['array'], 
                                                      noise_array, 
                                                      snr),
                            'sampling_rate': self.TARGET_SR,
                            'path': sample['audio']['path']
                        }
                    }
                    augmented_samples.append(aug_sample)

            phone_sample = {
                        'id': sample['id'],
                        'raw_transcription': sample['raw_transcription'],
                        'transcription': sample['transcription'],
                        'dataset_name': sample['dataset_name'],
                        'augmentation_type': 'phone',
                        'snr': 0.0,
                        'audio': {
                            'array': self.apply_phone_effect(sample['audio']['array']),
                            'sampling_rate': self.TARGET_SR,
                            'path': sample['audio']['path']
                        }
                    }
            augmented_samples.append(phone_sample)

            chunk = Dataset.from_list(augmented_samples, features=features)
            self.aug_final_dataset = concatenate_datasets([self.aug_final_dataset, chunk])

    def push_HF(self)->None:
        dataset_repo_name = self.config['save_dataset']['repo_dataset_name']
        private = self.config['save_dataset']['private']
        self.aug_final_dataset.push_to_hub(dataset_repo_name, private=private)
