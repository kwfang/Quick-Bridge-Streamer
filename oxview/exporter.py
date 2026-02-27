"""
PyTorch Dataset Code Generator
"""

from pathlib import Path


def generate_pytorch_snippet(data_path: Path, split: str = "train") -> str:
    """
    Generate PyTorch Dataset code snippet
    
    Args:
        data_path: Absolute path to dataset
        split: Data split, default "train"
    
    Returns:
        Executable Python code string
    """
    snippet = f'''import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from pathlib import Path
import numpy as np


class BridgeDataV2Dataset(Dataset):
    """
    BridgeData V2 PyTorch Dataset
    
    Auto-generated for: {data_path}
    """
    
    def __init__(self, data_path: str, split: str = "{split}"):
        self.data_path = Path(data_path)
        self.split = split
        self.tfrecord_files = sorted(self.data_path.glob(f"{{split}}.tfrecord-*"))
        
        # Sequence feature definitions
        self.sequence_features = {{
            "steps/observation/image_0": tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/language_instruction": tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/action": tf.io.FixedLenSequenceFeature([7], tf.float32),
            "steps/observation/state": tf.io.FixedLenSequenceFeature([7], tf.float32),
            "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64),
            "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64),
        }}
        
        self.context_features = {{
            "episode_metadata/episode_id": tf.io.FixedLenFeature([], tf.int64),
        }}
        
        # Build index: list all episodes
        self.episodes = []
        for file in self.tfrecord_files:
            ds = tf.data.TFRecordDataset(str(file))
            for proto in ds:
                context, sequence = tf.io.parse_sequence_example(
                    proto,
                    context_features=self.context_features,
                    sequence_features=self.sequence_features
                )
                num_steps = tf.shape(sequence["steps/action"])[0].numpy()
                for step_idx in range(num_steps):
                    self.episodes.append((str(file), step_idx))
                break  # Only parse first example per file to get metadata
        
        print(f"Loaded {{len(self.episodes)}} steps from {{len(self.tfrecord_files)}} files")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        file_path, step_idx = self.episodes[idx]
        
        # Parse specific episode
        ds = tf.data.TFRecordDataset(file_path)
        for proto in ds:
            context, sequence = tf.io.parse_sequence_example(
                proto,
                context_features=self.context_features,
                sequence_features=self.sequence_features
            )
            
            # Extract specific step
            image_bytes = sequence["steps/observation/image_0"][step_idx]
            image = tf.io.decode_jpeg(image_bytes, channels=3)
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
            image = torch.from_numpy(image.numpy()).permute(2, 0, 1)  # HWC -> CHW
            
            action = torch.from_numpy(sequence["steps/action"][step_idx].numpy())
            
            instruction = sequence["steps/language_instruction"][step_idx].numpy().decode("utf-8")
            
            return {{
                "image": image,
                "action": action,
                "instruction": instruction
            }}
        
        raise IndexError(f"Could not load index {{idx}}")


# Usage example
if __name__ == "__main__":
    dataset = BridgeDataV2Dataset("{data_path}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {{batch_idx}}:")
        print(f"  Images: {{batch['image'].shape}}")
        print(f"  Actions: {{batch['action'].shape}}")
        print(f"  Instructions: {{batch['instruction'][:2]}}")  # First 2 instructions
        
        if batch_idx >= 2:  # Only show 3 batches
            break
    
    print("\\\\nDataset loaded successfully! Ready for training.")
'''
    return snippet


def print_export_message(data_path: Path):
    """Print export success message and code snippet"""
    print("""
[SUCCESS] Session ended.
Need to load this data in PyTorch? Here is your snippet:
---------------------------------------------------------
""")
    print(generate_pytorch_snippet(data_path.absolute()))
    print("""---------------------------------------------------------
Copy the above to start training!
""")