"""
BridgeData V2 TFRecord Loader
Fixed version: Data is in standard Example format, not SequenceExample
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Iterator, Dict, List, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetScanner:
    """
    Scan data/ folder to discover all available datasets
    
    Supported directory structures:
    1. Multiple subfolders, each containing a dataset:
       data/
       ├── dataset_a/ (contains .tfrecord files)
       └── dataset_b/ (contains .tfrecord files)
    
    2. Flat structure where data/ folder itself is a dataset:
       data/
       ├── *.tfrecord-*
       ├── dataset_info.json
       └── features.json
    """
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
    
    def scan(self) -> List[Dict]:
        """
        Scan and return list of all available dataset information
        
        Returns:
            Each dict contains:
            - name: dataset name
            - path: dataset path
            - num_files: number of TFRecord files
            - has_info: whether dataset_info.json exists
            - description: dataset description (if available)
        """
        datasets = []
        
        # Case 1: Check datasets in subfolders
        for subdir in sorted(self.root_path.iterdir()):
            if subdir.is_dir():
                dataset_info = self._check_dataset_folder(subdir)
                if dataset_info:
                    datasets.append(dataset_info)
        
        # Case 2: If no subfolder datasets, check root directory itself
        if not datasets:
            dataset_info = self._check_dataset_folder(self.root_path)
            if dataset_info:
                datasets.append(dataset_info)
        
        return datasets
    
    def _check_dataset_folder(self, folder: Path) -> Optional[Dict]:
        """Check if specified folder is a valid dataset"""
        tfrecord_files = sorted(folder.glob("*.tfrecord-*"))
        if not tfrecord_files:
            return None
        
        info = {
            "name": folder.name,
            "path": folder,
            "num_files": len(tfrecord_files),
            "has_info": (folder / "dataset_info.json").exists(),
            "description": "",
        }
        
        # Try to read dataset_info.json for more information
        info_path = folder / "dataset_info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info["description"] = data.get("description", "")
                    info["name"] = data.get("name", folder.name)
            except Exception as e:
                logger.warning(f"Failed to read {info_path}: {e}")
        
        return info
    
    def select_dataset(self, datasets: List[Dict], interactive: bool = True) -> Optional[Dict]:
        """
        Let user select which dataset to load
        
        Args:
            datasets: list of datasets
            interactive: whether to prompt user for input in terminal
        
        Returns:
            Selected dataset info, or None if failed
        """
        if not datasets:
            logger.error("No datasets found!")
            return None
        
        if len(datasets) == 1:
            ds = datasets[0]
            logger.info(f"Auto-selected only available dataset: {ds['name']}")
            return ds
        
        # Print dataset list
        print("\n" + "=" * 60)
        print("Available Datasets:")
        print("=" * 60)
        for i, ds in enumerate(datasets, 1):
            desc = f" - {ds['description'][:50]}..." if ds['description'] else ""
            print(f"  [{i}] {ds['name']} ({ds['num_files']} files){desc}")
        print("=" * 60 + "\n")
        
        if not interactive:
            # Non-interactive mode, default to first dataset
            return datasets[0]
        
        # Interactive selection
        while True:
            try:
                choice = input(f"Select dataset (1-{len(datasets)}, default=1): ").strip()
                if not choice:
                    choice = "1"
                idx = int(choice) - 1
                if 0 <= idx < len(datasets):
                    return datasets[idx]
                else:
                    print(f"Invalid choice. Please enter 1-{len(datasets)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                return None


class BridgeDataLoader:
    """
    BridgeData V2 Data Loader
    
    Actual data format:
    - Standard tf.train.Example (not SequenceExample)
    - Data stored as flattened lists
    - Each episode has 28 steps (from features.json)
    """

    def __init__(self, data_path: Path, split: str = "train"):
        self.data_path = Path(data_path)
        self.split = split
        self.tfrecord_files = self._find_tfrecord_files()
        
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.data_path}")
        
        logger.info(f"Found {len(self.tfrecord_files)} TFRecord files for split '{split}'")
        
        # Number of steps per episode (determined from testing to be 28)
        self.steps_per_episode = 28

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files"""
        pattern = f"*{self.split}.tfrecord-*"
        files = sorted(self.data_path.glob(pattern))
        return files
    
    def _parse_episode(self, proto: tf.Tensor) -> Dict:
        """
        Parse Episode
        
        Data stored as flattened lists, needs reshaping
        Uses protobuf for direct parsing to handle variable-length data
        """
        from tensorflow.core.example import example_pb2
        
        example = example_pb2.Example()
        example.ParseFromString(proto.numpy())
        
        # Get episode ID
        episode_id = example.features.feature["episode_metadata/episode_id"].int64_list.value[0]
        
        # Get action and determine number of steps
        action_flat = example.features.feature["steps/action"].float_list.value
        num_steps = len(action_flat) // 7
        
        # Parse all data
        action = np.array(action_flat, dtype=np.float32).reshape(num_steps, 7)
        state = np.array(example.features.feature["steps/observation/state"].float_list.value, dtype=np.float32).reshape(num_steps, 7)
        is_first = np.array(example.features.feature["steps/is_first"].int64_list.value, dtype=np.int64)
        is_last = np.array(example.features.feature["steps/is_last"].int64_list.value, dtype=np.int64)
        is_terminal = np.array(example.features.feature["steps/is_terminal"].int64_list.value, dtype=np.int64)
        reward = np.array(example.features.feature["steps/reward"].float_list.value, dtype=np.float32)
        discount = np.array(example.features.feature["steps/discount"].float_list.value, dtype=np.float32)
        
        # Get image and instruction lists
        image_list = list(example.features.feature["steps/observation/image_0"].bytes_list.value)
        instruction_list = list(example.features.feature["steps/language_instruction"].bytes_list.value)
        
        return {
            "episode_id": episode_id,
            "action": action,
            "state": state,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "reward": reward,
            "discount": discount,
            "image_list": image_list,
            "instruction_list": instruction_list,
            "num_steps": num_steps,
        }
    
    def stream(self, limit: int = -1) -> Iterator[Dict]:
        """
        Create data stream
        
        Pipeline:
        1. TFRecordDataset (read episodes)
        2. _parse_episode (parse episodes)
        3. Break down into steps
        4. Add images and instructions
        """
        files = [str(f) for f in self.tfrecord_files]
        
        # Manual iteration to add images and instructions
        for file_path in files:
            ds = tf.data.TFRecordDataset(file_path)
            
            for proto in ds:
                # Parse episode
                episode = self._parse_episode(proto)
                episode_id = episode["episode_id"]
                num_steps = episode["num_steps"]
                
                # Break down into steps
                for step_idx in range(num_steps):
                    step = {
                        "episode_id": episode_id,
                        "action": episode["action"][step_idx],
                        "state": episode["state"][step_idx],
                        "is_first": episode["is_first"][step_idx],
                        "is_last": episode["is_last"][step_idx],
                        "is_terminal": episode["is_terminal"][step_idx],
                        "reward": episode["reward"][step_idx],
                        "discount": episode["discount"][step_idx],
                        "step_idx": step_idx,
                    }
                    
                    # Add image
                    image_bytes = episode["image_list"][step_idx]
                    step["image_0"] = tf.io.decode_jpeg(image_bytes, channels=3).numpy()
                    
                    # Add language instruction
                    step["language_instruction"] = episode["instruction_list"][step_idx].decode('utf-8')
                    
                    yield step
                    
                    # Check limit
                    limit -= 1
                    if limit == 0:
                        return
    
    def _to_numpy(self, step: Dict[str, tf.Tensor]) -> Dict:
        """Convert TensorFlow tensors to NumPy arrays"""
        result = {}
        for key, value in step.items():
            if isinstance(value, tf.Tensor):
                result[key] = value.numpy()
            else:
                result[key] = value
        return result
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            "num_tfrecord_files": len(self.tfrecord_files),
            "data_path": str(self.data_path),
            "split": self.split,
            "steps_per_episode": self.steps_per_episode,
        }