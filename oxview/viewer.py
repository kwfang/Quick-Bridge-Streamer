"""
Rerun Visualization - Web Streaming + Keep-alive Mechanism

Core Logic:
1. rr.init() initializes recording
2. rr.serve_web_viewer(open_browser=True) starts local web server
3. Wait for browser connection
4. Stream data (with throttling for real-time viewing)
5. Keep script alive after streaming completes
"""

import time
import logging
from pathlib import Path

import rerun as rr

from .loader import BridgeDataLoader
from .analytics import VectorQualityRadar

logger = logging.getLogger(__name__)


class RerunApp:
    """
    Rerun Visualization App - Fully Automatic Web Streaming
    
    User Experience:
    Run script -> Browser opens automatically -> Data starts playing
    No file dragging needed, no rerun-cli installation required
    """

    def __init__(self, data_loader: BridgeDataLoader, limit: int = -1):
        self.data_loader = data_loader
        self.limit = limit
        self.analytics = VectorQualityRadar(history_size=10)
        
        # Statistics
        self.frame_count = 0
        self.episode_count = 0
        self.current_episode_id = -1

    def run(self):
        """
        Fully automatic run flow:
        1. Initialize Rerun + start gRPC server
        2. Start Web Viewer and connect
        3. Stream data
        4. Keep alive (don't exit)
        """
        # ========== Step 1: Initialize + gRPC Server ==========
        logger.info("Initializing Rerun...")
        rr.init("OpenX-Explorer")
        
        # Key: Start gRPC data server first
        # serve_grpc() returns URI that Web Viewer needs to connect to
        logger.info("Starting gRPC data server...")
        grpc_uri = rr.serve_grpc()
        logger.info(f"gRPC server: {grpc_uri}")
        
        # Start Web UI and connect to gRPC server
        logger.info("Starting Web Viewer server...")
        rr.serve_web_viewer(connect_to=grpc_uri, open_browser=True)
        
        logger.info("=" * 60)
        logger.info("Web Viewer started!")
        logger.info("Browser should open automatically.")
        logger.info(f"Data will stream to: {grpc_uri}")
        logger.info("=" * 60)
        
        # ========== Step 2: Wait for connection to stabilize ==========
        logger.info("Waiting 2 seconds for connection to stabilize...")
        time.sleep(2)
        
        # ========== Step 3: Stream data ==========
        logger.info("Streaming data to viewer...")
        self._stream_data()
        
        # ========== Step 4: Keep alive ==========
        logger.info("=" * 60)
        logger.info(f"Done! Streamed {self.frame_count} frames from {self.episode_count} episodes.")
        logger.info("Viewer is still running. Press Ctrl+C to exit.")
        logger.info("=" * 60)
        
        # Print PyTorch code
        self._print_farewell()
        
        # Keep alive: block main thread to prevent script exit
        # Script exit = Web server closes = Browser disconnects
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")

    def _stream_data(self):
        """Stream data to Web Viewer"""
        self.frame_count = 0
        self.episode_count = 0
        self.current_episode_id = -1
        episode_step = 0  # Step counter within each episode
        
        # Accumulate data for each episode
        current_episode_data = {
            'actions': [],  # Accumulate all actions for current episode
            'alerts_frozen': [],
            'alerts_spam': [],
        }
        
        for step in self.data_loader.stream(limit=self.limit):
            # Detect episode switch
            episode_id = int(step["episode_id"])
            if episode_id != self.current_episode_id:
                # Switch to new episode - first send previous episode's complete data
                if self.current_episode_id != -1 and current_episode_data['actions']:
                    self._send_episode_data(self.current_episode_id, current_episode_data)
                
                # Reset state
                self.current_episode_id = episode_id
                self.episode_count += 1
                episode_step = 0
                
                # Clear accumulated data
                current_episode_data = {
                    'actions': [],
                    'alerts_frozen': [],
                    'alerts_spam': [],
                }
                
                # Reset analytics
                self.analytics.reset()
                
                logger.info(f"[Episode {episode_id}] streaming...")
            
            # Accumulate current step data
            action = step["action"]
            current_episode_data['actions'].append(action)
            
            # Real-time analysis
            detection = self.analytics.update(action)
            current_episode_data['alerts_frozen'].append(1.0 if detection["is_static"] else 0.0)
            current_episode_data['alerts_spam'].append(1.0 if detection["is_gripper_spam"] else 0.0)
            
            # Send visualization data (using static paths, timeline advances naturally)
            # Key: Only send current step data each time, Rerun will accumulate automatically
            self._send_current_step(step, episode_step, detection)
            
            # ----- Statistics -----
            self.frame_count += 1
            
            # Progress log
            if self.frame_count % 10 == 0:
                logger.info(f"  {self.frame_count} frames streamed...")
            
            # Increment step count within episode
            episode_step += 1
            
            # Throttling
            time.sleep(0.05)
        
        # Send last episode's data
        if current_episode_data['actions']:
            self._send_episode_data(self.current_episode_id, current_episode_data)
        
        logger.info(f"Streaming complete: {self.frame_count} frames, {self.episode_count} episodes")
    
    def _send_current_step(self, step: dict, step_idx: int, detection: dict):
        """Send visualization data for current step"""
        episode_id = step["episode_id"]
        
        # Use episode-specific path prefix
        prefix = f"episode_{episode_id}"
        
        # 1. RGB image
        rr.log(f"{prefix}/world/camera", rr.Image(step["image_0"]))
        
        # 2. Language instruction
        rr.log(f"{prefix}/info/instruction", rr.TextDocument(step["language_instruction"]))
        
        # 3. Action curves - use Scalar single-point mode, Rerun will connect into curves
        action = step["action"]
        action_labels = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]
        for label, value in zip(action_labels, action):
            # Use Scalar to send single value, timeline determined by step_idx
            rr.log(f"{prefix}/control/{label}", rr.Scalars(float(value)))
        
        # 4. Alerts
        rr.log(f"{prefix}/alerts/robot_frozen", rr.Scalars(1.0 if detection["is_static"] else 0.0))
        rr.log(f"{prefix}/alerts/gripper_spam", rr.Scalars(1.0 if detection["is_gripper_spam"] else 0.0))
    
    def _send_episode_data(self, episode_id: int, data: dict):
        """Send complete episode data (for verifying data integrity)"""
        logger.debug(f"[Episode {episode_id}] data complete: {len(data['actions'])} steps")

    def _print_farewell(self):
        """Print exit message and PyTorch code snippet"""
        data_path = self.data_loader.data_path.absolute()
        
        print(f"""
{'='*60}
[SUCCESS] Data visualization complete!

  Frames:   {self.frame_count}
  Episodes: {self.episode_count}

{'='*60}

Need to load this data in PyTorch? Here is your snippet:
{'-'*60}
""")
        print(self._generate_pytorch_snippet(data_path))
        print(f"""{'-'*60}
Copy the above to start training!
""")

    def _generate_pytorch_snippet(self, data_path: Path) -> str:
        """Generate PyTorch Dataset code"""
        return f"""import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow.core.example import example_pb2
from pathlib import Path
import numpy as np


class BridgeDataV2Dataset(Dataset):
    \"\"\"BridgeData V2 PyTorch Dataset
    Auto-generated by OpenX-Explorer
    Data path: {data_path}
    \"\"\"
    
    def __init__(self, data_path="{data_path}", split="train"):
        self.data_path = Path(data_path)
        self.tfrecord_files = sorted(self.data_path.glob(f"*{{split}}.tfrecord-*"))
        
        # Build index: (file_path, episode_idx_in_file, step_idx, num_steps)
        self.index = []
        for fpath in self.tfrecord_files:
            ds = tf.data.TFRecordDataset(str(fpath))
            for ep_idx, proto in enumerate(ds):
                ex = example_pb2.Example()
                ex.ParseFromString(proto.numpy())
                n = len(ex.features.feature["steps/action"].float_list.value) // 7
                for s in range(n):
                    self.index.append((str(fpath), ep_idx, s, n))
        
        print(f"BridgeDataV2Dataset: {{len(self.index)}} steps indexed")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        fpath, ep_idx, step_idx, num_steps = self.index[idx]
        
        ds = tf.data.TFRecordDataset(fpath)
        for i, proto in enumerate(ds):
            if i == ep_idx:
                ex = example_pb2.Example()
                ex.ParseFromString(proto.numpy())
                
                action = np.array(
                    ex.features.feature["steps/action"].float_list.value,
                    dtype=np.float32
                ).reshape(num_steps, 7)[step_idx]
                
                img_bytes = ex.features.feature[
                    "steps/observation/image_0"
                ].bytes_list.value[step_idx]
                image = tf.io.decode_jpeg(img_bytes, channels=3).numpy()
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
                
                instruction = ex.features.feature[
                    "steps/language_instruction"
                ].bytes_list.value[step_idx].decode("utf-8")
                
                return {{
                    "image": torch.from_numpy(image),
                    "action": torch.from_numpy(action),
                    "instruction": instruction,
                }}
        
        raise IndexError(f"Could not find episode at index {{idx}}")


# Usage:
# dataset = BridgeDataV2Dataset()
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# for batch in loader:
#     print(batch["image"].shape, batch["action"].shape)
#     break
"""