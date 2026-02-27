"""
Vector Analysis Engine - Ultra-fast anomaly detection based on Action vectors only
"""

import numpy as np
from typing import List, Dict, Deque
from collections import deque


class VectorQualityRadar:
    """
    Vector Quality Radar - Real-time data quality issue detection
    
    Analysis based solely on Action vector data (skips image analysis for performance)
    """
    
    def __init__(self, history_size: int = 10):
        """
        Args:
            history_size: Number of history frames for detecting gripper high-frequency switching
        """
        self.history_size = history_size
        self.gripper_history: Deque[float] = deque(maxlen=history_size)
        self.action_buffer: List[np.ndarray] = []
        self.frame_count = 0
    
    def reset(self):
        """Reset state"""
        self.gripper_history.clear()
        self.action_buffer = []
        self.frame_count = 0
    
    def update(self, action: np.ndarray) -> Dict[str, any]:
        """
        Update analyzer state and return detection results
        
        Args:
            action: 7-dimensional action vector
        
        Returns:
            Dictionary containing detection results
        """
        self.frame_count += 1
        
        # Extract gripper value (last element of action)
        gripper_value = action[-1]
        self.gripper_history.append(gripper_value)
        
        # Execute various detection methods
        is_static = self.detect_frozen_robot(action)
        is_spam = self.detect_gripper_spam()
        
        # Update action buffer for statistics
        self.action_buffer.append(action)
        if len(self.action_buffer) > 100:  # Keep last 100 frames
            self.action_buffer.pop(0)
        
        return {
            "is_static": is_static,
            "is_gripper_spam": is_spam,
            "gripper_value": gripper_value,
        }
    
    def detect_frozen_robot(self, action: np.ndarray, position_threshold: float = 1e-3, rotation_threshold: float = 1e-3) -> bool:
        """
        Detect if robot is in a static/invalid state
        
        BridgeData's Action is typically Delta EE Pose.
        If both position and rotation changes are below thresholds, robot is considered static.
        
        Args:
            action: 7-dimensional action vector
            position_threshold: Position change threshold
            rotation_threshold: Rotation change threshold
        
        Returns:
            True if robot static state detected
        """
        # Action structure: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper]
        # First 3 dims are position change, next 3 dims are rotation change
        
        position_delta = np.linalg.norm(action[:3])
        rotation_delta = np.linalg.norm(action[3:6])
        
        is_static = (position_delta < position_threshold) and (rotation_delta < rotation_threshold)
        
        return bool(is_static)
    
    def detect_gripper_spam(self, switch_threshold: float = 0.5, min_switches: int = 3) -> bool:
        """
        Detect if gripper is rapidly switching 0/1 in short time
        
        Args:
            switch_threshold: Threshold to consider as a switch
            min_switches: Minimum switch count to qualify as spam
        
        Returns:
            True if high-frequency gripper switching detected
        """
        if len(self.gripper_history) < self.history_size:
            return False
        
        # Convert to numpy array for computation
        history = np.array(self.gripper_history)
        
        # Calculate differences between adjacent frames
        diffs = np.abs(np.diff(history))
        
        # Count switches (diffs exceeding threshold)
        num_switches = np.sum(diffs > switch_threshold)
        
        is_spam = num_switches >= min_switches
        
        return bool(is_spam)
    
    def get_action_stats(self) -> Dict[str, np.ndarray]:
        """
        Calculate action statistics
        
        Used to check if Action has been normalized
        
        Returns:
            Dictionary containing min, max, mean
        """
        if not self.action_buffer:
            return {
                "min": np.zeros(7),
                "max": np.zeros(7),
                "mean": np.zeros(7),
                "std": np.zeros(7),
            }
        
        actions = np.array(self.action_buffer)
        
        return {
            "min": np.min(actions, axis=0),
            "max": np.max(actions, axis=0),
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
        }
    
    def get_summary(self) -> str:
        """Get analysis summary"""
        stats = self.get_action_stats()
        
        summary = f"""
Action Statistics (from last {len(self.action_buffer)} frames):
  Position (x,y,z):
    Range: [{stats['min'][:3].round(4)}, {stats['max'][:3].round(4)}]
    Mean:  {stats['mean'][:3].round(4)}
  Rotation (rx,ry,rz):
    Range: [{stats['min'][3:6].round(4)}, {stats['max'][3:6].round(4)}]
    Mean:  {stats['mean'][3:6].round(4)}
  Gripper:
    Range: [{stats['min'][6]:.4f}, {stats['max'][6]:.4f}]
    Mean:  {stats['mean'][6]:.4f}
"""
        return summary