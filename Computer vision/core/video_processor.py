"""
Video processing utilities
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Callable, Optional


class VideoProcessor:
    """
    Video processing class for detection
    """
    
    def __init__(self, video_path: Path):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        
        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FPS counter
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
        print(f"✓ Video loaded: {video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total Frames: {self.total_frames}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame
        
        Returns:
            Frame as numpy array, or None if end of video
        """
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            
            # Update FPS counter
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.current_fps = self.frame_count / elapsed
        
        return frame if ret else None
    
    def get_fps(self) -> float:
        """Get current processing FPS"""
        return self.current_fps
    
    def get_progress(self) -> float:
        """Get processing progress (0.0 to 1.0)"""
        if self.total_frames == 0:
            return 0.0
        return self.frame_count / self.total_frames
    
    def reset(self):
        """Reset video to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0
        self.start_time = time.time()
    
    def process(
        self,
        frame_callback: Callable[[np.ndarray, int], np.ndarray],
        display_size: tuple = None,
        window_name: str = "Video Processing",
        skip_frames: int = 0
    ):
        """
        Process video with callback function
        
        Args:
            frame_callback: Function to process each frame
                            Signature: f(frame, frame_num) -> processed_frame
            display_size: (width, height) for display, None to keep original
            window_name: OpenCV window name
            skip_frames: Skip N frames between processing (for speed)
        """
        print(f"\n🚀 Starting video processing...")
        print(f"Press 'q' or 'ESC' to quit\n")
        
        frame_skip_counter = 0
        
        while True:
            frame = self.get_frame()
            
            if frame is None:
                break
            
            # Skip frames if requested
            if skip_frames > 0:
                frame_skip_counter += 1
                if frame_skip_counter <= skip_frames:
                    continue
                frame_skip_counter = 0
            
            # Process frame
            try:
                processed_frame = frame_callback(frame, self.frame_count)
            except Exception as e:
                print(f"Error processing frame {self.frame_count}: {e}")
                processed_frame = frame
            
            # Resize for display if requested
            if display_size:
                processed_frame = cv2.resize(processed_frame, display_size)
            
            # Show frame
            cv2.imshow(window_name, processed_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
        
        # Cleanup
        self.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {self.frame_count} frames")
        print(f"Average FPS: {self.get_fps():.2f}")
    
    def save_video(
        self,
        output_path: Path,
        frame_callback: Callable[[np.ndarray, int], np.ndarray],
        codec: str = 'mp4v',
        fps: int = None
    ):
        """
        Save processed video to file
        
        Args:
            output_path: Output video path
            frame_callback: Function to process each frame
            codec: Video codec (mp4v, xvid, etc.)
            fps: Output FPS, None to use input FPS
        """
        fps = fps or self.fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.frame_width, self.frame_height)
        )
        
        print(f"\n💾 Saving video to: {output_path}")
        
        self.reset()
        
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            
            # Process frame
            processed_frame = frame_callback(frame, self.frame_count)
            
            # Write frame
            out.write(processed_frame)
            
            # Show progress
            if self.frame_count % 100 == 0:
                progress = self.get_progress()
                print(f"Progress: {progress:.1%}", end='\r')
        
        # Cleanup
        out.release()
        self.release()
        
        print(f"\n✓ Video saved: {output_path}")
    
    def release(self):
        """Release video capture"""
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
