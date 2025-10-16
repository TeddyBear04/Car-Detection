"""
Visualization utilities for Car Detection Project
"""

import cv2
import numpy as np
from typing import Tuple, List


def draw_box_with_label(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Draw bounding box with label on image
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        label: Label text
        color: Box color (BGR)
        thickness: Box thickness
        text_color: Text color (BGR)
    
    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    cv2.rectangle(
        image,
        (x1, y1 - label_h - 10),
        (x1 + label_w, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2
    )
    
    return image


def draw_info_panel(
    image: np.ndarray,
    info_dict: dict,
    position: str = 'top-left',
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draw information panel on image
    
    Args:
        image: Input image
        info_dict: Dictionary of info to display
        position: Panel position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
        alpha: Background transparency
    
    Returns:
        Image with info panel
    """
    h, w = image.shape[:2]
    
    # Calculate panel size
    max_text_width = 0
    line_height = 30
    padding = 10
    
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        max_text_width = max(max_text_width, text_w)
    
    panel_width = max_text_width + 2 * padding
    panel_height = len(info_dict) * line_height + 2 * padding
    
    # Determine position
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x, y = w - panel_width - 10, 10
    elif position == 'bottom-left':
        x, y = 10, h - panel_height - 10
    else:  # bottom-right
        x, y = w - panel_width - 10, h - panel_height - 10
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_width, y + panel_height),
        bg_color,
        -1
    )
    image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # Draw text
    text_y = y + padding + 20
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(
            image,
            text,
            (x + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )
        text_y += line_height
    
    return image


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw FPS counter on image
    
    Args:
        image: Input image
        fps: FPS value
        position: Text position
        color: Text color (BGR)
    
    Returns:
        Image with FPS counter
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    return image


def create_grid_view(images: List[np.ndarray], grid_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Create grid view from multiple images
    
    Args:
        images: List of images
        grid_size: Grid size (rows, cols). Auto-calculated if None
    
    Returns:
        Grid image
    """
    if not images:
        return None
    
    n = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Resize all images to same size
    resized_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (max_w, max_h))
        resized_images.append(resized)
    
    # Fill remaining slots with black images
    while len(resized_images) < rows * cols:
        resized_images.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_images = resized_images[i * cols:(i + 1) * cols]
        grid_row = np.hstack(row_images)
        grid_rows.append(grid_row)
    
    grid = np.vstack(grid_rows)
    return grid


def add_watermark(
    image: np.ndarray,
    text: str = "Car Detection System",
    position: str = 'bottom-right',
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (200, 200, 200),
    thickness: int = 1
) -> np.ndarray:
    """
    Add watermark text to image
    
    Args:
        image: Input image
        text: Watermark text
        position: Position ('bottom-left', 'bottom-right')
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
    
    Returns:
        Image with watermark
    """
    h, w = image.shape[:2]
    
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    if position == 'bottom-right':
        x = w - text_w - 10
        y = h - 10
    else:  # bottom-left
        x = 10
        y = h - 10
    
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return image
