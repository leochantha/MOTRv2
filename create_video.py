import cv2
import numpy as np
import os
import glob
import json
import sys
from collections import defaultdict
import argparse
import math

def detect_format(file_path):
    """
    Detect whether the file is in CSV format or JSON format
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if it's JSON format
    if first_line.startswith('{') or first_line.startswith('"'):
        return 'json'
    else:
        return 'csv'

def parse_tracking_data(txt_file):
    """
    Parse the tracking data from txt file
    Supports both CSV format and JSON format
    Returns a dictionary with frame_number as key and list of bounding boxes as value
    """
    format_type = detect_format(txt_file)
    
    if format_type == 'json':
        return parse_json_format(txt_file)
    else:
        return parse_csv_format(txt_file)

def parse_json_format(json_file):
    """
    Parse JSON format tracking data
    Format: {"image_path": ["x,y,width,height,confidence", ...], ...}
    """
    tracking_data = defaultdict(list)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for image_path, detections in data.items():
        # Extract frame number from image path
        # Assumes format like "volleyball/test/test1/img1/000001"
        frame_num = extract_frame_number(image_path)
        
        for i, detection_str in enumerate(detections):
            # Clean the detection string - remove quotes, newlines, and whitespace
            detection_str = detection_str.strip().strip('\n').strip('"').strip("'")
            if not detection_str:
                continue
                
            parts = detection_str.split(',')
            if len(parts) >= 4:
                try:
                    # Clean each part and convert to float
                    top_left_x = float(parts[0].strip().strip('"').strip("'"))
                    top_left_y = float(parts[1].strip().strip('"').strip("'"))
                    width = float(parts[2].strip().strip('"').strip("'"))
                    height = float(parts[3].strip().strip('"').strip("'"))
                    confidence = float(parts[4].strip().strip('"').strip("'")) if len(parts) > 4 else 1.0
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse detection '{detection_str}': {e}")
                    continue
                
                # Use detection index as object_id since no ID is provided
                object_id = i
                
                x1 = int(top_left_x)
                y1 = int(top_left_y)
                x2 = int(top_left_x + width)
                y2 = int(top_left_y + height)
                
                tracking_data[frame_num].append({
                    'object_id': object_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'center': (int(top_left_x + width/2), int(top_left_y + height/2))
                })
    
    return tracking_data

def extract_frame_number(image_path):
    """
    Extract frame number from image path
    Handles various naming conventions
    """
    # Split by '/' and get the last part
    filename = image_path.split('/')[-1]
    
    # Try to extract number from filename
    import re
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Take the last number found
    else:
        # If no numbers found, try to get from path
        path_numbers = re.findall(r'\d+', image_path)
        if path_numbers:
            return int(path_numbers[-1])
        else:
            return 1  # Default to frame 1

def parse_csv_format(txt_file):
    """
    Parse CSV format tracking data
    Format: frame_number,object_id,x,y,width,height,confidence,...
    """
    tracking_data = defaultdict(list)
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 6:
                frame_num = int(parts[0])
                object_id = int(parts[1])
                top_left_x = float(parts[2])
                top_left_y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                
                # Coordinates are already top-left corner positions
                x1 = int(top_left_x)
                y1 = int(top_left_y)
                x2 = int(top_left_x + width)
                y2 = int(top_left_y + height)
                
                tracking_data[frame_num].append({
                    'object_id': object_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'center': (int(top_left_x + width/2), int(top_left_y + height/2))
                })
    
    return tracking_data

def get_color_for_id(object_id):
    """
    Generate a consistent color for each object ID
    """
    np.random.seed(object_id)
    color = np.random.randint(0, 255, 3)
    return tuple(map(int, color))

def draw_bounding_boxes(image, detections, show_ids=True, show_confidence=False, display_mode='auto'):
    """
    Draw bounding boxes on the image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        show_ids: Whether to show object IDs
        show_confidence: Whether to show confidence scores
        display_mode: 'ids', 'confidence', or 'auto' (default)
    """
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        object_id = detection['object_id']
        confidence = detection['confidence']
        center = detection['center']
        
        # Get consistent color for this object ID
        color = get_color_for_id(object_id)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(image, center, 3, color, -1)
        
        # Prepare label text based on display mode
        label_parts = []
        
        if display_mode == 'confidence':
            # Show only confidence score
            label_parts.append(f"{confidence:.3f}")
        elif display_mode == 'ids':
            # Show only track ID
            label_parts.append(f"ID:{object_id}")
        else:  # 'auto' mode - use the function parameters
            if show_ids:
                label_parts.append(f"ID:{object_id}")
            if show_confidence:
                label_parts.append(f"{confidence:.3f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get text size for background rectangle (increased font scale to 1.0 and thickness to 2)
            font_scale = 1.0
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Draw text with bigger font
            cv2.putText(
                image, 
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
    
    return image

def resize_frame(frame, target_width, target_height):
    """
    Resize frame to target dimensions while maintaining aspect ratio
    """
    h, w = frame.shape[:2]
    
    # Calculate scaling factor to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create canvas with target dimensions and center the resized frame
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate position to center the frame
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    return canvas

def add_panel_label(frame, label, position='top'):
    """
    Add a label to identify the tracking model
    """
    h, w = frame.shape[:2]
    
    # Create label background
    label_height = 40
    label_bg = np.zeros((label_height, w, 3), dtype=np.uint8)
    label_bg[:] = (50, 50, 50)  # Dark gray background
    
    # Add text
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    text_x = (w - text_width) // 2
    text_y = (label_height + text_height) // 2
    
    cv2.putText(
        label_bg,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    # Combine with frame
    if position == 'top':
        combined = np.vstack([label_bg, frame])
    else:  # bottom
        combined = np.vstack([frame, label_bg])
    
    return combined

def create_comparison_video(frames_dir, tracking_files, model_names, output_video, 
                          fps=30, show_ids=True, show_confidence=False, 
                          frame_format='jpg', frame_name_pattern=None,
                          grid_layout=None):
    """
    Create a comparison video with multiple tracking models side by side
    
    Args:
        frames_dir: Directory containing individual frame images
        tracking_files: List of paths to tracking data files
        model_names: List of names for each tracking model
        output_video: Output video file path
        fps: Frames per second for output video
        show_ids: Whether to show object IDs
        show_confidence: Whether to show confidence scores
        frame_format: Format of frame images (jpg, png, etc.)
        frame_name_pattern: Custom pattern for frame naming
        grid_layout: Tuple (rows, cols) for grid layout, auto-calculated if None
    """
    
    num_models = len(tracking_files)
    if len(model_names) != num_models:
        raise ValueError("Number of model names must match number of tracking files")
    
    # Auto-calculate grid layout if not provided
    if grid_layout is None:
        if num_models <= 2:
            grid_layout = (1, num_models)
        elif num_models <= 4:
            grid_layout = (2, 2)
        elif num_models <= 6:
            grid_layout = (2, 3)
        elif num_models <= 9:
            grid_layout = (3, 3)
        else:
            # For more than 9 models, use a square-ish grid
            rows = int(math.ceil(math.sqrt(num_models)))
            cols = int(math.ceil(num_models / rows))
            grid_layout = (rows, cols)
    
    rows, cols = grid_layout
    print(f"Using {rows}x{cols} grid layout for {num_models} models")
    
    # Parse all tracking data
    print("Parsing tracking data for all models...")
    all_tracking_data = []
    for i, tracking_file in enumerate(tracking_files):
        print(f"  Parsing {model_names[i]}...")
        tracking_data = parse_tracking_data(tracking_file)
        all_tracking_data.append(tracking_data)
    
    # Get list of frame files
    if frame_name_pattern:
        # Use custom pattern - get frame numbers from first tracking file
        frame_files = []
        frame_nums = sorted(all_tracking_data[0].keys()) if all_tracking_data[0] else []
        for frame_num in frame_nums:
            frame_file = os.path.join(frames_dir, frame_name_pattern.format(frame_num))
            if os.path.exists(frame_file):
                frame_files.append(frame_file)
    else:
        # Auto-detect frame files
        frame_pattern = os.path.join(frames_dir, f"*.{frame_format}")
        frame_files = sorted(glob.glob(frame_pattern))
        
        if not frame_files:
            # Try common frame naming patterns
            patterns = [
                f"frame_*.{frame_format}",
                f"*_*.{frame_format}",
                f"*.{frame_format}"
            ]
            
            for pattern in patterns:
                frame_pattern = os.path.join(frames_dir, pattern)
                frame_files = sorted(glob.glob(frame_pattern))
                if frame_files:
                    break
    
    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir} with format {frame_format}")
    
    print(f"Found {len(frame_files)} frame files")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")
    
    original_height, original_width = first_frame.shape[:2]
    print(f"Original frame dimensions: {original_width}x{original_height}")
    
    # Calculate panel dimensions
    panel_width = original_width // cols
    panel_height = (original_height + 40) // rows  # +40 for label space
    
    # Calculate output video dimensions
    output_width = panel_width * cols
    output_height = panel_height * rows
    
    print(f"Panel dimensions: {panel_width}x{panel_height}")
    print(f"Output video dimensions: {output_width}x{output_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))
    
    # Process each frame
    processed_frames = 0
    for i, frame_file in enumerate(frame_files):
        if frame_name_pattern:
            # Frame number from tracking data keys
            frame_nums = sorted(all_tracking_data[0].keys()) if all_tracking_data[0] else []
            if i < len(frame_nums):
                frame_num = frame_nums[i]
            else:
                frame_num = i + 1
        else:
            # Extract frame number from filename or use index
            import re
            numbers = re.findall(r'\d+', os.path.basename(frame_file))
            if numbers:
                frame_num = int(numbers[-1])
            else:
                frame_num = i + 1
        
        # Read original frame
        original_frame = cv2.imread(frame_file)
        if original_frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        # Create grid canvas
        grid_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Process each model
        for model_idx in range(num_models):
            # Calculate grid position
            row = model_idx // cols
            col = model_idx % cols
            
            # Create a copy of the original frame
            frame_copy = original_frame.copy()
            
            # Get detections for this frame and model
            detections = all_tracking_data[model_idx].get(frame_num, [])
            
            # Draw bounding boxes
            if detections:
                frame_copy = draw_bounding_boxes(frame_copy, detections, show_ids, show_confidence)
            
            # Add frame number to frame
            cv2.putText(
                frame_copy,
                f"Frame: {frame_num}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Add detection count
            cv2.putText(
                frame_copy,
                f"Detections: {len(detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Resize frame to fit panel (accounting for label space)
            resized_frame = resize_frame(frame_copy, panel_width, panel_height - 40)
            
            # Add model label
            labeled_frame = add_panel_label(resized_frame, model_names[model_idx], 'top')
            
            # Place in grid
            start_y = row * panel_height
            end_y = start_y + panel_height
            start_x = col * panel_width
            end_x = start_x + panel_width
            
            # Ensure the labeled frame fits exactly in the panel
            labeled_frame = cv2.resize(labeled_frame, (panel_width, panel_height))
            grid_canvas[start_y:end_y, start_x:end_x] = labeled_frame
        
        # Fill empty panels with black
        for empty_idx in range(num_models, rows * cols):
            row = empty_idx // cols
            col = empty_idx % cols
            start_y = row * panel_height
            end_y = start_y + panel_height
            start_x = col * panel_width
            end_x = start_x + panel_width
            grid_canvas[start_y:end_y, start_x:end_x] = 0  # Black
        
        # Write frame to video
        out.write(grid_canvas)
        processed_frames += 1
        
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames}/{len(frame_files)} frames")
    
    # Release everything
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Comparison video saved as: {output_video}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Models compared: {', '.join(model_names)}")

def auto_name_from_path(path):
    """Generate a readable model name from file path."""
    from pathlib import Path
    p = Path(path)

    # For tracking results: tracker/test1_YOLOX.txt -> YOLOX
    if p.suffix == '.txt':
        name = p.stem
        # Try to extract model name from filename
        parts = name.split('_')
        if len(parts) > 1:
            return parts[-1]  # Last part after underscore
        return name

    # For JSON: det_db_motrv2_DFINE.json -> DFINE
    if p.suffix == '.json':
        name = p.stem
        for prefix in ['det_db_motrv2_', 'det_db_', 'detections_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    return p.stem


def find_tracking_files(search_dirs=None, pattern="*.txt"):
    """Auto-discover tracking result files."""
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ['./tracker', './results', './exps', '.']

    results = []
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        for f in search_path.rglob(pattern):
            # Skip non-tracking files
            if f.name in ['config.txt', 'git_status', 'git_diff', 'desc', 'output.log']:
                continue
            # Skip if file is too small (likely not tracking data)
            if f.stat().st_size < 100:
                continue
            results.append(str(f))

    return sorted(set(results))


def interactive_mode(frames_dir=None):
    """Interactive mode for easy visualization."""
    print("\n" + "="*60)
    print("  MOTRv2 Tracking Visualization - Interactive Mode")
    print("="*60 + "\n")

    # Step 1: Find frames directory
    if frames_dir is None:
        print("Enter path to frames directory (containing .jpg images):")
        frames_dir = input("> ").strip()

    if not os.path.exists(frames_dir):
        print(f"Error: Directory not found: {frames_dir}")
        return

    # Step 2: Auto-discover tracking files
    print("\nSearching for tracking results...")
    tracking_files = find_tracking_files()

    # Also search for JSON files
    json_files = find_tracking_files(pattern="*.json")
    tracking_files.extend(json_files)
    tracking_files = sorted(set(tracking_files))

    if not tracking_files:
        print("No tracking files found. Please specify paths manually.")
        print("Enter tracking file paths (comma-separated):")
        paths = input("> ").strip()
        tracking_files = [p.strip() for p in paths.split(',')]
    else:
        print(f"\nFound {len(tracking_files)} tracking files:")
        for i, f in enumerate(tracking_files):
            auto_name = auto_name_from_path(f)
            print(f"  [{i+1}] {f}")
            print(f"       -> Auto-name: {auto_name}")

        print("\nEnter file numbers to compare (comma-separated), or 'all':")
        selection = input("> ").strip()

        if selection.lower() == 'all':
            pass  # Keep all files
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                tracking_files = [tracking_files[i] for i in indices]
            except (ValueError, IndexError) as e:
                print(f"Invalid selection: {e}")
                return

    # Step 3: Generate names
    model_names = [auto_name_from_path(f) for f in tracking_files]

    print("\nAuto-generated names:")
    for f, name in zip(tracking_files, model_names):
        print(f"  {os.path.basename(f)} -> {name}")

    print("\nPress Enter to accept, or enter custom names (comma-separated):")
    custom = input("> ").strip()
    if custom:
        model_names = [n.strip() for n in custom.split(',')]

    # Step 4: Output path
    print("\nEnter output video path (default: tracking_comparison.mp4):")
    output_video = input("> ").strip() or "tracking_comparison.mp4"

    # Step 5: Create video
    print("\n" + "-"*60)
    print("Creating comparison video...")
    print("-"*60 + "\n")

    create_comparison_video(
        frames_dir=frames_dir,
        tracking_files=tracking_files,
        model_names=model_names,
        output_video=output_video,
        fps=30,
        show_ids=True,
        show_confidence=False
    )


def main():
    parser = argparse.ArgumentParser(
        description='Create tracking comparison video with multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (guided setup)
  python create_video.py --interactive

  # Quick mode with auto-discovery
  python create_video.py frames_dir output.mp4 --auto

  # Standard mode
  python create_video.py frames_dir output.mp4 \\
      --tracking-files tracker/a.txt tracker/b.txt \\
      --model-names "Model A" "Model B"
        """
    )

    # Positional arguments (optional in interactive/auto mode)
    parser.add_argument('frames_dir', nargs='?', help='Directory containing frame images')
    parser.add_argument('output_video', nargs='?', default='tracking_comparison.mp4',
                       help='Output video file path (default: tracking_comparison.mp4)')

    # Mode selection
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode with guided setup')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-discover tracking files and generate names')

    # Tracking files
    parser.add_argument('--tracking-files', '-t', nargs='+',
                       help='List of tracking data files')
    parser.add_argument('--model-names', '-n', nargs='+',
                       help='List of model names (auto-generated if not provided with --auto)')

    # Video options
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--no-ids', action='store_true', help='Hide object IDs')
    parser.add_argument('--show-confidence', action='store_true', help='Show confidence scores')
    parser.add_argument('--frame-format', default='jpg', help='Frame image format (default: jpg)')
    parser.add_argument('--frame-pattern', help='Custom frame naming pattern (e.g., "{:06d}.jpg")')
    parser.add_argument('--grid-layout', nargs=2, type=int, metavar=('ROWS', 'COLS'),
                       help='Custom grid layout (rows cols)')

    # Search options
    parser.add_argument('--search-dirs', nargs='+', default=['./tracker', './results', './exps'],
                       help='Directories to search for tracking files (with --auto)')

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_mode(args.frames_dir)
        return

    # Auto mode
    if args.auto:
        if not args.frames_dir:
            parser.error("frames_dir required with --auto mode")

        print("Auto-discovering tracking files...")
        tracking_files = find_tracking_files(args.search_dirs)
        tracking_files.extend(find_tracking_files(args.search_dirs, "*.json"))
        tracking_files = sorted(set(tracking_files))

        if not tracking_files:
            print("No tracking files found. Use --tracking-files to specify manually.")
            return

        print(f"Found {len(tracking_files)} files:")
        for f in tracking_files:
            print(f"  {f}")

        model_names = [auto_name_from_path(f) for f in tracking_files]

        create_comparison_video(
            frames_dir=args.frames_dir,
            tracking_files=tracking_files,
            model_names=model_names,
            output_video=args.output_video,
            fps=args.fps,
            show_ids=not args.no_ids,
            show_confidence=args.show_confidence,
            frame_format=args.frame_format,
            frame_name_pattern=args.frame_pattern,
            grid_layout=tuple(args.grid_layout) if args.grid_layout else None
        )
        return

    # Standard mode - require tracking files
    if not args.tracking_files:
        parser.error("--tracking-files required (or use --auto or --interactive)")

    # Auto-generate names if not provided
    model_names = args.model_names
    if not model_names:
        model_names = [auto_name_from_path(f) for f in args.tracking_files]
        print(f"Auto-generated model names: {model_names}")

    if len(model_names) != len(args.tracking_files):
        parser.error("Number of model names must match number of tracking files")

    grid_layout = tuple(args.grid_layout) if args.grid_layout else None

    create_comparison_video(
        frames_dir=args.frames_dir,
        tracking_files=args.tracking_files,
        model_names=model_names,
        output_video=args.output_video,
        fps=args.fps,
        show_ids=not args.no_ids,
        show_confidence=args.show_confidence,
        frame_format=args.frame_format,
        frame_name_pattern=args.frame_pattern,
        grid_layout=grid_layout
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # No arguments - run interactive mode
        interactive_mode()