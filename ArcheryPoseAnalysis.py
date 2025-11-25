import cv2
import numpy as np
from datetime import datetime
import csv
import time
from ultralytics import YOLO
import mediapipe as mp
import torch
import math
import os
import sys
from pathlib import Path
import argparse

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def elbow_angle(p1, p2):
    """
    Angle of line p1→p2 relative to horizontal.
    0° = horizontal
    +degrees = up
    -degrees = down
    """
    x1, y1 = p1
    x2, y2 = p2
    dy = -(y2 - y1)
    dx = (x2 - x1)
    angle = 180 - math.degrees(math.atan2(dy, dx))
    return angle


# YOLO keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def init_models(use_hands=True):
    """Initialize YOLO and MediaPipe models"""
    print("Loading models...")
    
    # Load YOLO Pose Model
    try:
        pose_model = YOLO('yolo11m-pose.pt')
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("CUDA not available, using CPU")
            device = 'cpu'
    except Exception as e:
        print(f"Error loading GPU model, using CPU: {e}")
        device = 'cpu'
        pose_model = YOLO('yolo11n-pose.pt')

    #device = 'cpu'
    #pose_model = YOLO('yolo11s-pose.pt')
    
    # Initialize MediaPipe Hands if needed
    hands = None
    mp_hands = None
    mp_drawing = None
    mp_drawing_styles = None
    
    if use_hands:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
    
    print("Models loaded successfully!")
    return pose_model, device, hands, mp_hands, mp_drawing, mp_drawing_styles


def process_frame(image, pose_model, device, hands, mp_hands, mp_drawing, mp_drawing_styles):
    """Process a single frame and return annotated image and data"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hands if enabled
    hand_results = None
    if hands is not None:
        hand_results = hands.process(image_rgb)
    
    # YOLO Pose
    results = pose_model(image, verbose=False, device=device)
    
    detections_data = []
    
    # Process YOLO detections
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        if keypoints is not None and len(keypoints) > 0:
            for person_id, (box, kpts) in enumerate(zip(boxes, keypoints)):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(image, f'Person {person_id+1}: {conf:.2f}',
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                kpts_data = kpts.xy[0].cpu().numpy()
                kpts_conf = kpts.conf[0].cpu().numpy()

                # Skeleton connections
                connections = [
                    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]

                for c1, c2 in connections:
                    if kpts_conf[c1] > 0.5 and kpts_conf[c2] > 0.5:
                        pt1 = tuple(kpts_data[c1].astype(int))
                        pt2 = tuple(kpts_data[c2].astype(int))
                        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

                # Draw keypoints
                for (px, py), conf_val in zip(kpts_data, kpts_conf):
                    if conf_val > 0.5:
                        cv2.circle(image, (int(px), int(py)), 4, (255, 0, 0), -1)

                angles = {
                    'left_elbow': None,
                    'right_elbow': None,
                    'left_shoulder': None,
                    'right_shoulder': None,
                    'shoulders': None
                }

                # Between shoulders
                if all(kpts_conf[i] > 0.5 for i in [5,6]):
                    angles['shoulders'] = elbow_angle(kpts_data[5], kpts_data[6])
                    cv2.putText(image, str(int(angles['shoulders'])),
                                (int(kpts_data[6][0]) + 10, int(kpts_data[6][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.line(image,(int(kpts_data[5][0]),int(kpts_data[5][1])),
                            (int(kpts_data[6][0]),int(kpts_data[5][1])),(255,0,0),1)

                # LEFT elbow
                if all(kpts_conf[i] > 0.5 for i in [5, 7, 9]):
                    angles['left_elbow'] = calculate_angle(kpts_data[5], kpts_data[7], kpts_data[9])
                    cv2.putText(image, str(int(angles['left_elbow'])),
                                (int(kpts_data[7][0]) + 10, int(kpts_data[7][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # RIGHT elbow
                if all(kpts_conf[i] > 0.5 for i in [6, 8, 10]):
                    angles['right_elbow'] = calculate_angle(kpts_data[6], kpts_data[8], kpts_data[10])
                    cv2.putText(image, str(int(angles['right_elbow'])),
                                (int(kpts_data[8][0]) - 40, int(kpts_data[8][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # LEFT shoulder
                if all(kpts_conf[i] > 0.5 for i in [5, 7, 11]):
                    angles['left_shoulder'] = calculate_angle(kpts_data[7], kpts_data[5], kpts_data[11])
                    cv2.putText(image, str(int(angles['left_shoulder'])),
                                (int(kpts_data[5][0]) + 10, int(kpts_data[5][1]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # RIGHT shoulder
                if all(kpts_conf[i] > 0.5 for i in [6, 8, 12]):
                    angles['right_shoulder'] = calculate_angle(kpts_data[8], kpts_data[6], kpts_data[12])
                    cv2.putText(image, str(int(angles['right_shoulder'])),
                                (int(kpts_data[6][0]) - 40, int(kpts_data[6][1]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Store detection data
                detection = {
                    'person_id': person_id + 1,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'keypoints': kpts_data,
                    'keypoints_conf': kpts_conf,
                    'angles': angles
                }
                detections_data.append(detection)

    # Draw hands
    if hand_results is not None and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return image, detections_data


def analyze_webcam(use_hands=True, output_name=None):
    """Analyze webcam feed with recording capability"""
    print("\n=== WEBCAM MODE ===")
    
    pose_model, device, hands, mp_hands, mp_drawing, mp_drawing_styles = init_models(use_hands)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = actual_fps if actual_fps > 0 else 30.0

    print(f"Webcam: {frame_width}x{frame_height} @ {fps} FPS")
    print("Press 'r' to START/STOP recording")
    print("Press SPACE to pause")
    print("Press ESC to quit")

    recording = False
    out = None
    csv_file = None
    csv_writer = None
    frame_number = 0
    recording_frame_count = 0

    def start_recording():
        nonlocal out, csv_file, csv_writer, recording_frame_count
        
        if output_name:
            base_name = output_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"webcam_{timestamp}"
        
        output_video = f"{base_name}.mp4"
        output_csv = f"{base_name}.csv"

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            print("⚠️ Trying mp4v codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print("❌ Failed to open video writer")
                out = None
                return False

        recording_frame_count = 0
        csv_file = open(output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        header = ['frame', 'time_seconds', 'person_id',
                  'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
        for name in KEYPOINT_NAMES:
            header.extend([f'{name}_x', f'{name}_y', f'{name}_conf'])
        header.extend(['left_elbow_angle', 'right_elbow_angle',
                       'left_shoulder_angle', 'right_shoulder_angle'])
        csv_writer.writerow(header)

        print(f"\n🎥 Recording: {output_video}")
        print(f"📄 CSV: {output_csv}")
        return True

    def stop_recording():
        nonlocal out, csv_file, recording_frame_count
        if out:
            out.release()
            print(f"✅ Video saved: {recording_frame_count} frames")
            out = None
        if csv_file:
            csv_file.close()
            csv_file = None
        print("🟥 Recording stopped\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_time = (frame_number - 1) / fps

        image = cv2.flip(frame, 1)
        image, detections = process_frame(image, pose_model, device, hands, 
                                         mp_hands, mp_drawing, mp_drawing_styles)

        if recording:
            cv2.putText(image, "● REC", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(image, f"Frames: {recording_frame_count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(image, "Press 'r' to record | ESC to quit", 
                   (10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if recording and out is not None:
            out.write(image)
            recording_frame_count += 1
            
            if csv_writer:
                for det in detections:
                    row = [frame_number, f"{current_time:.3f}", det['person_id'],
                           f"{det['bbox'][0]:.2f}", f"{det['bbox'][1]:.2f}",
                           f"{det['bbox'][2]:.2f}", f"{det['bbox'][3]:.2f}",
                           f"{det['confidence']:.4f}"]
                    
                    for (px, py), cp in zip(det['keypoints'], det['keypoints_conf']):
                        row += [f"{px:.2f}", f"{py:.2f}", f"{cp:.4f}"]
                    
                    angles = det['angles']
                    row += [
                        f"{angles['left_elbow']:.2f}" if angles['left_elbow'] else "",
                        f"{angles['right_elbow']:.2f}" if angles['right_elbow'] else "",
                        f"{angles['left_shoulder']:.2f}" if angles['left_shoulder'] else "",
                        f"{angles['right_shoulder']:.2f}" if angles['right_shoulder'] else ""
                    ]
                    csv_writer.writerow(row)

        cv2.imshow("Pose Analysis - Webcam", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                if start_recording():
                    recording = True
            else:
                recording = False
                stop_recording()
        elif key == ord(' '):
            print("Paused - press any key...")
            cv2.waitKey(0)
        elif key == 27:
            break

    if recording:
        stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    if hands:
        hands.close()
    print("Webcam analysis finished")


def analyze_video(video_path, use_hands=True):
    """Analyze a video file"""
    print(f"\n=== VIDEO MODE ===")
    print(f"Input: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    pose_model, device, hands, mp_hands, mp_drawing, mp_drawing_styles = init_models(use_hands)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output files
    path = Path(video_path)
    output_video = str(path.parent / f"{path.stem}_analyzed{path.suffix}")
    output_csv = str(path.parent / f"{path.stem}_analyzed.csv")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    header = ['frame', 'time_seconds', 'person_id',
              'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
    for name in KEYPOINT_NAMES:
        header.extend([f'{name}_x', f'{name}_y', f'{name}_conf'])
    header.extend(['left_elbow_angle', 'right_elbow_angle',
                   'left_shoulder_angle', 'right_shoulder_angle'])
    csv_writer.writerow(header)

    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}")
    print(f"Processing {total_frames} frames...")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        current_time = (frame_number - 1) / fps

        image, detections = process_frame(frame, pose_model, device, hands,
                                         mp_hands, mp_drawing, mp_drawing_styles)

        # Progress indicator
        cv2.putText(image, f"Frame: {frame_number}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(image)

        for det in detections:
            row = [frame_number, f"{current_time:.3f}", det['person_id'],
                   f"{det['bbox'][0]:.2f}", f"{det['bbox'][1]:.2f}",
                   f"{det['bbox'][2]:.2f}", f"{det['bbox'][3]:.2f}",
                   f"{det['confidence']:.4f}"]
            
            for (px, py), cp in zip(det['keypoints'], det['keypoints_conf']):
                row += [f"{px:.2f}", f"{py:.2f}", f"{cp:.4f}"]
            
            angles = det['angles']
            row += [
                f"{angles['left_elbow']:.2f}" if angles['left_elbow'] else "",
                f"{angles['right_elbow']:.2f}" if angles['right_elbow'] else "",
                f"{angles['left_shoulder']:.2f}" if angles['left_shoulder'] else "",
                f"{angles['right_shoulder']:.2f}" if angles['right_shoulder'] else ""
            ]
            csv_writer.writerow(row)

        if frame_number % 30 == 0:
            print(f"Progress: {frame_number}/{total_frames} ({frame_number*100//total_frames}%)")

    cap.release()
    out.release()
    csv_file.close()
    if hands:
        hands.close()
    
    print(f"✅ Video analysis complete!")
    print(f"   Output: {output_video}")
    print(f"   CSV: {output_csv}")


def analyze_images(folder_path, use_hands=True):
    """Analyze a folder of images"""
    print(f"\n=== IMAGE FOLDER MODE ===")
    print(f"Input folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return
    
    pose_model, device, hands, mp_hands, mp_drawing, mp_drawing_styles = init_models(use_hands)
    
    # Get image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print("No image files found in folder")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create output folder
    output_folder = Path(folder_path) / "analyzed"
    output_folder.mkdir(exist_ok=True)
    
    # CSV file
    output_csv = output_folder / "analysis.csv"
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    header = ['image_name', 'person_id',
              'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
    for name in KEYPOINT_NAMES:
        header.extend([f'{name}_x', f'{name}_y', f'{name}_conf'])
    header.extend(['left_elbow_angle', 'right_elbow_angle',
                   'left_shoulder_angle', 'right_shoulder_angle'])
    csv_writer.writerow(header)
    
    print(f"Output folder: {output_folder}")
    print(f"CSV: {output_csv}")
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️ Could not read image")
            continue
        
        processed_image, detections = process_frame(image, pose_model, device, hands,
                                                   mp_hands, mp_drawing, mp_drawing_styles)
        
        # Save analyzed image
        output_path = output_folder / f"{img_path.stem}_analyzed{img_path.suffix}"
        cv2.imwrite(str(output_path), processed_image)
        
        # Write CSV data
        for det in detections:
            row = [img_path.name, det['person_id'],
                   f"{det['bbox'][0]:.2f}", f"{det['bbox'][1]:.2f}",
                   f"{det['bbox'][2]:.2f}", f"{det['bbox'][3]:.2f}",
                   f"{det['confidence']:.4f}"]
            
            for (px, py), cp in zip(det['keypoints'], det['keypoints_conf']):
                row += [f"{px:.2f}", f"{py:.2f}", f"{cp:.4f}"]
            
            angles = det['angles']
            row += [
                f"{angles['left_elbow']:.2f}" if angles['left_elbow'] else "",
                f"{angles['right_elbow']:.2f}" if angles['right_elbow'] else "",
                f"{angles['left_shoulder']:.2f}" if angles['left_shoulder'] else "",
                f"{angles['right_shoulder']:.2f}" if angles['right_shoulder'] else ""
            ]
            csv_writer.writerow(row)
    
    csv_file.close()
    if hands:
        hands.close()
    
    print(f"✅ Image analysis complete!")
    print(f"   Analyzed images: {output_folder}")
    print(f"   CSV: {output_csv}")


def interactive_menu():
    """Interactive menu for selecting analysis mode"""
    print("\n" + "="*50)
    print("     POSE ANALYSIS TOOL")
    print("="*50)
    print("\nSelect analysis mode:")
    print("  1. Webcam (with recording)")
    print("  2. Video file")
    print("  3. Image folder")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        hands_choice = input("Enable hand tracking? (y/n): ").strip().lower()
        use_hands = hands_choice == 'y'
        output_name = input("Output filename (without extension, press Enter for timestamp): ").strip()
        output_name = output_name if output_name else None
        analyze_webcam(use_hands=use_hands, output_name=output_name)
        
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        hands_choice = input("Enable hand tracking? (y/n): ").strip().lower()
        use_hands = hands_choice == 'y'
        analyze_video(video_path, use_hands=use_hands)
        
    elif choice == '3':
        folder_path = input("Enter image folder path: ").strip()
        hands_choice = input("Enable hand tracking? (y/n): ").strip().lower()
        use_hands = hands_choice == 'y'
        analyze_images(folder_path, use_hands=use_hands)
        
    elif choice == '4':
        print("Exiting...")
        sys.exit(0)
        
    else:
        print("Invalid choice!")


def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(description='Pose Analysis Tool')
    parser.add_argument('--mode', choices=['webcam', 'video', 'images', 'interactive'],
                       default='interactive', help='Analysis mode')
    parser.add_argument('--input', help='Input video file or image folder path')
    parser.add_argument('--hands', action='store_true', help='Enable hand tracking')
    parser.add_argument('--no-hands', action='store_true', help='Disable hand tracking')
    parser.add_argument('--output', help='Output name for webcam mode')
    
    args = parser.parse_args()
    
    # Determine hand tracking setting
    if args.no_hands:
        use_hands = False
    elif args.hands:
        use_hands = True
    else:
        use_hands = True  # Default
    
    if args.mode == 'interactive':
        interactive_menu()
    elif args.mode == 'webcam':
        analyze_webcam(use_hands=use_hands, output_name=args.output)
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            sys.exit(1)
        analyze_video(args.input, use_hands=use_hands)
    elif args.mode == 'images':
        if not args.input:
            print("Error: --input required for images mode")
            sys.exit(1)
        analyze_images(args.input, use_hands=use_hands)


if __name__ == "__main__":
    main()