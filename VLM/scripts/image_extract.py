import os
import cv2
from tqdm import tqdm

def save_middle_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"[ERROR] No frames in video: {video_path}")
        cap.release()
        return False

    middle_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"[ERROR] Cannot read middle frame: {video_path}")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return True


def extract_images_from_segments(segment_dir, output_dir):
    for root, dirs, files in os.walk(segment_dir):
        for file in tqdm(files, desc="Extracting images"):
            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(root, file)

            # mp4 파일명에서 확장자 제거
            filename_without_ext = os.path.splitext(file)[0]

            output_filename = filename_without_ext + "_middle.jpg"
            output_path = os.path.join(output_dir, output_filename)

            save_middle_frame(video_path, output_path)


if __name__ == "__main__":
    segment_dir = "./dataset/segments"
    output_dir = "./dataset/extracted_images"

    extract_images_from_segments(segment_dir, output_dir)

    print("Done. Images saved to:", output_dir)