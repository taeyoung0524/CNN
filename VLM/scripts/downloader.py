import yt_dlp
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os


def download_video(video_id, output_path):
    ydl_opts = {
        'outtmpl': os.path.join(output_path, f'%(id)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'ignoreerrors': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_id])
        except:
            return video_id


def download_videos(video_ids, output_path):
    os.makedirs(output_path, exist_ok=True)
    failed_videos = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for video_id in video_ids:
            futures.append(executor.submit(download_video, video_id, output_path))
        for future in tqdm(futures, total=len(futures), desc='Downloading', unit='video'):
            result = future.result()
            if result:
                failed_videos.append(result)
                file_path = "failed_list.txt"
                write_list_to_file(file_path, failed_videos)
    return failed_videos


def write_list_to_file(file_path, input_list):
    try:
        with open(file_path, 'w') as file:
            for item in input_list:
                file.write(str(item) + '\n')
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    output_path = './dataset/original_video'
    video_ids = [
    "0B4x_mHgVN4",   # inject medication
    "HxT7E14euZ8",   # inject medication
    "YLMPAtkg-Kk",   # subcutaneous puncture
    "F88btEbzdw0",   # cleanse skin
    "SClSnPYIay0",   # inject medication
]

    failed_videos = download_videos(video_ids, output_path)

    if failed_videos:
        print('Video IDs unable to download:')
        for video_id in failed_videos:
            print(video_id)
    else:
        print('All videos downloaded successfully!')