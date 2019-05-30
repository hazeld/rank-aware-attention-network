import os
import argparse
import subprocess
import pandas as pd

parser = argparse.ArgumentParser('Program to download videos for BEST Dataset and EPIC-Skills')
parser.add_argument('vid_list_filename', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--trim', default=False, action='store_true', help='Optionally trim the videos to the temporal bounds used in the paper')
args = parser.parse_args()

def read_vid_list(filename):
    return pd.read_csv(filename)

def trim_videos(video_list_df, output_dir):
    for i, row in video_list_df.iterrows():
        if row['task'] == 'scrambled_eggs':
            continue
        vid_path = os.path.join(output_dir, row['task'], row['vid_id'] + '.mp4')
        if not os.path.isfile(vid_path):
            print('Video ' + row['vid_id'] + ' from task ' + row['task'] + ' not present, skipping')
            continue
        original_vid_path = vid_path.replace('.mp4', '_original.mp4')
        os.rename(vid_path, original_vid_path)
        duration = float(row['end_seconds']) - float(row['start_seconds'])
        subprocess.call(['ffmpeg', '-ss', str(row['start_seconds']), '-i', original_vid_path, '-t', str(duration), '-c:a', 'ac3', '-c:v', 'libx264', '--', vid_path])
        os.remove(original_vid_path)

def download_vids(video_list_df, output_dir):
    errors = []
    task_names = video_list_df.task.unique()
    for task in task_names:
        output_task_dir = os.path.join(output_dir, task)
        if not os.path.isdir(output_task_dir):
            os.mkdir(output_task_dir)
    for i, row in video_list_df.iterrows():
        try:
            subprocess.check_output(['youtube-dl', '-f', 'mp4', '-i', '-o', os.path.join(output_dir, row['task'],'%(id)s.%(ext)s'), '--', row['vid_id']])
        except subprocess.CalledProcessError:
            print(row['vid_id'])
            errors.append(row['vid_id'])
    return errors

if __name__ == '__main__':
    video_list_df = read_vid_list(args.vid_list_filename)
    errors = download_vids(video_list_df, args.output_dir)
    print('Unable download the following videos: ')
    print(errors)
    if args.trim:
        trim_videos(video_list_df, args.output_dir)
