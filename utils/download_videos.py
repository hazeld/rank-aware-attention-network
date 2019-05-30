import os
import argparse
import subprocess
import pandas as pd

parser = argparse.ArgumentParser('Program to download videos for BEST Dataset and EPIC-Skills')
parser.add_argument('vid_list_filename', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

def read_vid_list(filename):
    return pd.read_csv(filename)

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
