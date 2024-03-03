import os
import sys
import glob
import argparse


def png2avif(args):
    output_path = args.output_path +'qp_{}'.format(args.qp_value)
    os.makedirs(output_path, exist_ok=True)
    img_list = os.listdir(args.input_path)

    for i in img_list:
        retm = os.system('{} -hide_banner -y -loglevel error -i {}/{}'
                        ' -vf scale=ceil(iw/4):ceil(ih/4):flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5' 
                        ' -c:v libaom-av1 -qp 31 -preset 5 {}/{}.avif'.format(args.ffmpeg_dir, args.input_path, i, output_path, i[0:4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ffmpeg_dir", type=str, default="ffmpeg", help='path to ffmpeg.exe')
    parser.add_argument('--input_path', type=str, default='/Users/kihwan/Desktop/DIV2K_train_HR/', help = 'DIV2K HR input_path')
    parser.add_argument('--output_path', type=str, default='/Users/kihwan/Desktop/DIV2K_train_LR_avif/', help = 'DIV2K avif downsample path')
    parser.add_argument('--qp_value', type=int, default=31, help = 'qp value')

    args = parser.parse_args()

    png2avif(args)