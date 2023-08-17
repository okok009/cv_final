# python demo/webcam_demo.py configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py checkpoints/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth --camera-id 0
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
import torch
import numba
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

@numba.jit(nopython=True)
def image_matting(pred_img_data : np.ndarray, image : np.ndarray):
    for i in range(480):
        for j in range(640):
            for k in range(3):
                if pred_img_data[i, j, k] >= 20  and pred_img_data[i, j, k] <= 234:
                    break
                if k == 2:
                    image[i:i+3, j:j+3, k] = 0
                    image[i:i+3, j:j+3, k-1] = 0
                    image[i:i+3, j:j+3, k-2] = 0
    return image

def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    camera = cv2.VideoCapture(args.camera_id)
    want_label = []
    print('Press "Esc", "q" or "Q" to exit.')

    person, cup, keyboard, tv = False, False, False, False

    while True:
        ret_val, img = camera.read()

        img = cv2.resize(img, (640, 480))
        result = inference_detector(model, img, None, want_label)

        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            show=False, 
            image_matting = image_matting)

        img = visualizer.get_image()
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        cv2.imshow('result', img)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

        if key == ord("0"):
            if person == False:
                want_label.append(0)
                person = True
            else:
                want_label.remove(0)
                person = False

        if key == ord("1"):
            if cup == False:
                want_label.append(1)
                cup = True
            else:
                want_label.remove(1)
                cup = False

        if key == ord("2"):
            if keyboard == False:
                want_label.append(2)
                keyboard = True
            else:
                want_label.remove(2)
                keyboard = False

        if key == ord("3"):
            if tv == False:
                want_label.append(3)
                tv = True
            else:
                want_label.remove(3)
                tv = False



if __name__ == '__main__':
    main()
