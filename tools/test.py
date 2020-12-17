from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualize result')
parser.add_argument('--threshold', default=0.85, type=float,
                    help='IOU minimum threshold')
parser.add_argument('--model_name', default='', type=str,
                    help='Model architecture')
parser.add_argument('--skip_frames', default=1, type=int,
                    help='number of frames to skip after failure')
parser.add_argument('--experiment_name', default='', type=str,
                    help='what experiment is this')
parser.add_argument('--results_path', default='', type=str,
                    help='absolute path of results directory')
parser.add_argument('--dataset_directory', default='', type=str,
                    help='path to dataset directory')

args = parser.parse_args()

torch.set_num_threads(1)
vot_like_dataset = ['VOT2016', 'VOT2018', 'VOT2019']
vot_like_dataset.append('Sauron')
def main():
    is_gpu_cuda_available = torch.cuda.is_available()
    if not is_gpu_cuda_available:
        raise RuntimeError('Failed to locate a CUDA GPU. Program cannot continue..')
    num_gpus = torch.cuda.device_count()
    gpu_type = torch.cuda.get_device_name(0)
    print(f"You have {num_gpus} available of type: {gpu_type}")
    print("This might take a few minutes...Grab a cup of coffee\n")

    # load config
    cfg.merge_from_file(args.config)
    dataset_root = os.path.join(args.dataset_directory, args.dataset)
    print(f"dataset root-->{dataset_root}")

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.model_name
    print(f"Model name is {model_name}")

    total_lost = 0
    if args.dataset in vot_like_dataset:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0.85:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + args.skip_frames # skip 1 frame
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            save_path = os.path.join(args.results_path, args.dataset, model_name, args.experiment_name, video.name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            result_path = os.path.join(save_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            with open(os.path.join(save_path, '..', 'lost.txt'), 'a+') as f:
                f.write(f"{v_idx+1} Class: {video.name} | Time: {toc}s | Speed: {idx/toc}fps | Lost:{lost_number}  \n")

            print('({:3d}) Class: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        with open(os.path.join(save_path, '..', 'lost.txt'), 'a+') as f:
            f.write(f"Model architeture used --> {model_name} \ntotal lost: {total_lost} \n")
            f.write(f"SKIP FRAMES USED --> {args.skip_frames}")
    else:
        # OPE tracking
        # will be implemented if needed in future
        pass


if __name__ == '__main__':
    main()

