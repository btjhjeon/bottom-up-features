import _init_paths
import os
import numpy as np
import cv2
import torch
import argparse
from tqdm import tqdm
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from utils import get_image_blob, save_features, threshold_results


def parse_args():
    parser = argparse.ArgumentParser(description='Extract Bottom-up features')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file',
                        default='cfgs/faster_rcnn_resnet101.yml', type=str)
    parser.add_argument('--model', dest='model_file',
                        help='path to pretrained model',
                        default='models/bottomup_pretrained_10_100.pth', type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory with images',
                        default="images")
    parser.add_argument('--out_dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--boxes', dest='save_boxes',
                        help='save bounding boxes',
                        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load arguments.
    MIN_BOXES = 10
    MAX_BOXES = 100
    N_CLASSES = 1601
    CONF_THRESH = 0.2
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Works only with CUDA' 
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    cfg.CUDA = use_cuda
    np.random.seed(cfg.RNG_SEED)

    # Load the model.
    fasterRCNN = resnet(N_CLASSES, 101, pretrained=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(torch.load(args.model_file))
    fasterRCNN.to(device)
    fasterRCNN.eval()
    print('Model is loaded.')

    # Load images.
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    # Extract features.
    for im_file in tqdm(imglist):
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        blobs, im_scales = get_image_blob(im)
        assert len(im_scales) == 1, 'Only single-image batch is implemented'

        im_data = torch.from_numpy(blobs).permute(0, 3, 1, 2).to(device)
        im_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]]).to(device)
        gt_boxes = torch.zeros(1, 1, 5).to(device)
        num_boxes = torch.zeros(1).to(device)

        with torch.set_grad_enabled(False):
            rois, cls_prob, _, _, _, _, _, _, \
            pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        boxes = rois[:, :, 1:5].squeeze()
        boxes /= im_scales[0]
        cls_prob = cls_prob.squeeze()

        image_feat, image_bboxes = threshold_results(cls_prob, pooled_feat, boxes, CONF_THRESH, MIN_BOXES, MAX_BOXES)
        image_feat = image_feat.cpu().numpy()
        image_bboxes = image_bboxes.cpu().numpy()
        if not args.save_boxes:
            image_bboxes = None

        output_file = os.path.join(args.output_dir, im_file.split('.')[0]+'.npy')
        save_features(output_file, image_feat, image_bboxes)
        #torch.cuda.empty_cache()
