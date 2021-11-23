import os
import sys
import numpy as np
import torch
import paddle


def transfer(input_fp, output_fp):
    os.makedirs(os.path.dirname(input_fp), exist_ok=True)
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    torch_dict = torch.load(input_fp)
    paddle_dict = {}
    fc_names = []
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # input_fp = "../animegan2-pytorch/weights/face_paint_512_v2.pt"
        # output_fp = "./weights/face_paint_512_v2.pdparams"

        # input_fp = "../animegan2-pytorch/weights/celeba_distill.pt"
        # output_fp = "./weights/celeba_distill.pdparams"

        # input_fp = "../animegan2-pytorch/weights/face_paint_512_v1.pt"
        # output_fp = "./weights/face_paint_512_v1.pdparams"

        input_fp = "../animegan2-pytorch/weights/paprika.pt"
        output_fp = "./weights/paprika.pdparams"
    else:
        input_fp = sys.argv[1]
        output_fp = sys.argv[2]
    transfer(input_fp, output_fp)
