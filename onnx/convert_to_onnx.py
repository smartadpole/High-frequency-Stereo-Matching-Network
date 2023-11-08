from __future__ import print_function, division

from core.dlnr import DLNR, autocast
import torch
import argparse

maxdisp = 192
name = "MSNet2D"


def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='/your_path/dlnr.pth')
    parser.add_argument('--dataset', help="dataset for evaluation",
                        choices=["eth3d", "kitti", "things"] + [f"middlebury_{s}" for s in 'FHQ'], default='things')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--valid_iters', type=int, default=10, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "reg_cuda"], default="reg_cuda",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders", default=True)
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    return args

def test():
    print("Generating the disparity maps...")
    args = GetArgs()

    model = DLNR(args)
    model.cuda()
    model.eval()

    width = 624
    height = 192
    dummy_input_L = torch.randn(1, 3, height, width, device='cuda:0')
    dummy_input_R = torch.randn(1, 3, height, width, device='cuda:0')
    input_names = ['L', 'R']
    output_names = ['output']
    torch.onnx.export(
        model,
        (dummy_input_L,dummy_input_R),
        "./{}.onnx".format(name),
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)

if __name__ == '__main__':
    test()
