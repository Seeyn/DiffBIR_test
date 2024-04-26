import os
import math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from ldm.xformers_state import auto_xformers_status
from model.cldm import ControlLDM
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import auto_resize, pad
from utils.file import load_file_from_url
from utils.face_restoration_helper import FaceRestoreHelper
from model.spaced_sampler import SpacedSampler
from dataset.codeformer import CodeformerDataset
from inference import check_device
import einops
from torch.utils.data import DataLoader
import pickle

from typing import List, Tuple, Optional
from model.cond_fn import MSEGuidance
pretrained_models = {
    'general_v1': {
        'ckpt_url': 'https://huggingface.co/lxq007/DiffBIR/resolve/main/general_full_v1.ckpt',
        'swinir_url': 'https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt'
    },
    'face_v1': {
        'ckpt_url': 'https://huggingface.co/lxq007/DiffBIR/resolve/main/face_full_v1.ckpt'
    }
}

@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool,
    cond_fn: Optional[MSEGuidance],
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    return_latent = False
):
    """
    Apply DiffBIR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255]).
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
        cond_fn (Guidance | None): Guidance function that returns gradient to guide the predicted x_0.
        tiled (bool): If specified, a patch-based sampling strategy will be used for sampling.
        tile_size (int): Size of patch.
        tile_stride (int): Stride of sliding patch.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = control_imgs.shape[0]
    sampler = SpacedSampler(model, var_type="fixed_small")
    control = einops.rearrange(control_imgs, 'b h w c -> b c h w').to(model.device)
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    model.control_scales = [strength] * 13
    
    if cond_fn is not None:
        cond_fn.load_target(2 * control - 1)
    
    height, width = control.size(-2), control.size(-1)
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if not tiled:
        if return_latent:
            samples,latents = sampler.sample(
                steps=steps, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0, cond_fn=cond_fn,
                color_fix_type=color_fix_type,
                return_latent=return_latent
            )
            latents = [latents[i] for i in range(n_samples)]
        else:
            samples = sampler.sample(
                steps=steps, shape=shape, cond_img=control,
                positive_prompt="", negative_prompt="", x_T=x_T,
                cfg_scale=1.0, cond_fn=cond_fn,
                color_fix_type=color_fix_type,
                return_latent=return_latent)
    else:
        samples = sampler.sample_with_mixdiff(
            tile_size=tile_size, tile_stride=tile_stride,
            steps=steps, shape=shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=cond_fn,
            color_fix_type=color_fix_type
        )
    x_samples = samples.clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    # control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    if return_latent:
        return preds,stage1_preds,latents
    return preds, stage1_preds

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model
    # Specify the model ckpt path, and the official model can be downloaded direclty.
    parser.add_argument("--ckpt", type=str, help='Model checkpoint.', default='weights/face_full_v1.ckpt')
    parser.add_argument("--config", type=str, default='configs/model/cldm_twoS.yaml', help='Model config file.')
    parser.add_argument("--reload_swinir", action="store_true")
    parser.add_argument("--swinir_ckpt", type=str, default=None)

    # input and preprocessing
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sr_scale", type=float, default=2, help='An upscale factor.')
    parser.add_argument("--image_size", type=int, default=512, help='Image size as the model input.')
    parser.add_argument("--repeat_times", type=int, default=1, help='To generate multiple results for each input image.')
    parser.add_argument("--disable_preprocess_model", action="store_true")

    # face related
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')

    # Loading two DiffBIR models requires huge GPU memory capacity. Choose RealESRGAN as an alternative.
    parser.add_argument('--bg_upsampler', type=str, default='None', choices=['DiffBIR', 'RealESRGAN'], help='Background upsampler.')
    # TODO: support tiled for DiffBIR background upsampler
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler.')
    parser.add_argument('--bg_tile_stride', type=int, default=200, help='Tile stride for background sampler.')
    
    # postprocessing and saving
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    # change seed to finte-tune your restored images! just specify another random number.
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def build_diffbir_model(model_config, ckpt, swinir_ckpt=None):
    ''''
        model_config: model architecture config file.
        ckpt: checkpoint file path of the main model.
        swinir_ckpt: checkpoint file path of the swinir model.
            load swinir from the main model if set None.
    '''
    weight_root = os.path.dirname(ckpt)

    # download ckpt automatically if ckpt not exist in the local path
    if 'general_full_v1' in ckpt:
        ckpt_url = pretrained_models['general_v1']['ckpt_url']
        if swinir_ckpt is None:
            swinir_ckpt = f'{weight_root}/general_swinir_v1.ckpt'
            swinir_url  = pretrained_models['general_v1']['swinir_url']
    elif 'face_full_v1' in ckpt:
        # swinir ckpt is already included in the main model
        ckpt_url = pretrained_models['face_v1']['ckpt_url']
    else:
        # define a custom diffbir model
        #raise NotImplementedError('undefined diffbir model type!')
        pass
    
    if not os.path.exists(ckpt):
        ckpt = load_file_from_url(ckpt_url, weight_root)
    if swinir_ckpt is not None and not os.path.exists(swinir_ckpt):
        swinir_ckpt = load_file_from_url(swinir_url, weight_root)
    
    #model: ControlLDM = instantiate_from_config(OmegaConf.load(model_config))
    model = instantiate_from_config(OmegaConf.load(model_config))
    load_state_dict(model, torch.load(ckpt), strict=True)
    # reload preprocess model if specified
    if swinir_ckpt is not None:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(swinir_ckpt), strict=True)
    model.freeze()
    return model


def main() -> None:
    args = parse_args()
    img_save_ext = 'png'
    pl.seed_everything(args.seed)
    
    assert os.path.isdir(args.input)

    args.device = check_device(args.device)
    model = build_diffbir_model(args.config, args.ckpt, args.swinir_ckpt).to(args.device)

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(
        device=args.device, 
        upscale_factor=1, 
        face_size=args.image_size, 
        use_parse=True,
        det_model = args.detection_model
        )

    dataset_config = OmegaConf.load("./configs/dataset/face_train.yaml")
    
    dataset = instantiate_from_config(dataset_config['dataset'])
    test_dataloader = DataLoader(dataset,batch_size = 4, shuffle = False, num_workers=16, drop_last = False)
    for epoch in range(2):
        for batch_idx, batch in enumerate(test_dataloader):
            # read image
            lq = batch['hint']
            hq = batch['jpg'].permute(0,3,1,2)
            loc_left_eye = batch['loc_left_eye']
            loc_right_eye = batch['loc_right_eye']
            loc_mouth = batch['loc_mouth']
            
            preds, stage1_preds,latents = process(
                    model, lq, steps=args.steps,
                    strength=1,
                    color_fix_type=args.color_fix_type,
                    disable_preprocess_model=args.disable_preprocess_model,
                    cond_fn=None, tiled=False, tile_size=None, tile_stride=None,return_latent=True
                )

            for id in range(len(preds)):
                tmp = {'sr':stage1_preds[id].cpu().detach(),'latent':latents[id],'hq':hq[id],'pred':preds[id],'loc_left_eye':loc_left_eye[id],'loc_right_eye':loc_right_eye[id],'loc_mouth':loc_mouth[id]}#,'pred':preds[id]}
                name = '%d_%d_%d.pkl'%(epoch,batch_idx,id)
                output = os.path.join(args.output, name)
                with open(output,'wb') as f:
                    pickle.dump(tmp,f)



if __name__ == "__main__":
    main()
