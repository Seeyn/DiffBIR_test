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
    parser.add_argument("--ckpt", type=str, help='Model checkpoint.', default='None')
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
    
    if swinir_ckpt is not None and not os.path.exists(swinir_ckpt):
        swinir_ckpt = load_file_from_url(swinir_url, weight_root)
    
    #model: ControlLDM = instantiate_from_config(OmegaConf.load(model_config))
    model = instantiate_from_config(OmegaConf.load(model_config))
    #load_state_dict(model, torch.load(ckpt), strict=True)
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

    # set up the backgrouns upsampler
    if args.bg_upsampler == 'DiffBIR':
        # Loading two DiffBIR models consumes huge GPU memory capacity.
        bg_upsampler = build_diffbir_model(args.config, 'weights/general_full_v1.pth')
        bg_upsampler = bg_upsampler.to(args.device)
    elif args.bg_upsampler == 'RealESRGAN':
        from utils.realesrgan.realesrganer import set_realesrgan
        # support official RealESRGAN x2 & x4 upsample model.
        # Using x2 upsampler as default if scale is not specified as 4.
        bg_upscale = int(args.sr_scale) if int(args.sr_scale) in [2, 4] else 2
        print(f'Loading RealESRGAN_x{bg_upscale}plus.pth for background upsampling...')
        bg_upsampler = set_realesrgan(args.bg_tile, args.device, bg_upscale)
    else:
        bg_upsampler = None

    for file_path in list_image_files(args.input, follow_links=True):
        # read image
        lq = Image.open(file_path).convert("RGB")

        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        lq_resized = auto_resize(lq, args.image_size)
        x = pad(np.array(lq_resized), scale=64)

        face_helper.clean_all()
        if args.has_aligned: 
            # the input faces are already cropped and aligned
            face_helper.cropped_faces = [x]
        else:
            face_helper.read_image(x)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

        parent_dir, img_basename, _ = get_file_name_parts(file_path)
        rel_parent_dir = os.path.relpath(parent_dir, args.input)
        output_parent_dir = os.path.join(args.output, rel_parent_dir)
        cropped_face_dir = os.path.join(output_parent_dir, 'cropped_faces')
        restored_face_dir = os.path.join(output_parent_dir, 'restored_faces')
        restored_img_dir = os.path.join(output_parent_dir, 'restored_imgs')
        if not args.has_aligned:
            os.makedirs(cropped_face_dir, exist_ok=True)
            os.makedirs(restored_img_dir, exist_ok=True)
        os.makedirs(restored_face_dir, exist_ok=True)
        for i in range(args.repeat_times):
            basename =  f'{img_basename}_{i}' if i else img_basename
            restored_img_path = os.path.join(restored_img_dir, f'{basename}.{img_save_ext}')
            if os.path.exists(restored_img_path) or os.path.exists(os.path.join(restored_face_dir, f'{basename}.{img_save_ext}')):
                if args.skip_if_exist:
                    print(f"Exists, skip face image {basename}...")
                    continue
                else:
                    raise RuntimeError(f"Image {basename} already exist")
            
            try:
                preds, stage1_preds = process(
                    model, face_helper.cropped_faces, steps=args.steps,
                    strength=1,
                    color_fix_type=args.color_fix_type,
                    disable_preprocess_model=args.disable_preprocess_model,
                    cond_fn=None, tiled=False, tile_size=None, tile_stride=None
                )
            except RuntimeError as e:
                # Avoid cuda_out_of_memory error.
                print(f"{file_path}, error: {e}")
                continue
            
            for restored_face in preds:
                # unused stage1 preds
                # face_helper.add_restored_face(np.array(stage1_restored_face))
                face_helper.add_restored_face(np.array(restored_face))

            # paste face back to the image
            if not args.has_aligned:
                # upsample the background
                if bg_upsampler is not None:
                    print(f'upsampling the background image using {args.bg_upsampler}...')
                    if args.bg_upsampler == 'DiffBIR':
                        bg_img, _ = process(
                            bg_upsampler, [x], steps=args.steps,
                            color_fix_type=args.color_fix_type,
                            strength=1, disable_preprocess_model=args.disable_preprocess_model,
                            cond_fn=None, tiled=False, tile_size=None, tile_stride=None)
                        bg_img= bg_img[0]
                    elif args.bg_upsampler == 'RealESRGAN':
                        # resize back to the original size
                        w, h = x.shape[:2]
                        input_size = (int(w/args.sr_scale), int(h/args.sr_scale))
                        x = Image.fromarray(x).resize(input_size, Image.LANCZOS)
                        bg_img = bg_upsampler.enhance(np.array(x), outscale=args.sr_scale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)

                # paste each restored face to the input image
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img
                )

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                # save cropped face
                if not args.has_aligned: 
                    save_crop_path = os.path.join(cropped_face_dir, f'{basename}_{idx:02d}.{img_save_ext}')
                    Image.fromarray(cropped_face).save(save_crop_path)
                # save restored face
                if args.has_aligned:
                    save_face_name = f'{basename}.{img_save_ext}'
                    # remove padding
                    restored_face = restored_face[:lq_resized.height, :lq_resized.width, :]
                else:
                    save_face_name = f'{basename}_{idx:02d}.{img_save_ext}'
                save_restore_path = os.path.join(restored_face_dir, save_face_name)
                Image.fromarray(restored_face).save(save_restore_path)

            # save restored whole image
            if not args.has_aligned:
                # remove padding
                restored_img = restored_img[:lq_resized.height, :lq_resized.width, :]
                # save restored image
                Image.fromarray(restored_img).resize(lq.size, Image.LANCZOS).convert("RGB").save(restored_img_path)
            print(f"Face image {basename} saved to {output_parent_dir}")


if __name__ == "__main__":
    main()
