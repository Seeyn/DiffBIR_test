from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
from einops import rearrange, repeat
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision import transforms as tt
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from taming.modules.losses.vqperceptual import * 
from basicsr.losses.gan_loss import *
from basicsr.losses.basic_loss import *
from torchvision.ops import roi_align
from .discriminator import *
from utils.common import instantiate_from_config, load_state_dict
        

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        
        # instantiate preprocess module (SwinIR)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        x_samples = samples.clamp(0, 1)
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        log["samples"] = x_samples

        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass


class ControlLDMwDiscriminator(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        *args,
        **kwargs
    ) -> "ControlLDMwDiscriminator":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.discriminator = NLayerDiscriminator(input_nc=4,
                                                 n_layers=3,
                                                 use_actnorm=False
                                                 ).apply(weights_init)
        self.gan_loss = GANLoss('wgan_softplus',loss_weight = 0.1)
        # instantiate preprocess module (SwinIR)
        self.current_iter = 0
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)



    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        x_samples = samples.clamp(0, 1)
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        log["samples"] = x_samples

        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        return [opt,opt_d], []
        #return [opt],[]

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
    
    def training_step(self, batch, batch_idx,optimizer_idx=None):
        '''get data'''
        self.current_iter = batch_idx
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch,optimizer_idx)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def shared_step(self, batch,optimizer_idx,**kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c,optimizer_idx)
        return loss
    

    def forward(self, x, c, optimizer_idx,*args, **kwargs):
        if optimizer_idx == 0 or optimizer_idx is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
            if self.model.conditioning_key is not None:
                assert c is not None
                if self.cond_stage_trainable:
                    c = self.get_learned_conditioning(c)
                if self.shorten_cond_schedule:  # TODO: drop this option
                    tc = self.cond_ids[t].to(self.device)
                    c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

            loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs)
            
            t = torch.randint(0, 200, (x.shape[0],), device=self.device).long()
            #t = torch.zeros((x.shape[0],),device=self.device).long() + 200
            if self.model.conditioning_key is not None:
                assert c is not None
                if self.cond_stage_trainable:
                    c = self.get_learned_conditioning(c)
                if self.shorten_cond_schedule:  # TODO: drop this option
                    tc = self.cond_ids[t].to(self.device)
                    c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
            loss_, loss_dict_ =  self.d_losses(x,c,t,noise=None,optimize_d=False)
            
            loss += loss_
            loss_dict.update(loss_dict_)
            
        else:
            t = torch.randint(0, 200, (x.shape[0],), device=self.device).long()
            #t = torch.zeros((x.shape[0],),device=self.device).long() + 200

            if self.model.conditioning_key is not None:
                assert c is not None
                if self.cond_stage_trainable:
                    c = self.get_learned_conditioning(c)
                if self.shorten_cond_schedule:  # TODO: drop this option
                    tc = self.cond_ids[t].to(self.device)
                    c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
            loss, loss_dict =  self.d_losses(x,c,t,noise=None,optimize_d=True)
            
        
        return loss,loss_dict

     
    
    def d_losses(self, x_start, cond, t, noise=None, optimize_d = False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        model_output_ = model_output
        model_output = self.predict_start_from_noise(x_noisy,t,model_output)
        
        target = x_start
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if optimize_d:
            fake_d_pred = self.discriminator(model_output.detach())
            real_d_pred = self.discriminator(target)
            l_d = self.gan_loss(real_d_pred, True, is_disc=True) + self.gan_loss(fake_d_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d
            # In WGAN, real_score should be positive and fake_score should be negative
            loss_dict['real_score'] = real_d_pred.detach().mean()
            loss_dict['fake_score'] = fake_d_pred.detach().mean()
            
            if self.current_iter % 16 == 0:
                target.requires_grad = True
                real_pred = self.discriminator(target)
                l_d_r1 = r1_penalty(real_pred, target)
                l_d_r1 = (10. / 2 * l_d_r1 * 16 + 0 * real_pred[0])
                loss_dict['l_d_r1'] = l_d_r1.detach().mean()
                #print(l_d_r1.shape,l_d.shape)
                l_d += l_d_r1.mean()

            
            return l_d, loss_dict
        
        else:

            fake_g_pred = self.discriminator(model_output)
            l_g_gan = self.gan_loss(fake_g_pred,True, is_disc = False)
            loss_dict['l_g_gan'] = l_g_gan

            loss_simple = self.get_loss(model_output_, noise, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            # loss = loss_simple / torch.exp(self.logvar) + self.logvar
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})

            loss = self.l_simple_weight * loss.mean()

            loss_vlb = self.get_loss(model_output_, noise, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)
            loss_dict.update({f'{prefix}/loss': loss})
            loss += l_g_gan

            return loss, loss_dict
    

    
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        # model_output = self.predict_start_from_noise(x_noisy,t,model_output)
        #print(model_output.shape)  3,512,512

                                               
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        # target = x_start
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})      

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict



class ControlLDMwLocalDiscriminator(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        preprocess_config,
        *args,
        **kwargs
    ) -> "ControlLDMwLocalDiscriminator":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)

        self.automatic_optimization = False


        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        

        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()
        self.net_d_left_eye.train()
        self.net_d_right_eye.train()
        self.net_d_mouth.train()
        self.mouths = None

        self.gan_loss = GANLoss('vanilla',real_label_val=1.0, fake_label_val=0.0,loss_weight = 0.1)
        self.cri_l1 = L1Loss(loss_weight=1.0, reduction='mean')
        # instantiate preprocess module (SwinIR)
        self.current_iter = 0
        self.preprocess_model = instantiate_from_config(preprocess_config)
        frozen_module(self.preprocess_model)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        frozen_module(self.cond_encoder)



    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        self.gt = batch['jpg'].permute(0,3,1,2)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        if 'loc_left_eye' in batch:
            self.loc_left_eyes = batch['loc_left_eye']
            self.loc_right_eyes = batch['loc_right_eye']
            self.loc_mouths = batch['loc_mouth']
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        # log["control"] = c_cat
        # log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = c_lq
        # log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        x_samples = samples.clamp(0, 1)
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        log["samples"] = x_samples
        ''' 
        if self.mouths is not None:
            log["left_eye"] = (self.left_eyes.detach() +1 )/2
            log["left_eye_gt"] = (self.left_eyes_gt.detach() +1 )/2
            print(log["left_eye"].max(),log["left_eye"].min())
        '''
        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
   
        self.optimizer_d_left_eye = torch.optim.AdamW(self.net_d_left_eye.parameters(), lr=lr)
        self.optimizer_d_right_eye = torch.optim.AdamW(self.net_d_right_eye.parameters(), lr=lr)
        self.optimizer_d_mouth = torch.optim.AdamW(self.net_d_mouth.parameters(), lr=lr)
  
        return opt,self.optimizer_d_left_eye,self.optimizer_d_right_eye,self.optimizer_d_mouth 
        #return [opt],[]

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
    
    def training_step(self, batch, batch_idx):
        '''get data'''
        self.current_iter = batch_idx
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss_dict = self.shared_step(batch)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        
    
    def shared_step(self, batch,**kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        optimizer_g, optimizer_d_left_eye, optimizer_d_right_eye, optimizer_d_mouth = self.optimizers()


        # Diffusion loss
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=False)
        loss_total, loss_dict = 0,{}

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)

        # Generator loss
        t = torch.randint(0, 200, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=True)

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)

        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=False)
        loss_total += loss
        loss_dict.update(loss_dict_)
        
        optimizer_g.zero_grad()
        self.manual_backward(loss_total)
        optimizer_g.step()

        # Discriminator loss
        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=True)
        loss_dict.update(loss_dict_)

        optimizer_d_left_eye.zero_grad()
        optimizer_d_right_eye.zero_grad()
        optimizer_d_mouth.zero_grad()
        self.manual_backward(loss)
        optimizer_d_left_eye.step()
        optimizer_d_right_eye.step()
        optimizer_d_mouth.step()

        return loss_dict
    

    def forward(self, x, c,t, img_output=False,noise=None,*args, **kwargs):
        ####Generator
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        
        noise = default(noise,lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, c)
        if img_output:
            model_output_ = self.predict_start_from_noise(x_noisy,t,model_output)
            self.output =self.decode_first_stage(model_output_)

        return model_output,noise

     
    
    def d_losses(self, model_output, t,noise=None, optimize_d = False):
        
        #get location
        self.get_roi_regions()
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if optimize_d:
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye

            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye

            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth


            return l_d_left_eye + l_d_right_eye + loss_dict['l_d_mouth'], loss_dict
        
        else:
            l_g_total = 0
            fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_left_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_left_eye'] = l_g_gan
            # right eye
            fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_right_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_right_eye'] = l_g_gan
            # mouth
            fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
            l_g_gan = self.gan_loss(fake_mouth, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_mouth'] = l_g_gan

            _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
            _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
            _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

            def _comp_style(feat, feat_gt, criterion):
                return criterion(self._gram_mat(feat[0]), self._gram_mat(
                    feat_gt[0].detach())) * 0.5 + criterion(
                        self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

            # facial component style loss
            comp_style_loss = 0
            comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
            comp_style_loss = comp_style_loss * 200
            l_g_total += comp_style_loss
            loss_dict['l_g_comp_style_loss'] = comp_style_loss

            return l_g_total, loss_dict
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
    def p_losses(self, model_output, t,noise=None):
                 
        loss_dict = {}
        prefix = 'train'
        target = noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})      

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(512 / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images

        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape)
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape,self.left_eyes_gt.shape,self.right_eyes_gt.shape,self.mouths_gt.shape)
        # output
        #print('sr',(self.output.max(),self.output.min()),self.output.shape)
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

        #print('sr',(self.output.max(),self.output.min()),self.output.shape,self.left_eyes.shape,self.right_eyes.shape,self.mouths.shape)







class TwoStreamControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, learning_rate,
        preprocess_config, sd_locked=True, sync_path=None, synch_control=True,
                 control_mode='canny', ckpt_path_ctr=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)

        self.automatic_optimization = False

        self.control_key = control_key
        self.sd_locked = sd_locked
        self.control_mode = control_mode

        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        

        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()
        self.net_d_left_eye.train()
        self.net_d_right_eye.train()
        self.net_d_mouth.train()
        self.mouths = None

        self.gan_loss = GANLoss('vanilla',real_label_val=1.0, fake_label_val=0.0,loss_weight = 0.1)
        self.cri_l1 = L1Loss(loss_weight=1.0, reduction='mean')
        # instantiate preprocess module (SwinIR)
        self.current_iter = 0
        self.preprocess_model = instantiate_from_config(preprocess_config)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load('/lab/tangb_lab/12132338/BSR/DiffBIR_test/checkpoint/face_swinir_v1.ckpt').items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.preprocess_model.load_state_dict(new_state_dict)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))

        if ckpt_path_ctr is not None:
            self.load_control_ckpt(ckpt_path_ctr=ckpt_path_ctr)

        frozen_module(self.preprocess_model)
        frozen_module(self.cond_encoder)


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        self.gt = batch['jpg'].permute(0,3,1,2)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        if 'loc_left_eye' in batch:
            self.loc_left_eyes = batch['loc_left_eye']
            self.loc_right_eyes = batch['loc_right_eye']
            self.loc_mouths = batch['loc_mouth']
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])


    def apply_model(self, x_noisy, t, cond, precomputed_hint=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1) if self.control_mode != 'multi_control' else cond['c_latent'][0]

        eps = self.control_model(
            x=x_noisy, hint=cond_hint, timesteps=t, context=cond_txt, base_model=diffusion_model, precomputed_hint=precomputed_hint
        )

        return eps

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, hint=None):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if hint is None:
            return self.first_stage_model.decode(z)
        else:
            return self.first_stage_model.decode(z, hint=hint)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        '''
        log["control"] = c_cat
        try:
            log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        except:
            print('heihei')
        '''
        log["lq"] = c_lq
        # log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        x_samples = samples.clamp(0, 1)
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        log["samples"] = x_samples
        ''' 
        if self.mouths is not None:
            log["left_eye"] = (self.left_eyes.detach() +1 )/2
            log["left_eye_gt"] = (self.left_eyes_gt.detach() +1 )/2
            print(log["left_eye"].max(),log["left_eye"].min())
        '''
        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        '''
        params = params + list(self.enc_zero_convs_out.parameters())
        params = params + list(self.enc_zero_convs_in.parameters())
        params = params + list(self.model.middle_block_out.parameters())
        params = params + list(self.model.middle_block_in.parameters())
        params = params + list(self.model.dec_zero_convs_out.parameters())
        params = params + list(self.model.dec_zero_convs_in.parameters())
        params = params + list(self.model.input_hint_block.parameters())
        '''
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
   
        self.optimizer_d_left_eye = torch.optim.AdamW(self.net_d_left_eye.parameters(), lr=lr)
        self.optimizer_d_right_eye = torch.optim.AdamW(self.net_d_right_eye.parameters(), lr=lr)
        self.optimizer_d_mouth = torch.optim.AdamW(self.net_d_mouth.parameters(), lr=lr)
  
        return opt,self.optimizer_d_left_eye,self.optimizer_d_right_eye,self.optimizer_d_mouth 
        #return [opt],[]

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
    
    def training_step(self, batch, batch_idx):
        '''get data'''
        self.current_iter = batch_idx
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss_dict = self.shared_step(batch)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        
    
    def shared_step(self, batch,**kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        optimizer_g, optimizer_d_left_eye, optimizer_d_right_eye, optimizer_d_mouth = self.optimizers()


        # Diffusion loss
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=False)
        loss_total, loss_dict = 0,{}

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)
        
        '''
        # Generator loss
        t = torch.randint(0, 200, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=True)

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)
    
        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=False)
        loss_total += loss
        loss_dict.update(loss_dict_)
        '''
        optimizer_g.zero_grad()
        self.manual_backward(loss_total)
        optimizer_g.step()
        '''
        # Discriminator loss
        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=True)
        loss_dict.update(loss_dict_)

        optimizer_d_left_eye.zero_grad()
        optimizer_d_right_eye.zero_grad()
        optimizer_d_mouth.zero_grad()
        self.manual_backward(loss)
        optimizer_d_left_eye.step()
        optimizer_d_right_eye.step()
        optimizer_d_mouth.step()
        '''
        return loss_dict
    

    def forward(self, x, c,t, img_output=False,noise=None,*args, **kwargs):
        ####Generator
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        
        noise = default(noise,lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, c)
        if img_output:
            model_output_ = self.predict_start_from_noise(x_noisy,t,model_output)
            self.output =self.decode_first_stage(model_output_)

        return model_output,noise

     
    
    def d_losses(self, model_output, t,noise=None, optimize_d = False):
        
        #get location
        self.get_roi_regions()
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if optimize_d:
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye

            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye

            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth


            return l_d_left_eye + l_d_right_eye + loss_dict['l_d_mouth'], loss_dict
        
        else:
            l_g_total = 0
            fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_left_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_left_eye'] = l_g_gan
            # right eye
            fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_right_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_right_eye'] = l_g_gan
            # mouth
            fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
            l_g_gan = self.gan_loss(fake_mouth, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_mouth'] = l_g_gan

            _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
            _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
            _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

            def _comp_style(feat, feat_gt, criterion):
                return criterion(self._gram_mat(feat[0]), self._gram_mat(
                    feat_gt[0].detach())) * 0.5 + criterion(
                        self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

            # facial component style loss
            comp_style_loss = 0
            comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
            comp_style_loss = comp_style_loss * 200
            l_g_total += comp_style_loss
            loss_dict['l_g_comp_style_loss'] = comp_style_loss

            return l_g_total, loss_dict
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
    def p_losses(self, model_output, t,noise=None):
                 
        loss_dict = {}
        prefix = 'train'
        target = noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})      

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(512 / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images

        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape)
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape,self.left_eyes_gt.shape,self.right_eyes_gt.shape,self.mouths_gt.shape)
        # output
        #print('sr',(self.output.max(),self.output.min()),self.output.shape)
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

        #print('sr',(self.output.max(),self.output.min()),self.output.shape,self.left_eyes.shape,self.right_eyes.shape,self.mouths.shape)

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_separate_weights(self, ckpt_path_ctr, ckpt_path_base):
        ckpt_base = torch.load(ckpt_path_base)
        ckpt_ctr = torch.load(ckpt_path_ctr)
        mutual_state_dict = dict()
        mutual_state_dict.update(ckpt_base['state_dict'])
        mutual_state_dict.update(ckpt_ctr['state_dict'])

        self.load_state_dict(mutual_state_dict, strict=False)
        print(['WEIGHTS LOADED'])

    def load_control_ckpt(self, ckpt_path_ctr):
        ckpt = torch.load(ckpt_path_ctr)
        load_state_dict(self,ckpt,strict=False)
        #self.load_state_dict(ckpt['state_dict'], strict=True)
        print(['CONTROL WEIGHTS LOADED',ckpt_path_ctr])

    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)  # load the base model checkpoints

        if synch_control:
            # add copy for control_model weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        print('control_model.control' + key[15:],':',ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape,ckpt_base['state_dict'][key].shape,self.state_dict()['control_model.control' + key[15:]].shape)
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            '''
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                #print('0:control_model.control' + key[15:],':',control_dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 1
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                #print('1:control_model.control' + key[15:],':',control_dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:, :control_dim, ...]
                            '''
                            control_w_s = self.state_dict()['control_model.control' + key[15:]].shape
                            dims = []
                            for index, each in enumerate(ckpt_base['state_dict'][key].shape):
                                if each != control_w_s[index]:
                                    dims.append(control_w_s[index])
                            weight = ckpt_base['state_dict'][key]
                            
                            for index, dim in enumerate(dims):
                                weight = torch.cat([weight,weight],index)
                                if index == 0:
                                    weight = weight[:dim,...]
                                else:
                                    weight = weight[:,:dim,...]
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = weight
                        else:
                            #print('haha')
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]

        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')

class TwoStreamControlLDMCFW(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, learning_rate,
        preprocess_config, sd_locked=True, sync_path=None, synch_control=True,
                 control_mode='canny', ckpt_path_ctr=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)

        self.automatic_optimization = False

        self.control_key = control_key
        self.sd_locked = sd_locked
        self.control_mode = control_mode

        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        

        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()
        self.net_d_left_eye.train()
        self.net_d_right_eye.train()
        self.net_d_mouth.train()
        self.mouths = None

        self.gan_loss = GANLoss('vanilla',real_label_val=1.0, fake_label_val=0.0,loss_weight = 0.1)
        self.cri_l1 = L1Loss(loss_weight=1.0, reduction='mean')
        # instantiate preprocess module (SwinIR)
        self.current_iter = 0
        self.preprocess_model = instantiate_from_config(preprocess_config)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load('/lab/tangb_lab/12132338/BSR/DiffBIR_test/checkpoint/face_swinir_v1.ckpt').items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.preprocess_model.load_state_dict(new_state_dict)
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))

        if ckpt_path_ctr is not None:
            self.load_control_ckpt(ckpt_path_ctr=ckpt_path_ctr)

        frozen_module(self.preprocess_model)
        frozen_module(self.cond_encoder)


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        self.gt = batch['jpg'].permute(0,3,1,2)
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply preprocess model
        control = self.preprocess_model(control)
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)
        if 'loc_left_eye' in batch:
            self.loc_left_eyes = batch['loc_left_eye']
            self.loc_right_eyes = batch['loc_right_eye']
            self.loc_mouths = batch['loc_mouth']
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control])


    def apply_model(self, x_noisy, t, cond, precomputed_hint=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1) if self.control_mode != 'multi_control' else cond['c_latent'][0]

        eps = self.control_model(
            x=x_noisy, hint=cond_hint, timesteps=t, context=cond_txt, base_model=diffusion_model, precomputed_hint=precomputed_hint
        )

        return eps

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, hint=None):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        if hint is None:
            return self.first_stage_model.decode(z)
        else:
            return self.first_stage_model.decode(z, hint=hint)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        '''
        log["control"] = c_cat
        try:
            log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        except:
            print('heihei')
        '''
        log["lq"] = c_lq
        # log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        
        
        samples = self.sample_log(
            # TODO: remove c_concat from cond
            c_cat,steps=sample_steps
        )
        #x_samples = self.decode_first_stage(samples)
        x_samples = samples.clamp(0, 1)
        #x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        log["samples"] = x_samples
        ''' 
        if self.mouths is not None:
            log["left_eye"] = (self.left_eyes.detach() +1 )/2
            log["left_eye_gt"] = (self.left_eyes_gt.detach() +1 )/2
            print(log["left_eye"].max(),log["left_eye"].min())
        '''
        return log

    @torch.no_grad()
    def sample_log(self, control, steps):
        sampler = SpacedSampler(self)
        height, width = control.size(-2), control.size(-1)
        n_samples = len(control)
        shape = (n_samples, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.device, dtype=torch.float32)
        samples = sampler.sample(
            steps, shape, cond_img=control,
            positive_prompt="", negative_prompt="", x_T=x_T,
            cfg_scale=1.0, cond_fn=None,
            color_fix_type='wavelet'
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        '''
        params = params + list(self.enc_zero_convs_out.parameters())
        params = params + list(self.enc_zero_convs_in.parameters())
        params = params + list(self.model.middle_block_out.parameters())
        params = params + list(self.model.middle_block_in.parameters())
        params = params + list(self.model.dec_zero_convs_out.parameters())
        params = params + list(self.model.dec_zero_convs_in.parameters())
        params = params + list(self.model.input_hint_block.parameters())
        '''
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
   
        self.optimizer_d_left_eye = torch.optim.AdamW(self.net_d_left_eye.parameters(), lr=lr)
        self.optimizer_d_right_eye = torch.optim.AdamW(self.net_d_right_eye.parameters(), lr=lr)
        self.optimizer_d_mouth = torch.optim.AdamW(self.net_d_mouth.parameters(), lr=lr)
  
        return opt,self.optimizer_d_left_eye,self.optimizer_d_right_eye,self.optimizer_d_mouth 
        #return [opt],[]

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
    
    def training_step(self, batch, batch_idx):
        '''get data'''
        self.current_iter = batch_idx
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss_dict = self.shared_step(batch)
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        
    
    def shared_step(self, batch,**kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        optimizer_g, optimizer_d_left_eye, optimizer_d_right_eye, optimizer_d_mouth = self.optimizers()


        # Diffusion loss
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=False)
        loss_total, loss_dict = 0,{}

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)
        
        '''
        # Generator loss
        t = torch.randint(0, 200, (x.shape[0],), device=self.device).long()
        model_output,noise = self(x, c,t,img_output=True)

        loss, loss_dict_ = self.p_losses(model_output,t,noise)
        loss_total += loss
        loss_dict.update(loss_dict_)
    
        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=False)
        loss_total += loss
        loss_dict.update(loss_dict_)
        '''
        optimizer_g.zero_grad()
        self.manual_backward(loss_total)
        optimizer_g.step()
        '''
        # Discriminator loss
        loss, loss_dict_ = self.d_losses(model_output,t,noise,optimize_d=True)
        loss_dict.update(loss_dict_)

        optimizer_d_left_eye.zero_grad()
        optimizer_d_right_eye.zero_grad()
        optimizer_d_mouth.zero_grad()
        self.manual_backward(loss)
        optimizer_d_left_eye.step()
        optimizer_d_right_eye.step()
        optimizer_d_mouth.step()
        '''
        return loss_dict
    

    def forward(self, x, c,t, img_output=False,noise=None,*args, **kwargs):
        ####Generator
        
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        
        noise = default(noise,lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, c)
        if img_output:
            model_output_ = self.predict_start_from_noise(x_noisy,t,model_output)
            self.output =self.decode_first_stage(model_output_)

        return model_output,noise

     
    
    def d_losses(self, model_output, t,noise=None, optimize_d = False):
        
        #get location
        self.get_roi_regions()
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if optimize_d:
            fake_d_pred, _ = self.net_d_left_eye(self.left_eyes.detach())
            real_d_pred, _ = self.net_d_left_eye(self.left_eyes_gt)
            l_d_left_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye

            # right eye
            fake_d_pred, _ = self.net_d_right_eye(self.right_eyes.detach())
            real_d_pred, _ = self.net_d_right_eye(self.right_eyes_gt)
            l_d_right_eye = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye

            # mouth
            fake_d_pred, _ = self.net_d_mouth(self.mouths.detach())
            real_d_pred, _ = self.net_d_mouth(self.mouths_gt)
            l_d_mouth = self.gan_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth


            return l_d_left_eye + l_d_right_eye + loss_dict['l_d_mouth'], loss_dict
        
        else:
            l_g_total = 0
            fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(self.left_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_left_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_left_eye'] = l_g_gan
            # right eye
            fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(self.right_eyes, return_feats=True)
            l_g_gan = self.gan_loss(fake_right_eye, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_right_eye'] = l_g_gan
            # mouth
            fake_mouth, fake_mouth_feats = self.net_d_mouth(self.mouths, return_feats=True)
            l_g_gan = self.gan_loss(fake_mouth, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan_mouth'] = l_g_gan

            _, real_left_eye_feats = self.net_d_left_eye(self.left_eyes_gt, return_feats=True)
            _, real_right_eye_feats = self.net_d_right_eye(self.right_eyes_gt, return_feats=True)
            _, real_mouth_feats = self.net_d_mouth(self.mouths_gt, return_feats=True)

            def _comp_style(feat, feat_gt, criterion):
                return criterion(self._gram_mat(feat[0]), self._gram_mat(
                    feat_gt[0].detach())) * 0.5 + criterion(
                        self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

            # facial component style loss
            comp_style_loss = 0
            comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
            comp_style_loss = comp_style_loss * 200
            l_g_total += comp_style_loss
            loss_dict['l_g_comp_style_loss'] = comp_style_loss

            return l_g_total, loss_dict
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
    def p_losses(self, model_output, t,noise=None):
                 
        loss_dict = {}
        prefix = 'train'
        target = noise
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})      

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        face_ratio = int(512 / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images

        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape)
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        #print('gt',(self.gt.max(),self.gt.min()),self.gt.shape,self.left_eyes_gt.shape,self.right_eyes_gt.shape,self.mouths_gt.shape)
        # output
        #print('sr',(self.output.max(),self.output.min()),self.output.shape)
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

        #print('sr',(self.output.max(),self.output.min()),self.output.shape,self.left_eyes.shape,self.right_eyes.shape,self.mouths.shape)

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_separate_weights(self, ckpt_path_ctr, ckpt_path_base):
        ckpt_base = torch.load(ckpt_path_base)
        ckpt_ctr = torch.load(ckpt_path_ctr)
        mutual_state_dict = dict()
        mutual_state_dict.update(ckpt_base['state_dict'])
        mutual_state_dict.update(ckpt_ctr['state_dict'])

        self.load_state_dict(mutual_state_dict, strict=False)
        print(['WEIGHTS LOADED'])

    def load_control_ckpt(self, ckpt_path_ctr):
        ckpt = torch.load(ckpt_path_ctr)
        load_state_dict(self,ckpt,strict=False)
        #self.load_state_dict(ckpt['state_dict'], strict=True)
        print(['CONTROL WEIGHTS LOADED',ckpt_path_ctr])

    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)  # load the base model checkpoints

        if synch_control:
            # add copy for control_model weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        print('control_model.control' + key[15:],':',ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape,ckpt_base['state_dict'][key].shape,self.state_dict()['control_model.control' + key[15:]].shape)
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            '''
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                #print('0:control_model.control' + key[15:],':',control_dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 1
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                #print('1:control_model.control' + key[15:],':',control_dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:, :control_dim, ...]
                            '''
                            control_w_s = self.state_dict()['control_model.control' + key[15:]].shape
                            dims = []
                            for index, each in enumerate(ckpt_base['state_dict'][key].shape):
                                if each != control_w_s[index]:
                                    dims.append(control_w_s[index])
                            weight = ckpt_base['state_dict'][key]
                            
                            for index, dim in enumerate(dims):
                                weight = torch.cat([weight,weight],index)
                                if index == 0:
                                    weight = weight[:dim,...]
                                else:
                                    weight = weight[:,:dim,...]
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = weight
                        else:
                            #print('haha')
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]

        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
