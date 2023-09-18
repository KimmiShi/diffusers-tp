import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from apex.optimizers import FusedAdam
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import time
import functools

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def batch_inp(bs20inp, target_bs):
    mbs=2
    bs4inp = bs20inp[:mbs]
    if target_bs<mbs:
        return bs20inp[:target_bs]
    if target_bs==mbs:
        return bs20inp
    num = int(target_bs/mbs)
    out = torch.cat([bs4inp.clone().detach() for _ in  range(num)])
    if out.dim()==4:
        out = out.to(memory_format=torch.channels_last).contiguous()
    print(out.shape)
    return out



def train(model, vae, optimizer_class, batchsize, use_zero=False, use_amp=True, h=512, w=512, is_xl=False):
    timesteps = torch.arange(batchsize, dtype=torch.int64).cuda()+100
    prompt_embeds = torch.rand([batchsize,77,768], dtype=torch.float16).cuda()
    time_ids = torch.rand([batchsize,6], dtype=torch.float16).cuda()
    text_embeds = torch.rand([batchsize,1280], dtype=torch.float16).cuda()
    encoder_hidden_states = torch.rand([batchsize,77,768], dtype=torch.float32).cuda()

    model_input = torch.rand([batchsize, 3, h, w], dtype=torch.float32).cuda()

    if not use_amp:
        prompt_embeds = prompt_embeds.float()
        text_embeds = text_embeds.float()
        time_ids = time_ids.float()

    unet_added_conditions = {
        "time_ids": time_ids,
        "text_embeds": text_embeds
    }


    model.enable_gradient_checkpointing()
    # model.enable_xformers_memory_efficient_attention()
    # torch._dynamo.config.suppress_errors = True
    # model=torch.compile(model)
    perf_times = []
    if use_zero:
        # model = DDP(model)
        model = FSDP(model, sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP)
        opt =optimizer_class(model.parameters())
        # opt = ZeroRedundancyOptimizer(model.parameters(),
        #                                 optimizer_class=optimizer_class,
        #                                 parameters_as_bucket_view=True,)
    else:
        opt =optimizer_class(model.parameters())

    from torch.profiler import profile, record_function, ProfilerActivity

    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=0, warmup=5, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof/unet_720p_tp2'),
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True)
    # prof.start()
    for ind in range(20):
        # if ind==5:
        beg = time.time()

        torch.cuda.synchronize()
        vae_beg = time.time()
        with torch.no_grad():
            noisy_model_input = vae.encode(model_input).latent_dist.sample().mul_(0.18215)
        torch.cuda.synchronize()
        print("vae time:", time.time()-vae_beg)


        with torch.autocast(dtype=torch.float16, device_type='cuda', enabled=use_amp):
        # with accelerator.accumulate(unet):
            # import pdb;pdb.set_trace()
            # print("before fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
            # with torch.no_grad():

            # model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states).sample
            # print("after fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
            # import pdb;pdb.set_trace()
            # loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
            # print("after fwd", torch.cuda.max_memory_allocated()/1e9)
            if is_xl:
                model_pred = model(
                            noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                        ).sample
                # print("after fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
                # import pdb;pdb.set_trace()
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
            else:
                model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
                # print("after fwd", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
                # import pdb;pdb.set_trace()
                loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")

        # loss = F.mse_loss(model_pred.float(), torch.rand_like(model_pred).float(), reduction="mean")
        # print(loss)

        loss.backward()
        # print("after bwd", torch.cuda.max_memory_allocated()/1e9)

        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        if ind>10:
           perf_times.append(time.time()-beg)
        beg=time.time()
        # prof.step()

    # if torch.distributed.get_rank()==0:
    print("max mem", torch.cuda.max_memory_allocated()/1e9)
    print(perf_times)
    # prof.stop()



enable_tf32()
# rank, world_size, port, addr=setup_distributed_slurm()

pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"

pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=None,
    low_cpu_mem_usage=False, device_map=None
).cuda()
unet.train()
# unet = unet.to(memory_format=torch.channels_last)

vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()


# optimizer_class = FusedAdam
optimizer_class = functools.partial(torch.optim.Adam,fused = True)
# optimizer_class = torch.optim.AdamW
train(unet, vae, optimizer_class,  16,
      use_amp=True, use_zero=False, h=512, w=512, is_xl ='xl' in pretrained_model_name_or_path)
