import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import os
import os.path as osp
from time import perf_counter
from utils import util


class projector:
    def __init__(self, config, reg_noise_wgt, device: torch.device, outdir):
        self.device = device
        self.projector_params = config["projector_params"]["general"]
        self.batch_size = self.projector_params["batch_size"]
        self.num_steps = self.projector_params["num_steps"]
        self.w_avg_samples = self.projector_params["w_avg_samples"]
        self.initial_learning_rate = self.projector_params["initial_learning_rate"]
        self.initial_noise_factor = self.projector_params["initial_noise_factor"]
        self.lr_rampdown_length = self.projector_params["lr_rampdown_length"]
        self.lr_rampup_length = self.projector_params["lr_rampup_length"]
        self.noise_ramp_length = self.projector_params["noise_ramp_length"]
        self.regularize_noise_weight = reg_noise_wgt
        self.verbose = self.projector_params["verbose"]
        self.outdir = outdir
        self.config = config

    def logprint(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def init(self, num_images=None):
        num_images = num_images or self.batch_size
        self.logprint(
            f"Computing W midpoint and stddev using {self.w_avg_samples} samples..."
        )
        z_samples = np.random.RandomState(123).randn(self.w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to(self.device), None)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        self.w_avg = np.mean(w_samples, axis=0, keepdims=True)
        self.w_std = (np.sum((w_samples - self.w_avg) ** 2) / self.w_avg_samples) ** 0.5

        self.noise_bufs = {
            name: buf
            for (name, buf) in self.G.synthesis.named_buffers()
            if "noise_const" in name
        }

        self.w_opt = torch.tensor(
            np.repeat(self.w_avg, repeats=num_images, axis=0),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.w_out = torch.zeros(
            [num_images] + list(self.w_opt.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            [self.w_opt] + list(self.noise_bufs.values()),
            betas=(0.9, 0.999),
            lr=self.initial_learning_rate,
        )
        for buf in self.noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    def execute_steps(self):
        for step in range(self.num_steps):

            # Learning rate schedule.
            t = step / self.num_steps
            w_noise_scale = (
                self.w_std
                * self.initial_noise_factor
                * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            )
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(self.w_opt) * w_noise_scale
            ws = (self.w_opt + w_noise).repeat([1, self.G.mapping.num_ws, 1])
            synth_images = self.G.synthesis(ws, noise_mode="const")

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

            # Features for synth images.
            synth_features = self.D(
                synth_images, resize_images=False, return_lpips=True
            )
            dist = (self.target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in self.noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (
                        noise * torch.roll(noise, shifts=1, dims=3)
                    ).mean() ** 2
                    reg_loss += (
                        noise * torch.roll(noise, shifts=1, dims=2)
                    ).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * float(self.regularize_noise_weight)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.logprint(
                f"step {step+1:>4d}/{self.num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}"
            )

            # Save projected W for each optimization step.
            w_out = self.w_opt.detach()

            # Normalize noise.
            with torch.no_grad():
                for buf in self.noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        return w_out.repeat([1, self.G.mapping.num_ws, 1])

    def project(self, dataloader):
        for _indx, data in enumerate(dataloader):
            start_time = perf_counter()

            seed = 303
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.G, self.D = util.load_networks(self.config, self.device)

            images, names = data

            self.init(num_images=len(images))

            self.print_mem_stats()

            target_images = images.to(self.device).to(torch.float32)

            if target_images.shape[2] > 256:
                target_images = F.interpolate(
                    target_images, size=(256, 256), mode="area"
                )

            self.target_features = self.D(
                target_images, resize_images=False, return_lpips=True
            )
            projections = self.execute_steps()
            self.save_projections(projections, names)
            self.logprint(
                f"Elapsed: {(perf_counter()-start_time):.1f} s for batch: {_indx}\nImages Projected: {names}"
            )
        return True

    def save_projections(self, projections, names):

        for i in range(np.shape(projections)[0]):
            projected_w = projections[i, :, :]
            synth_image = self.G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            name_arr = names[i].split(".")
            save_to = osp.join(self.outdir, name_arr[0], self.regularize_noise_weight)
            os.makedirs(save_to, exist_ok=True)
            PIL.Image.fromarray(synth_image, "RGB").save(
                f"{save_to}/projected.{name_arr[1]}"
            )
    
    def print_mem_stats(self):
        # setting device on GPU if available, else CPU
        print('Current device:', torch.cuda.device_count(), flush=True)
        print('Using device:', self.device, flush=True)

        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(), flush=True)
            print('\nMemory Usage:\n', flush=True)
            print('Allocated:', round(torch.cuda.memory_allocated()/1024**3,1), 'GB', flush=True)
            print('Cached:   ', round(torch.cuda.memory_reserved()/1024**3,1), 'GB', flush=True)

        print(torch.cuda.memory_summary(device=self.device, abbreviated=True), flush=True)