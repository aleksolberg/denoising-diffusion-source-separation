from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from PIL import Image
import torchvision

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 100,           # number of steps
    sampling_timesteps = 100,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1',            # L1 or L2
    
)

trainer = Trainer(
    diffusion,
    source_folder = 'datasets/randomMIDI/PianoViolin11025/jpeg_amp_only/train/ins3',
    mix_folder = 'datasets/randomMIDI/PianoViolin11025/jpeg_amp_only/train/mix',
    train_batch_size = 2,
    train_lr = 4e-5,
    train_num_steps = 150,           # total training steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    num_samples = 1,
    save_and_sample_every = 20
)

trainer.train()

sampled_image = diffusion.sample(batch_size = 1)
torchvision.utils.save_image(sampled_image, 'spectrogram.png')