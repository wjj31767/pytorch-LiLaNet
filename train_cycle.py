import os
import warnings
from argparse import ArgumentParser
from torch.autograd import Variable
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar,PiecewiseLinear, ParamGroupScheduler
from ignite.handlers import ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from ignite.contrib.handlers import TensorboardLogger, WandBLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import torch.nn.functional as F
from lilanet.datasets import CYCLEDENSE, Image2ImageDataset,Normalize, Compose, RandomHorizontalFlip
from lilanet.datasets.transforms import ToTensor
from lilanet.model import Generator,Discriminator
from lilanet.utils import save
import itertools
import random
import shutil
from functools import partial
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers):
    normalize = Normalize(mean=CYCLEDENSE.mean(), std=CYCLEDENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    val_transforms = Compose([
        ToTensor(),
        normalize
    ])

    train_loader_AB = DataLoader(Image2ImageDataset(CYCLEDENSE(
            root=data_dir,
            split='train',
            clss = 'clear',
            transform=transforms)
        , CYCLEDENSE(
            root=data_dir,
            split='train',
            clss = 'rain',
            transform=transforms))
        ,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_loader_AB = DataLoader(Image2ImageDataset(
        CYCLEDENSE(
            root=data_dir,
            split='val',
            clss="clear",
            transform=val_transforms),
        CYCLEDENSE(
            root=data_dir,
            split='val',
            clss="rain",
            transform=val_transforms)
    ),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader_AB, val_loader_AB


def run(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator_A2B = Generator(2,args.ngf,2,args.num_lila)
    generator_B2A = Generator(2,args.ngf,2,args.num_lila)
    discriminator_A = Discriminator(2, args.ndf, 1)
    discriminator_B = Discriminator(2, args.ndf, 1)
    generator_A2B.xavier_weight_init()
    generator_B2A.xavier_weight_init()
    discriminator_A.normal_weight_init(mean=0.0, std=0.02)
    discriminator_B.normal_weight_init(mean=0.0, std=0.02)
    generator_A2B.cuda()
    generator_B2A.cuda()
    discriminator_A.cuda()
    discriminator_B.cuda()

    normalize = Normalize(mean=CYCLEDENSE.mean(), std=CYCLEDENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])
    datasetA = CYCLEDENSE(
        root=args.dataset_dir,
        split='train',
        clss='clear',
        transform=transforms)
    datasetB = CYCLEDENSE(
        root=args.dataset_dir,
        split='train',
        clss='rain',
        transform=transforms)
    train_loader_AB,  val_loader_AB = get_data_loaders(
        args.dataset_dir, args.batch_size, args.val_batch_size, args.num_workers)
    for i in train_loader_AB:
        print(i['A'].shape,i['B'].shape)
        break
    # Loss function
    MSE_loss = torch.nn.MSELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()

    # optimizers
    optimizer_G = optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()), lr=args.lrD,
                             betas=(args.beta1, args.beta2))


    buffer_size = 50
    fake_a_buffer = []
    fake_b_buffer = []

    def toggle_grad(model, on_or_off):
        # https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py#L674
        for param in model.parameters():
            param.requires_grad = on_or_off
    def buffer_insert_and_get(buffer, batch):
        output_batch = []
        for b in batch:
            b = b.unsqueeze(0)
            # if buffer is not fully filled:
            if len(buffer) < buffer_size:
                output_batch.append(b)
                buffer.append(b.cpu())
            elif random.uniform(0, 1) > 0.5:
                # Add newly created image into the buffer and put ont from the buffer into the output
                random_index = random.randint(0, buffer_size - 1)
                output_batch.append(buffer[random_index].clone().to(device))
                buffer[random_index] = b.cpu()
            else:
                output_batch.append(b)
        return torch.cat(output_batch, dim=0)

    amp_enabled = True
    lambda_idt = 0.5
    lambda_cycle_A = 10.0
    lambda_cycle_B = 10.0
    amp_scaler = GradScaler(enabled=amp_enabled)
    def discriminator_forward_pass(discriminator, batch_real, batch_fake, fake_buffer):
        decision_real = discriminator(batch_real)
        batch_fake = buffer_insert_and_get(fake_buffer, batch_fake)
        decision_fake = discriminator(batch_fake)
        return decision_real, decision_fake

    def compute_loss_generator(batch_decision, batch_real, batch_fake,batch_rec, lambda_idt,lambda_cycle,loss_mode = 'lsgan'):
        # loss gan
        target = torch.ones_like(batch_decision)
        # lsgan
        if loss_mode == 'lsgan':
            loss_gan = F.mse_loss(batch_decision, target)
        # vanilla
        elif loss_mode == 'vanilla':
            loss_gan = F.binary_cross_entropy_with_logits(batch_decision,target)
        elif loss_mode == 'wgangp':
            loss = -batch_decision.mean()
        # loss Idt
        loss_idt = F.l1_loss(batch_fake,batch_real)*lambda_idt
        # loss cycle
        loss_cycle = F.l1_loss(batch_rec, batch_real) * lambda_cycle
        return loss_gan + loss_cycle+loss_idt

    def compute_loss_discriminator(decision_real, decision_fake,loss_mode='lsgan'):
        # loss = mean (D_b(y) − 1)^2 + mean D_b(G(x))^2
        if loss_mode in ['lsgan','vanilla']:
            loss = F.mse_loss(decision_fake, torch.zeros_like(decision_fake))
            loss += F.mse_loss(decision_real, torch.ones_like(decision_real))
        else:
            # wgangp
            pass
        return loss

    def update_fn(engine, batch):
        generator_A2B.train()
        generator_B2A.train()
        discriminator_A.train()
        discriminator_B.train()

        real_a = convert_tensor(batch['A'], device=device, non_blocking=True)
        real_b = convert_tensor(batch['B'], device=device, non_blocking=True)

        # Update generators

        # Disable grads computation for the discriminators:
        toggle_grad(discriminator_A, False)
        toggle_grad(discriminator_B, False)

        with autocast(enabled=amp_enabled):
            fake_b = generator_A2B(real_a)
            rec_a = generator_B2A(fake_b)
            fake_a = generator_B2A(real_b)
            rec_b = generator_A2B(fake_a)
            decision_fake_a = discriminator_A(fake_a)
            decision_fake_b = discriminator_B(fake_b)

            # Compute loss for generators and update generators
            # loss_a2b = GAN loss: mean (D_b(G(x)) − 1)^2 + Forward cycle loss: || F(G(x)) - x ||_1
            loss_a2b = compute_loss_generator(decision_fake_b, real_a, fake_b, rec_a, lambda_idt,lambda_cycle_A,loss_mode='lsgan')

            # loss_b2a = GAN loss: mean (D_a(F(x)) − 1)^2 + Backward cycle loss: || G(F(y)) - y ||_1
            loss_b2a = compute_loss_generator(decision_fake_a, real_b, fake_a, rec_b, lambda_idt,lambda_cycle_B,loss_mode='lsgan')

            # total generators loss:
            loss_generators = loss_a2b + loss_b2a

        optimizer_G.zero_grad()
        amp_scaler.scale(loss_generators).backward()
        amp_scaler.step(optimizer_G)

        decision_fake_a = rec_a = decision_fake_b = rec_b = None

        # Enable grads computation for the discriminators:
        toggle_grad(discriminator_A, True)
        toggle_grad(discriminator_B, True)

        with autocast(enabled=amp_enabled):
            decision_real_a, decision_fake_a = discriminator_forward_pass(discriminator_A, real_a, fake_a.detach(),
                                                                          fake_a_buffer)
            decision_real_b, decision_fake_b = discriminator_forward_pass(discriminator_B, real_b, fake_b.detach(),
                                                                          fake_b_buffer)
            # Compute loss for discriminators and update discriminators
            # loss_a = mean (D_a(y) − 1)^2 + mean D_a(F(x))^2
            loss_a = compute_loss_discriminator(decision_real_a, decision_fake_a)

            # loss_b = mean (D_b(y) − 1)^2 + mean D_b(G(x))^2
            loss_b = compute_loss_discriminator(decision_real_b, decision_fake_b)

            # total discriminators loss:
            loss_discriminators = 0.5 * (loss_a + loss_b)

        optimizer_D.zero_grad()
        amp_scaler.scale(loss_discriminators).backward()
        amp_scaler.step(optimizer_D)
        amp_scaler.update()

        return {
            "loss_generators": loss_generators.item(),
            "loss_generator_a2b": loss_a2b.item(),
            "loss_generator_b2a": loss_b2a.item(),
            "loss_discriminators": loss_discriminators.item(),
            "loss_discriminator_a": loss_a.item(),
            "loss_discriminator_b": loss_b.item(),
        }

    trainer = Engine(update_fn)

    metric_names = [
        'loss_discriminators',
        'loss_generators',
        'loss_discriminator_a',
        'loss_discriminator_b',
        'loss_generator_a2b',
        'loss_generator_b2a'
    ]

    def output_transform(out, name):
        return out[name]

    for name in metric_names:
        # here we cannot use lambdas as they do not store argument `name`
        RunningAverage(output_transform=partial(output_transform, name=name)).attach(trainer, name)
    # Epoch-wise progress bar with display of training losses
    ProgressBar(persist=True, bar_format="").attach(trainer, metric_names=['loss_discriminators', 'loss_generators','loss_discriminator_a','loss_discriminator_b',
        'loss_generator_a2b',
        'loss_generator_b2a'])
    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logger = TensorboardLogger(log_dir="/logs/cycle_gan_horse2zebra_tb_logs/{}".format(exp_name))

    tb_logger.attach(trainer,
                     log_handler=OutputHandler('training', metric_names),
                     event_name=Events.ITERATION_COMPLETED)


    def evaluate_fn(engine, batch):
        generator_A2B.eval()
        generator_B2A.eval()
        with torch.no_grad():
            real_a = convert_tensor(batch['A'], device=device, non_blocking=True)
            real_b = convert_tensor(batch['B'], device=device, non_blocking=True)

            fake_b = generator_A2B(real_a)
            rec_a = generator_B2A(fake_b)

            fake_a = generator_B2A(real_b)
            rec_b = generator_A2B(fake_a)

        return {
            'real_a': real_a,
            'real_b': real_b,
            'fake_a': fake_a,
            'fake_b': fake_b,
            'rec_a': rec_a,
            'rec_b': rec_b,
        }

    evaluator = Engine(evaluate_fn)

    @trainer.on(Events.EPOCH_STARTED)
    def run_evaluation(engine):
        evaluator.run(val_loader_AB)

    lr = args.lr

    milestones_values = [
        (0, lr),
        (100, lr),
        (200, 0.0)
    ]
    gen_lr_scheduler = PiecewiseLinear(optimizer_D, param_name='lr', milestones_values=milestones_values)
    desc_lr_scheduler = PiecewiseLinear(optimizer_G, param_name='lr', milestones_values=milestones_values)

    lr_scheduler = ParamGroupScheduler([gen_lr_scheduler, desc_lr_scheduler],
                                       names=['gen_lr_scheduler', 'desc_lr_scheduler'])

    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)

    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer_G, "lr"),
                     event_name=Events.EPOCH_STARTED)

    def to_np(x):
        return x.data.cpu().numpy()

    def _normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    def plot_train_result(real_image, gen_image, recon_image, epoch, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
        fig, axes = plt.subplots(4, 3, figsize=fig_size)
        imgs = [to_np(real_image[0][0,:,:]), to_np(gen_image[0][0,:,:]), to_np(recon_image[0][0,:,:]),
                to_np(real_image[0][1, :, :]), to_np(gen_image[0][1, :, :]), to_np(recon_image[0][1, :, :]),
                to_np(real_image[1][0,:,:]), to_np(gen_image[1][0,:,:]), to_np(recon_image[1][0,:,:]),
                to_np(real_image[1][1,:,:]), to_np(gen_image[1][1,:,:]), to_np(recon_image[1][1,:,:])]
        for ax, img in zip(axes.flatten(), imgs):
            ax.axis('off')
            ax.set_adjustable('box')
            # Scale to 0-255
            img = np.expand_dims(img,0)
            img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)

        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

        # save figure
        if save:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_fn = os.path.join(save_dir , 'Result_epoch_{:d}'.format(epoch + 1) + '.png')
            plt.savefig(save_fn)

        if show:
            plt.show()
        else:
            plt.close()

    test_real_A = datasetA[303]
    test_real_B = datasetB[303]
    test_fake_B = generator_A2B(test_real_A.unsqueeze(0).cuda())
    test_fake_A = generator_B2A(test_real_B.unsqueeze(0).cuda())
    test_recon_A = generator_B2A(test_fake_B)
    test_recon_B = generator_A2B(test_fake_A)
    plot_train_result([test_real_A, test_real_B], [test_fake_B.squeeze(0), test_fake_A.squeeze(0)],
                      [test_recon_A.squeeze(0), test_recon_B.squeeze(0)],
                      0, save=True, save_dir=args.output_dir)
    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = trainer.state.epoch if trainer.state is not None else 1
        test_real_A = datasetA[303]
        test_real_B = datasetB[303]
        test_fake_B = generator_A2B(test_real_A.unsqueeze(0).cuda())
        test_fake_A = generator_B2A(test_real_B.unsqueeze(0).cuda())
        test_recon_A = generator_B2A(test_fake_B)
        test_recon_B = generator_A2B(test_fake_A)
        plot_train_result([test_real_A, test_real_B], [test_fake_B.squeeze(0), test_fake_A.squeeze(0)], [test_recon_A.squeeze(0), test_recon_B.squeeze(0)],
                          epoch, save=True, save_dir=args.output_dir)

        name = 'epoch{}.pth'.format(epoch)
        to_save = {
            "generator_A2B": generator_A2B,
            "discriminator_B": discriminator_B,
            "generator_B2A": generator_B2A,
            "discriminator_A": discriminator_A,

            "optimizer_G": optimizer_G,
            "optimizer_D": optimizer_D,
            "epoch": epoch,
        }
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        save(to_save, args.output_dir, 'checkpoint_{}'.format(name))


    print("Start training")
    trainer.run(train_loader_AB,max_epochs=args.epochs)
    tb_logger.close()
if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=4,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='manual seed')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="data/dense",
                        help="location of the dataset")
    parser.add_argument("--eval-on-start", type=bool, default=False,
                        help="evaluate before training")
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='grad accumulation')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--num_lila', type=int, default=5, help='number of lila blocks in generator')
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--decay_epoch', type=int, default=100, help='start decaying learning rate after this number')
    shutil.rmtree("checkpoints")
    os.mkdir("checkpoints")
    run(parser.parse_args())
