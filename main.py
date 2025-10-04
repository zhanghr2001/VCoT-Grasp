import os
import argparse

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import get_cosine_schedule_with_warmup, get_constant_schedule, set_seed
from accelerate import Accelerator

from model import ArchConfig, VCoTGraspConfig, VCoTGraspForConditionalGeneration, VCoTGraspProcessor
from data import get_dataloaders


def main(args):
    set_seed(args.seed)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.tensorboard_root)
    hyper_params = vars(args)
    accelerate_hyper_params = {
        "num_processes": accelerator.num_processes,
        "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
    }
    hyper_params.update(**accelerate_hyper_params)

    arch_config = ArchConfig(args.use_bbox, args.action_head)
    if not args.load_checkpoint_dir:
        # init a model
        config = VCoTGraspConfig.from_json_file("model/config.json")
        config.arch_config = arch_config
        model = VCoTGraspForConditionalGeneration(config)
        processor = VCoTGraspProcessor(arch_config)
    else:
        model = VCoTGraspForConditionalGeneration.from_pretrained(args.load_checkpoint_dir).to(torch.bfloat16)
        processor = VCoTGraspProcessor(arch_config)

    model.set_trainable(
        image_encoder=args.train_image_encoder,
        image_projector=args.train_image_projector,
        embeddings=args.train_embeddings,
        lm=args.train_lm,
        action_head=True,
    )
    train_dataloader, eval_dataloader, _ = get_dataloaders(
        arch_config,
        args.train_dataset,
        processor,
        args.train_batch_size_per_gpu,
        args.eval_batch_size_per_gpu,
        bbox_ratio=args.bbox_ratio,
        data_ratio=args.data_ratio,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # on multiple gpus, lr scheduler will step multiple times after one gradient update, so multiply steps by num_processes, see https://github.com/huggingface/accelerate/issues/2142
    num_training_steps = (len(train_dataloader) * args.train_epoch) // accelerator.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.lr_scheduler == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif args.lr_scheduler == "cosine_schedule":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=5e-6)
    elif args.lr_scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    else:
        raise NotImplementedError("Invalid scheduler")
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    accelerator.init_trackers(args.run_name, config=hyper_params)
    save_dir = os.path.join(args.save_root, args.run_name)
    start_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, args.train_epoch):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                loss_info = outputs.loss_info
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                loss = accelerator.reduce(loss, "mean")  # per sample loss
                loss_info = "".join([f" {k}: {v:.5f} " for k, v in loss_info.items()])
                accelerator.log({"epoch": epoch, "loss": loss.item()}, step=global_step)
                accelerator.print(f"epoch: {epoch}  global_step: {global_step}  loss: {loss.item():.5f} " + loss_info)

                # eval, must check in "if accelerator.sync_gradients"
                if global_step % args.eval_every_n_steps == 0:
                    model.eval()
                    losses = []
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**batch, use_cache=False)
                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(args.eval_per_device_batch_size)))

                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    accelerator.log({"epoch": epoch, "eval_loss": eval_loss.item()}, step=global_step)
                    accelerator.print(f"epoch: {epoch}, global_step: {global_step}, eval_loss: {eval_loss.item()}")
                    model.train()

                if global_step % args.save_every_n_steps == 0:
                    # in deepspeed, all processes must call save_state method and not just the process with rank 0, see https://github.com/huggingface/diffusers/issues/2606
                    # accelerator.save_state() is only done on main process, so do not use "if accelerator.is_main_process"
                    version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}")
                    if args.save_state:
                        state_version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}_state")
                        accelerator.save_state(state_version_dir)  # save model, optimizer, scheduler, scaler, RNG
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(version_dir)

    version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}")
    if args.save_state:
        state_version_dir = os.path.join(save_dir, f"epoch{epoch}_step{global_step}_state")
        accelerator.save_state(state_version_dir)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(version_dir)

    accelerator.end_training()
    accelerator.print("Training end")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--load-checkpoint-dir", type=str, default="", help="finished checkpoint dir")
    parser.add_argument("--save-root", type=str, default="./checkpoint", help="save directory")
    parser.add_argument("--tensorboard-root", type=str, default="./checkpoint/tensorboard", help="tensorboard directory")
    parser.add_argument("--run-name", type=str, default="run", help="name of this training phase")
    parser.add_argument("--save-state", action="store_true", help="save grad, optimizer states, etc")

    # architecture choice
    parser.add_argument("--use-bbox", action="store_true")
    parser.add_argument("--action-head", type=str, help="MLP, Diffusion, LM_pretrained, LM_new")

    # to freeze or train model parts, default freeze
    parser.add_argument("--train-image-encoder", action="store_true")
    parser.add_argument("--train-image-projector", action="store_true")
    parser.add_argument("--train-embeddings", action="store_true", help="freeze or train lm's input and output embeddings")
    parser.add_argument("--train-lm", action="store_true", help="freeze or train lm decoders")

    # training parameters
    parser.add_argument("--train-dataset", type=str, help="train dataset")
    parser.add_argument("--train-epoch", type=int, help="train epoch, not incremental")
    parser.add_argument("--train-batch-size-per-gpu", type=int, help="train batch size")
    parser.add_argument("--eval-batch-size-per-gpu", type=int, help="eval batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--lr-scheduler", type=str, default="cosine_schedule_with_warmup")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="ratio of warmup steps and training steps")
    parser.add_argument("--weight-decay", type=float, default=0, help="weight decay")
    parser.add_argument("--eval-every-n-steps", type=int, help="evaluate every n steps")
    parser.add_argument("--save-every-n-steps", type=int, help="save every n steps")
    parser.add_argument("--data-ratio", type=float, default=1.0, help="data used for training")
    parser.add_argument("--bbox-ratio", type=float, default=1.0, help="bbox used for training")

    parser.add_argument("--seed", type=int, default=42, help="only positive value enables a fixed seed")

    args = parser.parse_args()

    main(args)
