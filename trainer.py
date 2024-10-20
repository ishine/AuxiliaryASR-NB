# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

from utils import calc_wer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils import *
from transformers import AutoModel
from torch.nn.utils import weight_norm, spectral_norm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
import torchaudio
import whisper
import wandb


class Trainer(object):
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cuda"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = True
        #self.whisper_model = AutoModel.from_pretrained("NbAiLabBeta/nb-whisper-small-verbatim")
        #self.whisper_model.to(self.device)
        #self.whisper_model.eval()


    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr


    def adaptive_gradient_clipping(self, clip_factor=0.01, eps=1e-6):
        
        """
        Adaptive Gradient Clipping (AGC) for a model's parameters.
        
        Args:
            model (torch.nn.Module): The model whose gradients are being clipped.
            clip_factor (float): The scaling factor λ. Typically a small value (e.g., 0.01).
            eps (float): A small epsilon value to prevent division by zero.
        """
        for param in self.model.parameters():
            # Skip if there is no gradient
            if param.grad is None:
                continue
            
            # Compute the L2 norm of weights and gradients
            weight_norm = torch.norm(param, p=2).clamp(min=eps)  # Ensure norm is at least eps to prevent divide by zero
            grad_norm = torch.norm(param.grad, p=2)
    
            # Calculate adaptive clipping threshold
            clip_value = clip_factor * weight_norm
    
            # Clip the gradient if its norm exceeds the threshold
            if grad_norm > clip_value:
                param.grad.mul_(clip_value / grad_norm)

    @staticmethod
    def get_image(arrs):
        pil_images = []
        height = 0
        width = 0
        for arr in arrs:
            uint_arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
            pil_image = Image.fromarray(uint_arr)
            pil_images.append(pil_image)
            height += uint_arr.shape[0]
            width = max(width, uint_arr.shape[1])

        palette = Image.new('L', (width, height))
        curr_heigth = 0
        for pil_image in pil_images:
            palette.paste(pil_image, (0, curr_heigth))
            curr_heigth += pil_image.size[1]

        return palette

    def run(self, batch):
        self.optimizer.zero_grad()
        batch = [b.to(self.device) for b in batch]
        #batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        
        #text_input, text_input_length, mel_input, mel_input_length, paths, wave_input = batch
        text_input, text_input_length, mel_input, mel_input_length = batch

        mel_input_length = mel_input_length // (2 ** self.model.n_down)
        future_mask = self.model.get_future_mask(
            mel_input.size(2)//(2**self.model.n_down), unmask_future_steps=0).to(self.device)
        mel_mask = self.model.length_to_mask(mel_input_length)
        text_mask = self.model.length_to_mask(text_input_length)
        ppgs, s2s_pred, s2s_attn = self.model(
            mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
        
        loss_ctc = self.criterion['ctc'](ppgs.log_softmax(dim=2).transpose(0, 1),
                                      text_input, mel_input_length, text_input_length)

        loss_s2s = 0
        for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
            loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_s2s /= text_input.size(0)

        #feature_matching_loss = self._calculate_feature_matching_loss(wave_input, mel_input, "MSE")

        #lambda_feature = 0.2  # Weight for feature matching loss

        #loss = loss_ctc + loss_s2s + feature_matching_loss * lambda_feature
        loss = loss_ctc + loss_s2s

        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                
                print(f"No gradients for parameter {name}")
        self.adaptive_gradient_clipping(clip_factor=0.01)
        #torch.nn.utils.clip_grad_value_(self.model.parameters(), 50)
        self.optimizer.step()
        self.scheduler.step()
        return {'loss': loss.item(),
                'ctc': loss_ctc.item(),
                's2s': loss_s2s.item()
}


    def _calculate_feature_matching_loss(self, wave_input, mel_input, loss_type="MSE"):
        # Whisper feature extraction
        with torch.no_grad():
    
            resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).to(self.device)
            wave_input_squeezed = [wave.squeeze(1) if wave.ndim == 3 else wave for wave in wave_input]
            max_audio_length = max([wave.shape[-1] for wave in wave_input_squeezed])
            wave_input_padded = [whisper.pad_or_trim(wave, max_audio_length) for wave in wave_input_squeezed]
            whisper_inputs_resampled = torch.stack([resampler(a.to(self.device)) for a in wave_input_padded])
            whisper_mel = torch.stack([whisper.log_mel_spectrogram(a) for a in whisper_inputs_resampled]).to(self.device)
    
            # Pad or trim whisper_mel to match the target frames
            target_frames = 3000
            current_frames = whisper_mel.shape[-1]
            if current_frames > target_frames:
                whisper_mel = whisper_mel[:, :, :target_frames]
            elif current_frames < target_frames:
                pad_size = target_frames - current_frames
                whisper_mel = F.pad(whisper_mel, (0, pad_size), mode='constant', value=0)
    
            # Extract whisper embeddings
            whisper_embeddings = self.whisper_model.encoder(whisper_mel).last_hidden_state

        # Project whisper embeddings to match encoder features
        encoder_features = self.model.get_feature(mel_input)
        encoder_seq_len = encoder_features.size(1)
        encoder_feature_dim = encoder_features.size(-1)
        downsample = nn.AdaptiveAvgPool1d(encoder_seq_len).to(self.device)
        whisper_embeddings_projected_downsampled = downsample(whisper_embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        project_feature_dim = nn.Linear(whisper_embeddings_projected_downsampled.size(-1), encoder_feature_dim).to(self.device)
        whisper_embeddings_projected_downsampled = project_feature_dim(whisper_embeddings_projected_downsampled).detach()

        # Calculate feature matching loss
        if loss_type == "MSE":
            feature_matching_loss = F.mse_loss(encoder_features, whisper_embeddings_projected_downsampled)
        elif loss_type == "KL":
            feature_matching_loss = F.kl_div(F.log_softmax(encoder_features, dim=-1), F.softmax(whisper_embeddings_projected_downsampled, dim=-1), reduction='batchmean')
        else:
            raise ValueError("Invalid loss type specified. Choose between 'MSE' and 'KL'.")

        return feature_matching_loss

    
    def _train_epoch(self):
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        true_labels = []
        predicted_labels = []
        results_table = wandb.Table(columns=["Sample_ID", "Ground_Truth", "Prediction", "WER"])
    
        total_sentence_errors = 0
        total_sentences = 0
    
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device) for b in batch]
    
            text_input, text_input_length, mel_input, mel_input_length = batch
            mel_input_length = mel_input_length // (2 ** self.model.n_down)
            future_mask = self.model.get_future_mask(
                mel_input.size(2) // (2 ** self.model.n_down), unmask_future_steps=0
            ).to(self.device)
            mel_mask = self.model.length_to_mask(mel_input_length)
            text_mask = self.model.length_to_mask(text_input_length)
            ppgs, s2s_pred, s2s_attn = self.model(
                mel_input, src_key_padding_mask=mel_mask, text_input=text_input
            )
            loss_ctc = self.criterion['ctc'](
                ppgs.log_softmax(dim=2).transpose(0, 1),
                text_input, mel_input_length, text_input_length
            )
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, text_input, text_input_length):
                loss_s2s += self.criterion['ce'](_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= text_input.size(0)
    
            loss = loss_ctc + loss_s2s
    
            eval_losses["eval/ctc"].append(loss_ctc.item())
            eval_losses["eval/s2s"].append(loss_s2s.item())
            eval_losses["eval/loss"].append(loss.item())
    
            # Compute WER
            _, amax_ppgs = torch.max(ppgs, dim=2)
            wers = [
                calc_wer(
                    target[:text_length],
                    pred[:mel_length],
                    ignore_indexes=[0, 1, 2, 3, 4]
                )
                for target, pred, text_length, mel_length in zip(
                    text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu()
                )
            ]
            eval_losses["eval/wer"].extend(wers)
    
            # Token-level accuracy
            _, amax_s2s = torch.max(s2s_pred, dim=2)
            acc = [
                torch.eq(target[:length], pred[:length]).float().mean().item()
                for target, pred, length in zip(
                    text_input.cpu(), amax_s2s.cpu(), text_input_length.cpu()
                )
            ]
            eval_losses["eval/acc"].extend(acc)
    
            # Compute average confidence
            for idx, (pred, length) in enumerate(zip(s2s_pred, text_input_length)):
                # Get softmax probabilities
                probs = F.softmax(pred[:length], dim=-1)  # Shape: (sequence_length, num_classes)
                max_probs, _ = torch.max(probs, dim=-1)  # Shape: (sequence_length,)
    
                # Calculate average confidence
                avg_confidence = max_probs.mean().item()
                eval_losses["eval/avg_confidence"].append(avg_confidence)
    
            # Compute Sentence Error Rate (SER) and prepare data for results table
            for idx, (target, pred, text_length, mel_length) in enumerate(
                zip(text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())
            ):
                target_filtered = [
                    int(t.item()) for t in target[:text_length]
                    if int(t.item()) not in [0, 1, 2, 3, 4]
                ]
                pred_filtered = [
                    int(p.item()) for p in pred[:mel_length]
                    if int(p.item()) not in [0, 1, 2, 3, 4]
                ]
    
                # Check for exact match
                if target_filtered != pred_filtered:
                    total_sentence_errors += 1
                total_sentences += 1
    
                # Prepare strings for the results table
                target_str = " ".join(map(str, target_filtered))
                pred_str = " ".join(map(str, pred_filtered))
    
                # Calculate WER for the table (if needed)
                wers = calc_wer(
                    target[:text_length], pred[:mel_length], ignore_indexes=[0, 1, 2, 3, 4]
                )
    
                # Add data to the WandB table
                results_table.add_data(idx, target_str, pred_str, wers)
    
            if eval_steps_per_epoch <= 2:
                eval_images["eval/image"].append(
                    self.get_image([s2s_attn[0].cpu().numpy()])
                )
    
        # After processing all batches, compute SER
        ser_value = total_sentence_errors / total_sentences if total_sentences > 0 else 0
        eval_losses["eval/ser"] = ser_value
    
        # Average the collected losses
        eval_losses = {
            key: np.mean(value) if isinstance(value, list) else value
            for key, value in eval_losses.items()
        }
        eval_losses.update(eval_images)
        eval_losses["results_table"] = results_table
        return eval_losses

