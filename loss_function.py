import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[256, 512, 1024]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes

    def forward(self, enhanced, target):
        """
        enhanced, target: [B, 1, T] or [B, T]
        """
        # Remove channel dim if exists
        if enhanced.ndim == 3:
            enhanced = enhanced.squeeze(1)  # → [B, T]
        if target.ndim == 3:
            target = target.squeeze(1)  # → [B, T]

        total_loss = 0

        for fft in self.fft_sizes:
            # Compute STFT
            enh_stft = torch.stft(
                enhanced,
                n_fft=fft,
                hop_length=fft // 4,
                win_length=fft,
                return_complex=True
            )

            tgt_stft = torch.stft(
                target,
                n_fft=fft,
                hop_length=fft // 4,
                win_length=fft,
                return_complex=True
            )

            # L1 magnitude loss
            total_loss += F.l1_loss(torch.abs(enh_stft), torch.abs(tgt_stft))

        return total_loss / len(self.fft_sizes)

