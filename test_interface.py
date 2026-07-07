import os
import torch
import torchaudio
from tqdm import tqdm
from proposed_DS_U2Net_HA import DS_U2Net_HA


noisy_folder = "ATH_TO_NOISY_DATASET" # path of noisy test dataset
save_folder = "PATH_TO_ENHANCED_RESULTS"   # path of enhanced dataset to save
checkpoint_path = "PATH_OF_BEST_CHECKPOINT" # path of best checkpoint

os.makedirs(save_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DS_U2Net_HA().to(device)

chkpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(chkpt["model"])
model.eval()

for file_name in tqdm(os.listdir(noisy_folder)):
    if not file_name.lower().endswith(".wav"):
        continue

    file_path = os.path.join(noisy_folder, file_name)

    noisy_waveform, sample_rate = torchaudio.load(file_path)

    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

    original_length = noisy_waveform.shape[-1]

    noisy_waveform = noisy_waveform.to(device)

    with torch.no_grad():
        noisy_waveform = noisy_waveform.unsqueeze(0)  # [1, 1, T]

        enhanced_waveform = model(noisy_waveform)

        if isinstance(enhanced_waveform, tuple):
            enhanced_waveform = enhanced_waveform[0]

    enhanced_waveform = enhanced_waveform.squeeze(0)

    if enhanced_waveform.dim() == 1:
        enhanced_waveform = enhanced_waveform.unsqueeze(0)

    if enhanced_waveform.shape[-1] > original_length:
        enhanced_waveform = enhanced_waveform[:, :original_length]
    elif enhanced_waveform.shape[-1] < original_length:
        pad_len = original_length - enhanced_waveform.shape[-1]
        enhanced_waveform = torch.nn.functional.pad(enhanced_waveform, (0, pad_len))

    enhanced_waveform = enhanced_waveform.detach().cpu()

    save_path = os.path.join(save_folder, f"enhanced_{file_name}")
    torchaudio.save(save_path, enhanced_waveform, sample_rate)

    print(f"✅ saved : {save_path}")