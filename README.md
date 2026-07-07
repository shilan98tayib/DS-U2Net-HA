# DS-U2Net-HA
DS-U²Net-HA: Dual-Stage Nested U-Net with Hybrid Attention Mechanism Model for Monaural Speech Enhancement. The proposed model enhances noisy speech by integrating a dual-stage of U-Net architecture with a hybrid attention mechanism to improve speech quality and intelligibility while effectively suppressing background noise.
________________________________________
# Repository Structure
- proposed_DS_U2Net_HA.py      # Proposed DS-U²Net-HA model
- masking_stage.py             # First-stage masking network only
- mapping_stage.py             # Second-stage mapping network only
- TSAIC.py                     # Hybrid attention module
- Baseline.py                  # Baseline U²Net
- Baseline_TLS.py              # Baseline + TLS
- Baseline_TSAIC.py            # Baseline + TSAIC
- Baseline_DualStages.py       # Baseline + Dual Stages
- train_interface.py           # Training script
- trainer.py                   # Training/validation functions
- evaluate.py                  # Objective evaluation
- test_interface.py            # Testing SE model script
- dataloader.py                # Dataset loader
- tools.py                     # STFT, iSTFT and utilities
- loss_function.py             # MR-STFT loss
- config.py                    # Configuration file

- files_of_tables_and_figures/
________________________________________
# Requirements
- The project has been tested using:
  - Python == 3.8
  - torch == 2.4.1
  - torchaudio == 2.4.1
  - numpy == 1.24.4
  - scipy == 1.10.1
  - soundfile == 0.13.1
  - librosa == 0.11.0
  - matplotlib == 3.7.5
  - tqdm == 4.67.1
  - pesq == 0.0.5
  - pystoi == 0.4.1
  - pysepm == 0.1
  - tensorboardX == 2.6.2.2
# Install dependencies
- pip install -r requirements.txt
________________________________________
# Dataset Preparation
- Modify the dataset paths inside config.py :
  - noisy_dirs_for_train = "../Dataset/train/NOISY/"
  - clean_dirs_for_train = "../Dataset/train/ CLEAN/"
  - noisy_dirs_for_valid = "../Dataset/valid/noisy/"
  - clean_dirs_for_valid = "../Dataset/valid/ NOISY/"
- FS = 16000
- WIN_LEN = 400
- HOP_LEN = 100
- FFT_LEN = 512
- Batch_size = 2
- learning_rate = 1e-4
- max_epoch = 50

# Example
- Dataset/
  - train/
   - NOISY/
   - CLEAN/
  -  valid/
   -  NOISY/
   - CLEAN/
________________________________________
# Training
- Select the desired model inside config.py : model_mode = model_type[6] 
- Available models:
  - Baseline.
  - Baseline+TLS.
  - Baseline+TSAIC.
  - Baseline+Dual-Stages.
  - Masking-Stage.
  - Mapping-Stage.
  - DS-U2Net-HA.
- Run training: python train_interface.py
________________________________________
# For testing the model:
- Specify:
  - Noisy speech folder path.
  - Best checkpoint path.
  - Output path. 
- inside: test_interface.py
- Then run: python test_interface.py
- Enhanced speech files will be saved automatically.
________________________________________
# Evaluation
- Objective evaluation includes:
 - PESQ 
 - STOI 
 - CSIG 
 - CBAK 
 - COVL 
- Run: python evaluate.py
- The evaluation results are exported to an Excel file.
________________________________________
# Notes
- The VoiceBank+DEMAND dataset is not redistributed in this repository.
- Only dataset preparation instructions are provided.
- The VoiceBank+Demand dataset is available: https://github.com/jim-schwoebel/voice_datasets .
________________________________________
# Citation
- If you use this repository in your research, please cite to the paper.
- The name od paper is :
# DS-U²Net-HA: Dual-Stage Nested U-Net with Hybrid Attention Mechanism Model for Monaural Speech Enhancement
________________________________________


