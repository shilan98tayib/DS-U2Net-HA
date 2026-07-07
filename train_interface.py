import os
import time
import torch
from Baseline import Baseline_U2Net
from Baseline_TLS import Baseline_TLS
from Baseline_TSAIC import Baseline_TSAIC
from Baseline_DualStages import Baseline_DualStages
from masking_stage import MaskingStage
from mapping_stage import MappingStage
from proposed_DS_U2Net_HA import DS_U2Net_HA
from trainer import train, valid, joint_train, joint_valid
from dataloader import create_dataloader
import tools
import config as cfg
import warnings
import random
import numpy as np


warnings.filterwarnings("ignore")
SEED = cfg.SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed: {SEED}")


#######################################################################
#                     Set a job and a log folder                      #
#######################################################################
dir2sav = cfg.job_dir
dir2log = cfg.logs_dir

# make the folder
if not os.path.exists(dir2sav):
    os.makedirs(dir2sav)
if not os.path.exists(dir2log):
    os.makedirs(dir2log)

#######################################################################
#                            Model init                               #
#######################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# define model
if cfg.model_mode == 'Baseline':
    model = Baseline_U2Net().to(DEVICE)
elif cfg.model_mode == 'Baseline+TLS':
    model = Baseline_TLS().to(DEVICE)
elif cfg.model_mode == 'Baseline+TSAIC':
    model = Baseline_TSAIC().to(DEVICE)
elif cfg.model_mode == 'Baseline+Dual-Stages':
    model = Baseline_DualStages().to(DEVICE)
elif cfg.model_mode == 'Masking-Stage':
    model = MaskingStage().to(DEVICE)
elif cfg.model_mode == 'Mapping-Stage':
    model = MappingStage().to(DEVICE)
elif cfg.model_mode == 'DS-U2Net-HA':
    model = DS_U2Net_HA().to(DEVICE)

if not os.path.exists(cfg.pretrained_addr):
    model.apply(tools.adaptive_weight_init)

# define train mode
if cfg.joint_loss:
    trainer = joint_train
    validator = joint_valid
else:
    trainer = train
    validator = valid


optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate , betas=(0.9, 0.999), weight_decay=1e-5) #improvment
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
total_params = tools.cal_total_params(model)

# load the params if there is pretrained model
epoch_start_idx = 1
if os.path.exists(cfg.pretrained_addr):
    print('Load the pretrained model...')
    chkpt = torch.load(cfg.pretrained_addr + '/chkpt_{}.pt'.format(cfg.chkpt_num))
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])
    epoch_start_idx = chkpt['epoch'] + 1

    dir2sav = cfg.pretrained_addr

#######################################################################
#                          Create Dataloader                          #
#######################################################################
train_loader = create_dataloader(mode='train')
valid_loader = create_dataloader(mode='valid')


#######################################################################
#                       Confirm model intormation                     #
#######################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# save the status information
tools.write_status(dir2sav)


#######################################################################
#######################################################################
#                               Main                                  #
#######################################################################
#######################################################################
writer = tools.Writer(dir2log)
train_log_fp = open(dir2sav + '/train_log.txt', 'a')


print('Main program start...')
for epoch in range(epoch_start_idx, cfg.max_epoch + 1):
    st_time = time.time()

    # train
    train_loss = trainer(model, train_loader, optimizer, writer, epoch, DEVICE)

    # save checkpoint file to resume training
    save_path = str(dir2sav + '/chkpt_%d.pt' % epoch)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'seed': SEED
    }, save_path)

    # validate
    valid_loss, pesq, stoi, csig, cbak, covl = validator(model, valid_loader, writer, epoch, DEVICE)
    # ACTIVATE LR
    scheduler.step(valid_loss)
    print('EPOCH[{}] T {:.6f} |  V {:.6f} takes {:.3f} seconds'
          .format(epoch, train_loss, valid_loss, time.time() - st_time))
    print(
        'PESQ {:.6f} | STOI {:.6f} | CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}'
        .format(pesq, stoi, csig, cbak, covl)
    )

    # write train log
    train_log_fp.write('EPOCH[{}] T {:.6f} |  V {:.6f} takes {:.3f} seconds\n'
                       .format(epoch, train_loss, valid_loss, time.time() - st_time))
    train_log_fp.write(
        'PESQ {:.6f} | STOI {:.6f} | CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}\n'
        .format(pesq, stoi, csig, cbak, covl)
    )

print('Training has been finished.')
train_log_fp.close()


