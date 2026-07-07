import torch
import tools
from loss_function import MultiResolutionSTFTLoss


#######################################################################
#                            For train                                #
#######################################################################
def train(model, train_loader, optimizer, writer, EPOCH, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        loss = model.loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss(train_loss, EPOCH)

    return train_loss


def joint_train(model, train_loader, optimizer, writer, EPOCH, DEVICE):
    # initialization
    train_loss = 0
    train_main_loss = 0
    train_sub_loss1 = 0
    batch_num = 0

    # loss function
    stft_loss_fn = MultiResolutionSTFTLoss()

    # train
    model.train()
    for inputs, targets in tools.Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)

        clean_mag, _ = model.stft(targets)
        outputs_mags, _ = model.stft(outputs)

        main_loss = model.loss(outputs, targets)


        sub_loss1 = stft_loss_fn(outputs, targets)
        r1 = 0.7
        r2 = 0.3

        loss = r1 * main_loss + r2 * sub_loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_main_loss += main_loss.item()
        train_sub_loss1 += sub_loss1.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss(train_loss, EPOCH)
    writer.log_train_joint_loss(train_main_loss/batch_num, train_sub_loss1/batch_num,EPOCH)

    return train_loss


#######################################################################
#                          For validation                             #
#######################################################################
def valid(model, valid_loader, writer, EPOCH, DEVICE):
    # initialization
    valid_loss = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0
    avg_csig = 0
    avg_cbak = 0
    avg_covl = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            loss = model.loss(outputs, targets)

            valid_loss += loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = tools.cal_pesq(enhanced_wavs, clean_wavs)
            stoi = tools.cal_stoi(enhanced_wavs, clean_wavs)
            csig, cbak, covl = tools.cal_composite(enhanced_wavs,clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi
            avg_csig += csig
            avg_cbak += cbak
            avg_covl += covl

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num
        avg_csig /= batch_num
        avg_cbak /= batch_num
        avg_covl /= batch_num

    # tensorboard
    writer.log_valid_loss(valid_loss, EPOCH)

    writer.log_scores(avg_pesq,avg_stoi,avg_csig,avg_cbak,avg_covl,EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_covl


def joint_valid(model, valid_loader, writer, EPOCH, DEVICE):
    # initialization
    valid_loss = 0
    valid_main_loss = 0
    valid_sub_loss1 = 0
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0
    avg_csig = 0
    avg_cbak = 0
    avg_covl = 0

    # loss function
    stft_loss_fn = MultiResolutionSTFTLoss()

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in tools.Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)

            clean_mag, _ = model.stft(targets)
            outputs_mags, _ = model.stft(outputs)

            main_loss = model.loss(outputs, targets)

            sub_loss1 = stft_loss_fn(outputs, targets)

            r1 = 0.7
            r2 = 0.3
            loss = r1 * main_loss + r2 * sub_loss1

            valid_loss += loss.item()
            valid_main_loss += main_loss.item()
            valid_sub_loss1 += sub_loss1.item()

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = tools.cal_pesq(enhanced_wavs, clean_wavs)
            stoi = tools.cal_stoi(enhanced_wavs, clean_wavs)
            csig, cbak, covl = tools.cal_composite(enhanced_wavs,clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi
            avg_csig += csig
            avg_cbak += cbak
            avg_covl += covl

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num
        avg_csig /= batch_num
        avg_cbak /= batch_num
        avg_covl /= batch_num

        # tensorboard
        writer.log_valid_loss(valid_loss, EPOCH)
        writer.log_valid_joint_loss(valid_main_loss/batch_num, valid_sub_loss1/batch_num,EPOCH)
        writer.log_scores(avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_covl, EPOCH)
        writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

        return valid_loss, avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_covl