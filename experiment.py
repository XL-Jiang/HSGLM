import os
import random
import warnings

import numpy as np
import torch
import util
from HSGLM import HSGLM as model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore', category=UserWarning)

def step(model, criterion, t1,t2,t3,label,phd_ftrs,clip_grad=0.0, device='cpu',optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    # run model
    logit,latent,attention = model( t1.to(device),t2.to(device),t3.to(device),phd_ftrs.to(device))
    loss = criterion(logit, label.to(device))

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return logit,latent, loss, attention

def train(opt):
    # set seed and device
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    else:
        device = torch.device("cpu")

    print(' Loading Dataset')
    # define dataset
    if opt.dataset == 'ABIDE1':
        dataset = DatasetABIDEI(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro, atlas=opt.atlas,site=opt.site)
    elif opt.dataset == 'ABIDE2':
        dataset = DatasetABIDEII(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro, atlas=opt.atlas,site=opt.site)
    elif  opt.dataset=='ADHD':
        dataset = DatasetADHD(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro, atlas=opt.atlas,site=opt.site)
    # elif  opt.dataset=='MDD':
    #     dataset = DatasetMDD(sourcedir='F:/', threshod=50, dynamic_length=opt.dynamic_length, atlas='AAL-116',site='NYU')
    else:
        raise
    timeseries1, timeseries2,timeseries3,label,phonetic_data = dataset.load_data()
    num_ROIs1 = timeseries1.shape[2]
    num_ROIs2 = timeseries2.shape[2]
    num_ROIs3 = timeseries3.shape[2]
    phd_dim = phonetic_data.shape[1]
    cv_splits = dataset.data_split(opt.folds)
    logger = util.logger.Logger(opt.folds, 2)
    # resume checkpoint if file exists
    if os.path.isfile(os.path.join('{}_result'.format(opt.dataset), 'checkpoint.pth')):
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    train_losses = []
    val_losses = []
    # start training
    print("\r\n=====Start Train =====")
    for fold in range(opt.folds):
        print("\r\n========================== Fold {} ==========================".format(fold + 1))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        random.shuffle(train_ind)

        train_dataset = TensorDataset(torch.from_numpy(timeseries1[train_ind]), torch.from_numpy(timeseries2[train_ind]),torch.from_numpy(timeseries3[train_ind]),torch.from_numpy(label[train_ind]),torch.from_numpy(phonetic_data[train_ind]))
        train_dataloader = DataLoader(train_dataset, batch_size=opt.minibatch_size, shuffle=False,num_workers=opt.num_workers, pin_memory=True)

        test_dataset = TensorDataset(torch.from_numpy(timeseries1[test_ind]),torch.from_numpy(timeseries2[test_ind]),torch.from_numpy(timeseries3[test_ind]), torch.from_numpy(label[test_ind]),torch.from_numpy(phonetic_data[test_ind]))
        test_dataloader = DataLoader(test_dataset, batch_size=opt.minibatch_size, shuffle=False,num_workers=opt.num_workers, pin_memory=True)
        if not os.path.exists(os.path.join('{}_result'.format(opt.dataset), 'model', str(fold))):
            os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'model', str(fold)))
        summary_writer = SummaryWriter(os.path.join('{}_result'.format(opt.dataset), 'summary', str(fold), 'train'), )
        summary_writer_val = SummaryWriter(os.path.join('{}_result'.format(opt.dataset), 'summary', str(fold), 'val'), )
        if not os.path.exists(os.path.join('{}_result'.format(opt.dataset), 'attention1')):
            os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention1'))
        if not os.path.exists(os.path.join('{}_result'.format(opt.dataset), 'attention2')):
            os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention2'))
        if not os.path.exists(os.path.join('{}_result'.format(opt.dataset), 'attention3')):
            os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention3'))

        HSGLM = model(
            input_dim1=num_ROIs1,
            input_dim2=num_ROIs2,
            input_dim3=num_ROIs3,
            # number of brain regions at AAL spatial scale# number of brain regions at HO spatial scale
            hidden_dim=opt.hidden_dim,
            topk=opt.topk,
            phd_dim =phd_dim,
            num_classes=2,
            num_heads=opt.num_heads,
            num_time=(opt.dynamic_length-opt.window_size)//opt.window_stride+1,
            window_size = opt.window_size,
            kernel_size = opt.kernel_size,
            cls_token=opt.cls_token,
            readout=opt.readout)
        HSGLM  = HSGLM.to(device)
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        optimizer = torch.optim.Adam(LG_MDGL.parameters(), lr=opt.lr,weight_decay = opt.wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.max_lr, epochs=opt.num_epochs,steps_per_epoch=len(train_dataloader), pct_start=0.2, div_factor=opt.max_lr/opt.lr, final_div_factor=1000)

        # Fusion
        print('training')
        for epoch in range(opt.num_epochs):
            logger.initialize(fold)
            train_loss_accumulate = 0.0
            valid_loss_accumulate = 0.0
            for i, (batch_timeseries1, batch_timeseries2,batch_timeseries3,batch_label,batch_phd_data) in enumerate(train_dataloader):
                # move data to cuda
                batch_timeseries1 = batch_timeseries1.to(device)
                batch_timeseries2 = batch_timeseries2.to(device)
                batch_timeseries3 = batch_timeseries3.to(device)
                batch_label = batch_label.to(device)
                batch_phd_data = batch_phd_data.to(device)
                # split timeseries
                dyn_t1, sampling_points1 = util.bold.process_dynamic_timeseries(batch_timeseries1, opt.window_size, opt.window_stride,opt.dynamic_length)
                sampling_endpoints1 = [p + opt.window_size for p in sampling_points1]

                dyn_t2, sampling_points2 = util.bold.process_dynamic_timeseries(batch_timeseries2, opt.window_size,opt.window_stride, opt.dynamic_length)
                sampling_endpoints2 = [q + opt.window_size for q in sampling_points2]

                dyn_t3, sampling_points3 = util.bold.process_dynamic_timeseries(batch_timeseries3, opt.window_size,opt.window_stride, opt.dynamic_length)
                sampling_endpoints3 = [r + opt.window_size for r in sampling_points3]

                dyn_t1 = torch.tensor(dyn_t1, dtype=torch.float32)
                dyn_t2 = torch.tensor(dyn_t2, dtype=torch.float32)
                dyn_t3 = torch.tensor(dyn_t3, dtype=torch.float32)
                train_label = torch.tensor(batch_label, dtype=torch.long)
                phd_ftrs = torch.tensor(batch_phd_data, dtype=torch.float32)

                logit, latent,loss, attention = step(
                    model= HSGLM,
                    criterion=criterion,
                    t1=dyn_t1,
                    t2=dyn_t2,
                    t3=dyn_t3,
                    label=train_label,
                    phd_ftrs = phd_ftrs,
                    clip_grad=opt.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                train_loss_accumulate += loss.detach().cpu().numpy()
                logger.add(k=fold, pred=pred.detach().cpu().numpy(), true=train_label.detach().cpu().numpy(),prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i + epoch * len(train_dataloader))
            #print Evalution
            metrics = logger.evaluate(fold)
            print('epoch--{}'.format(epoch + 1), metrics)
            #summarize results
            samples = logger.get(fold)
            metrics = logger.evaluate(fold)
            summary_writer.add_scalar('loss', train_loss_accumulate / len(train_dataloader), epoch)
            if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1],epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key == 'fold']
            torch.save(attention['node-attention1'],os.path.join('{}_result'.format(opt.dataset), 'attention1', 'f_' + str(fold) + '_e_' + str(epoch) + '.pt'))
            torch.save(attention['node-attention2'],os.path.join('{}_result'.format(opt.dataset), 'attention2', 'f_' + str(fold) + '_e_' + str(epoch) + '.pt'))
            torch.save(attention['node-attention3'],os.path.join('{}_result'.format(opt.dataset), 'attention3', 'f_' + str(fold) + '_e_' + str(epoch) + '.pt'))
            # save checkpoint
            torch.save({
                'fold': fold,
                'epoch': epoch + 1,
                'model': HSGLM.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join('{}_result'.format(opt.dataset), 'checkpoint.pth'))
            train_losses.append(train_loss_accumulate / len(train_dataloader))

            if opt.validate:
                print('validating. not for testing purposes')
                logger.initialize(fold)
                for i, (batch_timeseries1, batch_timeseries2,batch_timeseries3,batch_label,batch_phd_data) in enumerate(test_dataloader):
                    with torch.no_grad():
                        # move data to cuda
                        batch_timeseries1 = batch_timeseries1.to(device)
                        batch_timeseries2 = batch_timeseries2.to(device)
                        batch_timeseries3 = batch_timeseries3.to(device)
                        batch_label = batch_label.to(device)
                        batch_phd_data = batch_phd_data.to(device)
                        # split timeseries
                        dyn_t1, sampling_points1= util.bold.process_dynamic_timeseries(batch_timeseries1, opt.window_size,opt.window_stride, opt.dynamic_length)
                        sampling_endpoints1 = [p + opt.window_size for p in sampling_points1]

                        dyn_t2, sampling_points2 = util.bold.process_dynamic_timeseries(batch_timeseries2, opt.window_size,opt.window_stride, opt.dynamic_length)
                        sampling_endpoints2 = [q + opt.window_size for q in sampling_points2]

                        dyn_t3, sampling_points3 = util.bold.process_dynamic_timeseries(batch_timeseries3,opt.window_size,opt.window_stride, opt.dynamic_length)
                        sampling_endpoints3 = [r + opt.window_size for r in sampling_points3]

                        dyn_t1 = torch.tensor(dyn_t1, dtype=torch.float32)
                        dyn_t2 = torch.tensor(dyn_t2, dtype=torch.float32)
                        dyn_t3 = torch.tensor(dyn_t3, dtype=torch.float32)
                        test_label = torch.tensor(batch_label, dtype=torch.long)
                        phd_ftrs = torch.tensor(batch_phd_data, dtype=torch.float32)

                        logit, latent,loss, attention = step(
                            model = HSGLM,
                            criterion=criterion,
                            t1=dyn_t1,
                            t2=dyn_t2,
                            t3=dyn_t3,
                            label=test_label,
                            phd_ftrs = phd_ftrs,
                            clip_grad=opt.clip_grad,
                            device=device,
                            optimizer=None,
                            scheduler=None,
                        )
                        pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                        prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                        valid_loss_accumulate += loss.detach().cpu().numpy()
                        logger.add(k=fold, pred=pred.detach().cpu().numpy(), true=test_label.detach().cpu().numpy(),prob=prob.detach().cpu().numpy())
                val_losses.append(valid_loss_accumulate / len(test_dataloader))
                samples = logger.get(fold)
                metrics = logger.evaluate(fold)
                print('---Valid'.format(epoch + 1), metrics)
                summary_writer_val.add_scalar('loss', valid_loss_accumulate / len(test_dataloader), epoch)
                if dataset.num_classes > 1: summary_writer_val.add_pr_curve('precision-recall', samples['true'],samples['prob'][:, 1], epoch) #
                [summary_writer_val.add_scalar(key, value, epoch) for key, value in metrics.items() if not key == 'fold']
                #[summary_writer_val.add_image(key,make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True),epoch) for key, value in attention.items()]#将最后一个时间步的注意力权重可视化为图像，并记录到 TensorBoard 中
                # [summary_writer_val.add_image("node-attention", make_grid(attention["node-attention"][-1], normalize=True,scale_each=True), epoch)]
                summary_writer_val.flush()

        # finalize fold
        torch.save(HSGLM.state_dict(), os.path.join('{}_result'.format(opt.dataset), 'model', str(fold), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})
    summary_writer.close()
    summary_writer_val.close()
    os.remove(os.path.join('{}_result'.format(opt.dataset), 'checkpoint.pth'))

def test(opt):
    # set seed and device
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    else:
        device = torch.device("cpu")
    os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention'), exist_ok=True)
    print(' Loading Dataset')
    # define dataset
    if opt.dataset == 'ABIDE1':
        dataset = DatasetABIDEI(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro,atlas=opt.atlas, site=opt.site)
    elif opt.dataset == 'ABIDE2':
        dataset = DatasetABIDEII(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro,atlas=opt.atlas, site=opt.site)
    elif  opt.dataset=='ADHD':
        dataset = DatasetADHD(sourcedir='F:/', threshod=opt.threshod, dynamic_length=opt.dynamic_length,shapiro = opt.shapiro, atlas=opt.atlas,site=opt.site)
    # elif  opt.dataset=='MDD':
    #   dataset = DatasetMDD(sourcedir='F:/', threshod=50, dynamic_length=opt.dynamic_length, atlas='AAL-116',site='NYU')
    else:raise
    timeseries1, timeseries2,timeseries3,label,phonetic_data = dataset.load_data()
    num_ROIs1 = timeseries1.shape[2]
    num_ROIs2 = timeseries2.shape[2]
    num_ROIs3 = timeseries3.shape[2]
    phd_dim = phonetic_data.shape[1]
    cv_splits = dataset.data_split(opt.folds)
    logger = util.logger.Logger(opt.folds, 2)  #
    # start testing
    print("\r\n=====Start Test =====")
    folds_metrics = {}
    for fold in range(opt.folds):
        os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention1', str(fold)), exist_ok=True)
        os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention2', str(fold)), exist_ok=True)
        os.makedirs(os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention3', str(fold)), exist_ok=True)
        print("\r\n========================== Fold {} ==========================".format(fold + 1))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        logger.initialize(fold)

        train_dataset = TensorDataset(torch.from_numpy(timeseries1[train_ind]),torch.from_numpy(timeseries2[train_ind]),torch.from_numpy(timeseries3[train_ind]),torch.from_numpy(label[train_ind]),torch.from_numpy(phonetic_data[train_ind]))
        train_dataloader = DataLoader(train_dataset, batch_size=opt.minibatch_size, shuffle=False)

        test_dataset = TensorDataset(torch.from_numpy(timeseries1[test_ind]),torch.from_numpy(timeseries2[test_ind]),torch.from_numpy(timeseries3[test_ind]), torch.from_numpy(label[test_ind]), torch.from_numpy(phonetic_data[test_ind]))
        test_dataloader = DataLoader(test_dataset, batch_size=opt.minibatch_size, shuffle=False,num_workers=opt.num_workers, pin_memory=True)
        HSGLM = model(
            input_dim1=num_ROIs1,
            input_dim2=num_ROIs2,
            input_dim3=num_ROIs3,

            hidden_dim=opt.hidden_dim,
            topk=opt.topk,
            phd_dim=phd_dim,
            num_classes=2,
            num_heads=opt.num_heads,
            num_time=(opt.dynamic_length - opt.window_size) // opt.window_stride + 1,
            window_size=opt.window_size,
            kernel_size=opt.kernel_size,
            cls_token=opt.cls_token,
            readout=opt.readout)
        HSGLM = LHSGLM.to(device)
        HSGLM.load_state_dict(torch.load(os.path.join('{}_result'.format(opt.dataset), 'model', str(fold), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()
        fold_attention1 = {'node_attention': []}
        fold_attention2 = {'node_attention': []}
        fold_attention3 = {'node_attention': []}
        summary_writer = SummaryWriter(os.path.join('{}_result'.format(opt.dataset), 'summary', str(fold), 'test'))

        #start test
        logger.initialize(fold)
        loss_accumulate = 0.0
        for i, (batch_timeseries1, batch_timeseries2,batch_timeseries3,batch_label,batch_phd_data) in enumerate(test_dataloader):
            with torch.no_grad():
                # move data to cuda
                batch_timeseries1 = batch_timeseries1.to(device)
                batch_timeseries2 = batch_timeseries2.to(device)
                batch_timeseries3 = batch_timeseries3.to(device)
                batch_label = batch_label.to(device)
                batch_phd_data = batch_phd_data.to(device)
                # split timeseries
                dyn_t1, sampling_points1 = util.bold.process_dynamic_timeseries(batch_timeseries1, opt.window_size, opt.window_stride, opt.dynamic_length)
                sampling_endpoints1 = [p + opt.window_size for p in sampling_points1]

                dyn_t2, sampling_points2 = util.bold.process_dynamic_timeseries(batch_timeseries2, opt.window_size,opt.window_stride, opt.dynamic_length)
                sampling_endpoints2 = [q + opt.window_size for q in sampling_points2]

                dyn_t3, sampling_points3 = util.bold.process_dynamic_timeseries(batch_timeseries3, opt.window_size, opt.window_stride, opt.dynamic_length)
                sampling_endpoints3 = [r + opt.window_size for r in sampling_points3]

                dyn_t1 = torch.tensor(dyn_t1, dtype=torch.float32)
                dyn_t2 = torch.tensor(dyn_t2, dtype=torch.float32)
                dyn_t3 = torch.tensor(dyn_t3, dtype=torch.float32)
                test_label = torch.tensor(batch_label, dtype=torch.long)
                phd_ftrs = torch.tensor(batch_phd_data, dtype=torch.float32)
                logit,latent, loss, attention= step(
                    model=HSGLM,
                    criterion=criterion,
                    t1=dyn_t1,
                    t2=dyn_t2,
                    t3=dyn_t3,
                    label=test_label,
                    phd_ftrs = phd_ftrs,
                    clip_grad=opt.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )
                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                logger.add(k=fold, pred=pred.detach().cpu().numpy(), true=test_label.detach().cpu().numpy(),prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()
                fold_attention1['node_attention'].append(attention['node-attention1'].detach().cpu().numpy())
                fold_attention2['node_attention'].append(attention['node-attention2'].detach().cpu().numpy())
                fold_attention2['node_attention'].append(attention['node-attention3'].detach().cpu().numpy())
                # latent_accumulate.append(latent.detach().cpu().numpy())
        # summarize results
        samples = logger.get(fold)
        metrics = logger.evaluate(fold)
        summary_writer.add_scalar('loss', loss_accumulate / len(test_dataloader))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:, 1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key == 'fold']
        summary_writer.flush()

        # finalize fold
        folds_metrics[f'fold{fold + 1}'] = metrics
        print('fold--{}:'.format(fold + 1), metrics)
        logger.to_csv('{}_result'.format(opt.dataset), fold)
        for key, value in fold_attention1.items():
            torch.save(value, os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention1', str(fold), f'{key}.pth'))
        for key, value in fold_attention2.items():
            torch.save(value, os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention2', str(fold), f'{key}.pth'))
        for key, value in fold_attention3.items():
            torch.save(value,os.path.join('{}_result'.format(opt.dataset), 'attention', 'attention3', str(fold), f'{key}.pth'))

    # finalize experiment
    # logger.to_csv('{}_result'.format(opt.dataset))
    final_metrics = logger.evaluate()
    print('Mean:',final_metrics)
    torch.save(logger.get(), os.path.join('{}_result'.format(opt.dataset), 'samples.pkl'))

