import copy
import csv
import glob
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '/home/gmission/vs-projects/YOLO-World/DATA/CCTV_CocoFormat'

def train(args, params):
    # Model
    model = nn.yolo_v11_n(len(params['names']))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # Load training images directly from directory
    filenames = glob.glob(f'{data_dir}/images/train/*.jpg')
    
    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[0]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

@torch.no_grad()
def test(args, params, model=None):
    # Load validation images directly from directory
    filenames = glob.glob(f'{data_dir}/images/val/*.jpg')

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def freeze_layers(model, freeze):
    
    model_to_freeze = model.module if hasattr(model, 'module') else model
    frozen_count = 0
    
    if isinstance(freeze, int):
        if freeze > 0:
            named_params = list(model_to_freeze.named_parameters())
            total_params = len(named_params)
            
            head_keywords = ['cv2.', 'cv3.', 'dfl', 'head.cv2', 'head.cv3', 'head.dfl', 
                           '.reg', '.cls', 'head.reg', 'head.cls']
            
            backbone_params = []
            head_params = []
            
            for name, param in named_params:
                is_head = any(keyword in name for keyword in head_keywords)
                if is_head:
                    head_params.append((name, param))
                else:
                    backbone_params.append((name, param))
            
            if len(head_params) < 10:
                print(f'⚠️  Warning: Only detected {len(head_params)} head parameters')
                print(f'   Using last 20% of parameters as detection head (safer approach)')
                
                head_start_idx = int(len(named_params) * 0.8)
                backbone_params = named_params[:head_start_idx]
                head_params = named_params[head_start_idx:]
            
            print(f'\n📊 Model structure:')
            print(f'   Total parameters: {total_params}')
            print(f'   Backbone/Neck: {len(backbone_params)} parameters')
            print(f'   Detection Head: {len(head_params)} parameters')
            
            freeze_limit = min(freeze, len(backbone_params))
            
            if freeze >= len(backbone_params):
                print(f'\n⚠️  Warning: Requested freeze={freeze} but only {len(backbone_params)} backbone parameters')
                print(f'   Limiting to {freeze_limit} to preserve detection head')
            
            print(f'\n🔒 Freezing strategy:')
            print(f'   Requested: {freeze} layers')
            print(f'   Actually freezing: {freeze_limit} backbone layers')
            print(f'   Keeping trainable: {len(head_params)} head layers')
            
            print(f'\n❄️  Frozen layers:')
            for i, (name, param) in enumerate(backbone_params):
                if i < freeze_limit:
                    param.requires_grad = False
                    frozen_count += 1
                    # Show first 3 and last 2 frozen layers
                    if i < 3 or i >= freeze_limit - 2:
                        print(f'   [{i+1:3d}] {name}')
                    elif i == 3:
                        print(f'   ... ({freeze_limit - 5} more layers) ...')
                else:
                    param.requires_grad = True
            
            print(f'\n🔥 Trainable head layers:')
            for idx, (name, param) in enumerate(head_params):
                param.requires_grad = True
                if idx < 3:
                    print(f'   {name}')
                elif idx == 3 and len(head_params) > 3:
                    print(f'   ... ({len(head_params) - 3} more head parameters) ...')
            
            print(f'\n✓ Successfully frozen {frozen_count}/{len(backbone_params)} backbone layers')
            print(f'✓ All {len(head_params)} detection head parameters remain trainable')
    
    elif isinstance(freeze, list):
        head_keywords = ['cv2.', 'cv3.', 'dfl', 'head.cv2', 'head.cv3', 'head.dfl', 
                        '.reg', '.cls', 'head.reg', 'head.cls']
        
        print(f'\n🔒 Freezing specific layers matching: {freeze}')
        print(f'❄️  Frozen layers:')
        
        for name, param in model_to_freeze.named_parameters():
            is_head = any(keyword in name for keyword in head_keywords)
            
            if not is_head and any(layer_name in name for layer_name in freeze):
                param.requires_grad = False
                frozen_count += 1
                print(f'   {name}')
            else:
                param.requires_grad = True
        
        print(f'\n✓ Frozen {frozen_count} parameters (head layers excluded)')
    else:
        for param in model_to_freeze.parameters():
            param.requires_grad = True
        frozen_count = 0
    
    return frozen_count


def finetune(args, params):
    
    print(f'\n📦 Loading pretrained model...')
    print(f'   Path: {args.pretrained_path}')
    
    try:
        checkpoint = torch.load(args.pretrained_path, map_location='cuda')
    except Exception as e:
        print(f'❌ Error loading checkpoint: {e}')
        return
    
    model = nn.yolo_v11_n(len(params['names']))
    
    if 'model' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model'].state_dict())
            start_epoch = checkpoint.get('epoch', 0)
            print(f'✓ Loaded model (trained for {start_epoch} epochs)')
        except RuntimeError as e:
            print(f'⚠️  Warning: State dict mismatch: {e}')
            print(f'   Attempting partial weight loading...')
            
            pretrained_dict = checkpoint['model'].state_dict()
            model_dict = model.state_dict()
            
            matched_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            
            print(f'✓ Loaded {len(matched_dict)}/{len(model_dict)} matching weights')
            start_epoch = 0
    else:
        model.load_state_dict(checkpoint)
        start_epoch = 0
        print(f'✓ Loaded model weights')
    
    model.cuda()
    
    if args.freeze > 0:
        print('\n' + '='*80)
        print('LAYER FREEZING CONFIGURATION')
        print('='*80)
        
        frozen_count = freeze_layers(model, args.freeze)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen = total - trainable
        
        print(f'\n📊 Parameter Statistics:')
        print(f'   Total:      {total:>12,} parameters')
        print(f'   Frozen:     {frozen:>12,} parameters ({100*frozen/total:>5.1f}%)')
        print(f'   Trainable:  {trainable:>12,} parameters ({100*trainable/total:>5.1f}%)')
    else:
        print(f'\n🔥 No layer freezing - training all parameters')
        trainable = sum(p.numel() for p in model.parameters())
        total = trainable
        print(f'   Trainable parameters: {trainable:,}')
    
    print('\n' + '='*80)
    print('OPTIMIZER CONFIGURATION')
    print('='*80)
    
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    weight_decay = params['weight_decay'] * args.batch_size * args.world_size * accumulate / 64
    
    finetune_lr = args.finetune_lr if hasattr(args, 'finetune_lr') and args.finetune_lr else params['min_lr'] * 0.1
    
    print(f'\n⚙️  Training hyperparameters:')
    print(f'   Learning rate:        {finetune_lr:.2e}')
    print(f'   Weight decay:         {weight_decay:.2e}')
    print(f'   Momentum:             {params["momentum"]}')
    print(f'   Batch size:           {args.batch_size}')
    print(f'   Gradient accumulation: {accumulate}')
    print(f'   World size (GPUs):    {args.world_size}')
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        raise ValueError('❌ ERROR: No trainable parameters! All layers are frozen.')
    
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=finetune_lr,
            betas=(params['momentum'], 0.999),
            weight_decay=weight_decay
        )
        print(f'   Optimizer:            AdamW')
    else:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=finetune_lr,
            momentum=params['momentum'],
            weight_decay=weight_decay,
            nesterov=True
        )
        print(f'   Optimizer:            SGD (Nesterov)')
    
    ema = util.EMA(model) if args.local_rank == 0 else None
    
    print('\n' + '='*80)
    print('DATASET CONFIGURATION')
    print('='*80)
    
    print(f'\n📁 Loading training data...')
    print(f'   Data directory: {data_dir}')
    
    filenames = glob.glob(f'{data_dir}/images/train/*.jpg')
    
    if len(filenames) == 0:
        raise ValueError(f'❌ ERROR: No training images found in {data_dir}/images/train/')
    
    print(f'   Found: {len(filenames)} training images')
    
    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)
    
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn
    )
    
    num_steps = len(loader)
    print(f'   Steps per epoch: {num_steps}')
    
    scheduler = util.LinearLR(args, params, num_steps)
    
    if args.distributed:
        print(f'\n🌐 Setting up Distributed Data Parallel...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        print(f'✓ DDP enabled on {args.world_size} GPUs')
    
    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    
    finetune_weights_dir = 'weights/finetune'
    if args.local_rank == 0:
        if not os.path.exists(finetune_weights_dir):
            os.makedirs(finetune_weights_dir)
        print(f'\n💾 Checkpoint directory: {finetune_weights_dir}')
    
    finetune_epochs = args.finetune_epochs if hasattr(args, 'finetune_epochs') and args.finetune_epochs else max(args.epochs // 10, 100)
    
    print('\n' + '='*80)
    print(f'STARTING TRAINING: {finetune_epochs} EPOCHS')
    print('='*80 + '\n')
    
    with open(f'{finetune_weights_dir}/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=[
                'epoch', 'box', 'cls', 'dfl',
                'Recall', 'Precision', 'mAP@50', 'mAP'
            ])
            logger.writeheader()
        
        for epoch in range(finetune_epochs):
            model.train()
            
            if args.freeze > 0:
                model_to_check = model.module if hasattr(model, 'module') else model
                for module in model_to_check.modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        if not module.weight.requires_grad:
                            if 'BatchNorm' in module.__class__.__name__:
                                module.eval()
            
            if args.distributed:
                sampler.set_epoch(epoch)
            
            if finetune_epochs - epoch == 10:
                loader.dataset.mosaic = False
                if args.local_rank == 0:
                    print('\n⚠️  Mosaic augmentation disabled for final 10 epochs\n')
            
            p_bar = enumerate(loader)
            
            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)
            
            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            
            for i, (samples, targets) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)
                
                samples = samples.cuda().float() / 255
                
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
                
                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
                
                loss_box *= args.batch_size * args.world_size
                loss_cls *= args.batch_size * args.world_size
                loss_dfl *= args.batch_size * args.world_size
                
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()
                
                if step % accumulate == 0:
                    amp_scale.unscale_(optimizer)
                    util.clip_gradients(model) 
                    
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    
                    if ema:
                        ema.update(model)
                
                torch.cuda.synchronize()
                
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                    s = ('%10s' * 2 + '%10.3g' * 3) % (
                        f'{epoch + 1}/{finetune_epochs}',
                        memory,
                        avg_box_loss.avg,
                        avg_cls_loss.avg,
                        avg_dfl_loss.avg
                    )
                    p_bar.set_description(s)
            
            if args.local_rank == 0:
                last = test(args, params, ema.ema)
                
                logger.writerow({
                    'epoch': str(epoch + 1).zfill(3),
                    'box': str(f'{avg_box_loss.avg:.3f}'),
                    'cls': str(f'{avg_cls_loss.avg:.3f}'),
                    'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                    'mAP': str(f'{last[0]:.3f}'),
                    'mAP@50': str(f'{last[1]:.3f}'),
                    'Recall': str(f'{last[2]:.3f}'),
                    'Precision': str(f'{last[3]:.3f}')
                })
                log.flush()
                
                if last[0] > best:
                    best = last[0]
                
                save = {
                    'epoch': start_epoch + epoch + 1,
                    'model': copy.deepcopy(ema.ema),
                    'pretrained_from': args.pretrained_path,
                    'frozen_layers': args.freeze,
                    'finetune_epochs': epoch + 1,
                    'best_mAP': best
                }
                
                torch.save(save, f=f'{finetune_weights_dir}/ft_last.pt')
                if best == last[0]:
                    torch.save(save, f=f'{finetune_weights_dir}/ft_best.pt')
                    if epoch > 0:  
                        print(f'\n✨ New best mAP: {best:.3f} (epoch {epoch+1})\n')
                
                del save
    
    if args.local_rank == 0:
        util.strip_optimizer(f'{finetune_weights_dir}/ft_best.pt')
        util.strip_optimizer(f'{finetune_weights_dir}/ft_last.pt')
        
        print('\n' + '='*80)
        print('✅ FINE-TUNING COMPLETE')
        print('='*80)
        print(f'   Best mAP:     {best:.3f}')
        print(f'   Total epochs: {finetune_epochs}')
        print(f'   Saved to:     {finetune_weights_dir}/')
        print(f'   Best model:   ft_best.pt')
        print(f'   Last model:   ft_last.pt')
        print('='*80 + '\n')


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')    
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--pretrained-path', default='weights/coco/best.pt', type=str)
    parser.add_argument('--finetune-epochs', default=None, type=int)
    parser.add_argument('--finetune-lr', default=None, type=float)
    parser.add_argument('--freeze', default=0, type=int)
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['SGD', 'AdamW'])

    args = parser.parse_args()
    
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)


    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.finetune:
        finetune(args, params)
    if args.test:
        test(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

