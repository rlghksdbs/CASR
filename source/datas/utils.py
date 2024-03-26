import os
from source.datas.benchmark import Benchmark
from source.datas.div2k import DIV2K
from source.datas.val_data_loader import VALPhaseLoader
from torch.utils.data import DataLoader

def create_datasets(args, mode='train'):
    if args.av1 == True:
        train_HR_path = '{}_train_HR'.format(args.train_set)
        validation_HR_path = '{}_val_HR'.format(args.val_set)
        
        if args.all_qp:
            train_LR_path = os.path.join('{}_train_LR_avif'.format(args.train_set), 'all')
            validation_LR_path = os.path.join('{}_val_LR_avif'.format(args.val_set), 'all')
        else: 
            train_LR_path = os.path.join('{}_train_LR_avif'.format(args.train_set), 'qp_{}'.format(args.qp_value))
            validation_LR_path = os.path.join('{}_val_LR_avif'.format(args.val_set), 'qp_{}'.format(args.qp_value))
    else:
        train_HR_path = '{}_train_HR'.format(args.train_set)
        validation_HR_path = '{}_val_HR'.format(args.val_set)
        
        train_LR_path = 'DIV2K_train_LR_bicubic'
        validation_LR_path = 'DIV2K_val_LR_bicubic'

    if mode == 'train':
        div2k = DIV2K(
            os.path.join(args.data_path, 'Train', train_HR_path), 
            os.path.join(args.data_path, 'Train', train_LR_path), 
            os.path.join(args.data_path, '{}_qpALL{}_cache'.format(args.train_set, str(args.all_qp))),
            train=True, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            normalize=args.normalize,
            av1 = args.av1,
            qp_value = args.qp_value, 
            all_qp = args.all_qp
        )
        train_dataloader = DataLoader(dataset=div2k, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)
        
        valid_dataloaders = []
        #if 'DIV2K' in args.eval_sets:
        div2k_val = DIV2K(
            os.path.join(args.data_path, 'Val', validation_HR_path), 
            os.path.join(args.data_path, 'Val', validation_LR_path), 
            os.path.join(args.data_path, '{}_qpALL{}_cache_val'.format(args.val_set, str(args.all_qp))),
            train=False, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            normalize=args.normalize,
            av1 = args.av1,
            qp_value = args.qp_value,
            all_qp = args.all_qp
        )
        valid_dataloaders += [{'name': str(args.val_set), 'dataloader': DataLoader(dataset=div2k_val, batch_size=1, shuffle=False)}]
    else: #test mode
        # test_loader = DIV2K(
        #         os.path.join(args.test_path, 'val_phase_HR/'), 
        #         os.path.join(args.test_path, 'val_phase_LR/'), 
        #         os.path.join(args.test_path, 'val_phase_cache'),
        #         train=False, 
        #         augment=args.data_augment, 
        #         scale=args.scale, 
        #         colors=args.colors, 
        #         patch_size=args.patch_size, 
        #         repeat=args.data_repeat, 
        #     )
        # test_dataloader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)
            
        test_loader = VALPhaseLoader(
            os.path.join(args.test_path, 'val_phase_HR/'), 
            os.path.join(args.test_path, 'val_phase_LR/'), 
            os.path.join(args.test_path, 'val_phase_cache'),
            train=False, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            normalize=args.normalize
        )
        test_dataloader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)
        return test_dataloader
    return train_dataloader, valid_dataloaders