import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import os
import re

from vpr_model import VPRModel
from utils.validation import get_validation_recalls
# Dataloader
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset

VAL_DATASETS = ['MSLS', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']
DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}
def str2bool(v):
    if v.lower() in ('yes', 'y'):
        return True
    elif v.lower() in ('no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def input_transform(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)
    
    if 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform=transform)

    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=transform)

    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)

def get_model(args):
    agg_config = {
        'num_channels': DINOV2_ARCHS[args.model_name],
        'num_clusters': args.K,
        'cluster_dim': args.D,
        'token_dim': 256,

        'bilinear': str2bool(args.bilinear),
        'singlebranch_mid_dim': args.singlebranch_mid_dim,
        'singlebranch_feature_dim': args.singlebranch_feature_dim,
        'singlebranch_split_dim': args.singlebranch_split_dim,
        'remove_mean': str2bool(args.remove_mean),
        'constant_norm': args.constant_norm,
        'post_norm': args.post_norm,
        'with_token': str2bool(args.with_token),
        'final_norm': str2bool(args.final_norm),
    }

    model = VPRModel(
        backbone_arch=args.model_name,
        backbone_config={
            # 'num_trainable_blocks': 4,
            # 'return_token': True,
            # 'norm_layer': True,
            'num_trainable_blocks': args.num_trainable_blocks,
            'num_recalib_blocks': args.num_recalib_blocks,
            'return_token': True,
            'norm_layer': True,
            'recalibration': args.recalibration,
        },
        agg_arch='CFProbing',
        agg_config=agg_config,
    )

    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters

    parser.add_argument(
        '--ckpt_versions',
        nargs='+',
        default=[0],
        help='ckpt versions to use',
    )
    
    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', nargs='*', default=None, help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=512, help='Eval Batch size')

    parser.add_argument("--bilinear", type=str, required=False, default='y', help="y:bilinear, n:singlebranch")
    parser.add_argument("--singlebranch_mid_dim", type=int, required=False, default=512, help="singlebranch mid dim")
    parser.add_argument("--singlebranch_feature_dim", type=int, required=False, default=192, help="singlebranch feature dim")
    parser.add_argument("--singlebranch_split_dim", type=int, required=False, default=128, help="singlebranch split dim")
    
    parser.add_argument("--recalibration", type=str, required=False, default='none', help="PEFT for recalibration")
    parser.add_argument("--num_recalib_blocks", type=int, required=False, default=4, help='num_recalib_blocks')
    parser.add_argument("--num_trainable_blocks", type=int, required=False, default=4, help='num_trainable_blocks')

    parser.add_argument("--remove_mean", type=str, required=False, default='y', help="remove mean")
    parser.add_argument("--constant_norm", type=str, required=False, default='softmax', help="constant norm")
    parser.add_argument("--post_norm", type=str, required=False, default='SqrtColL2', help="post norm")
    parser.add_argument("--with_token", type=str, required=False, default='y', help="with token")
    parser.add_argument("--final_norm", type=str, required=False, default='y', help="final norm")
    parser.add_argument("--ckpt", type=str, required=False, default='best', help="best or last ckpt")
    parser.add_argument("--max_epochs", type=int, required=False, default=15, help="max epochs")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, required=False, default=120, help="train_batch_size")
    parser.add_argument("--exp_name", type=str, required=False, default=None, help="exp_name")
    parser.add_argument("--D", type=int, required=False, default=128, help="scluster_dim")
    parser.add_argument("--K", type=int, required=False, default=64, help="num_clusters")
    parser.add_argument("--model_name", type=str, required=False, default='dinov2_vitb14', help="model_name")

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')
        
        args.image_size = tuple(map(int, args.image_size))

    return args

def get_ckpt_path(dir, type='last'):
    if type == 'last':
        return dir + '/last.ckpt'
    
    best_r1_path = 'last.ckpt'
    best_r1 = 0.0
    for file in os.listdir(dir):
        if 'R1' in file:
            r1 = re.findall(r'R1\[(.*?)\]', file)[0]
            if best_r1 < float(r1):
                best_r1 = float(r1)
                best_r1_path = file
    
    return dir + '/' + best_r1_path

def eval(args, model, ckpt_path):
    eval_res = {}

    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    
    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        val_loader = DataLoader(val_dataset, num_workers=16, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        
        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        print('total_size', descriptors.shape[0], num_queries + num_references)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=False,
        )

        del descriptors
        print('========> DONE!\n\n')
        
        eval_res[val_name] = preds
    
    return eval_res


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    if args.exp_name:
        if args.model_name != "dinov2_vitb14":
            logs_dir = "M{}-BP-B{}-RM{}-CN{}-PN{}-RECB_{}-NRECB_{}-NTRAIN_{}-EP{}-LR{}-BS{}-D{}-K{}-exp{}".format(
                args.model_name,
                args.bilinear,
                args.remove_mean,
                args.constant_norm,
                args.post_norm,
                args.recalibration,
                args.num_recalib_blocks,
                args.num_trainable_blocks,
                args.max_epochs,
                args.lr,
                args.train_batch_size,
                args.D,
                args.K,
                args.exp_name
            )
        else:
            logs_dir = "BP-B{}-RM{}-CN{}-PN{}-RECB_{}-NRECB_{}-NTRAIN_{}-EP{}-LR{}-BS{}-D{}-K{}-exp{}".format(
                args.bilinear,
                args.remove_mean,
                args.constant_norm,
                args.post_norm,
                args.recalibration,
                args.num_recalib_blocks,
                args.num_trainable_blocks,
                args.max_epochs,
                args.lr,
                args.train_batch_size,
                args.D,
                args.K,
                args.exp_name
            )
    else:
        if args.model_name != "dinov2_vitb14":
            logs_dir = "M{}-BP-B{}-RM{}-CN{}-PN{}-RECB_{}-NRECB_{}-NTRAIN_{}-EP{}-LR{}-BS{}-D{}-K{}".format(
                args.model_name,
                args.bilinear,
                args.remove_mean,
                args.constant_norm,
                args.post_norm,
                args.recalibration,
                args.num_recalib_blocks,
                args.num_trainable_blocks,
                args.max_epochs,
                args.lr,
                args.train_batch_size,
                args.D,
                args.K,
            )
        else:
            logs_dir = "BP-B{}-RM{}-CN{}-PN{}-RECB_{}-NRECB_{}-NTRAIN_{}-EP{}-LR{}-BS{}-D{}-K{}".format(
                args.bilinear,
                args.remove_mean,
                args.constant_norm,
                args.post_norm,
                args.recalibration,
                args.num_recalib_blocks,
                args.num_trainable_blocks,
                args.max_epochs,
                args.lr,
                args.train_batch_size,
                args.D,
                args.K,
            )
    
    model = get_model(args)

    eval_times_res = {}

    for ckpt_version in args.ckpt_versions:
        ckpt_dir = logs_dir + '/lightning_logs/version_' + str(ckpt_version) + '/checkpoints'
        ckpt_path = get_ckpt_path(ckpt_dir, args.ckpt)

        eval_res = eval(args, model, ckpt_path)

        eval_times_res[ckpt_version] = eval_res
    
    val_avg1, val_avg5, val_avg10 = {}, {}, {}
        
    for val_name in args.val_datasets:
        avg_1, avg_5, avg_10 = 0, 0 ,0

        for ckpt_version in args.ckpt_versions:
            avg_1 = avg_1 + eval_times_res[ckpt_version][val_name][1]
            avg_5 = avg_5 + eval_times_res[ckpt_version][val_name][5]
            avg_10 = avg_10 + eval_times_res[ckpt_version][val_name][10]
        
        val_avg1[val_name] = round(100 * avg_1 / len(args.ckpt_versions), 1)
        val_avg5[val_name] = round(100 * avg_5 / len(args.ckpt_versions), 1)
        val_avg10[val_name] = round(100 * avg_10 / len(args.ckpt_versions), 1)
            

    save_path = logs_dir + '/eval_res_' + args.ckpt + '.txt'
    with open(save_path, 'w') as f:
        f.write(logs_dir)
        f.write('\n')

        f.write('ckpt_version ')
        for val_name in args.val_datasets:
            f.write(val_name + ' ')
        f.write('\n')

        for ckpt_version in args.ckpt_versions:
            f.write(str(ckpt_version) + ' ')
            for val_name in args.val_datasets:
                k1 = round(100 * eval_times_res[ckpt_version][val_name][1], 1)
                k5 = round(100 * eval_times_res[ckpt_version][val_name][5], 1)
                k10 = round(100 * eval_times_res[ckpt_version][val_name][10], 1)

                f.write(str(k1) + ' ' + str(k5) + ' ' + str(k10) + ' ')
            f.write('\n')
        
        f.write('avg ')
        for val_name in args.val_datasets:
            f.write(str(val_avg1[val_name]) + ' ' + str(val_avg5[val_name]) + ' ' + str(val_avg10[val_name]) + ' ')
        f.write('\n')
