import pytorch_lightning as pl

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

import argparse
DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
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
    
    parser.add_argument("--max_epochs", type=int, required=False, default=15, help="max epochs")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, required=False, default=120, help="train_batch_size")
    parser.add_argument("--exp_name", type=str, required=False, default=None, help="exp_name")
    parser.add_argument("--D", type=int, required=False, default=128, help="scluster_dim")
    parser.add_argument("--K", type=int, required=False, default=64, help="num_clusters")
    parser.add_argument("--model_name", type=str, required=False, default='dinov2_vitb14', help="model_name")

    args = parser.parse_args()

    return args

def str2bool(v):
    if v.lower() in ('yes', 'y'):
        return True
    elif v.lower() in ('no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    args = parse_args()
    agg_config = {
        'num_channels': DINOV2_ARCHS[args.model_name],
        'num_clusters': args.K,
        'cluster_dim': args.D,
        'token_dim': 256,

        'bilinear': str2bool(args.bilinear),
        # 'recalibration': args.recalibration,
        'singlebranch_mid_dim': args.singlebranch_mid_dim,
        'singlebranch_feature_dim': args.singlebranch_feature_dim,
        'singlebranch_split_dim': args.singlebranch_split_dim,
        'remove_mean': str2bool(args.remove_mean),
        'constant_norm': args.constant_norm,
        'post_norm': args.post_norm,
        'with_token': str2bool(args.with_token),
        'final_norm': str2bool(args.final_norm),
    }
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
    
    datamodule = GSVCitiesDataModule(
        batch_size=args.train_batch_size,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=10,
        show_data_stats=True,
        val_set_names=['pitts30k_val', 'msls_val'], # pitts30k_val, msls_val
    )
    
    model = VPRModel(
        #---- Encoder
        backbone_arch = args.model_name,
        backbone_config={
            # 'model_name': args.model_name,
            'num_trainable_blocks': args.num_trainable_blocks,
            'num_recalib_blocks': args.num_recalib_blocks,
            'return_token': True,
            'norm_layer': True,
            'recalibration': args.recalibration,
        },
        agg_arch='CFProbing',
        agg_config=agg_config,

        # lr = 6e-5,
        lr = args.lr,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': args.max_epochs * 1000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='pitts30k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        # default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        default_root_dir=logs_dir, # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        # precision='16', # we use half precision to reduce  memory usage
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)