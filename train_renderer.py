from train import RendererTrainer

dataset_config = {
    'root_dir': '/home/basti/Development/University/3DVision/adapting-2D-ViTs-for-3D-point-cloud-understanding/.data/h5_files',
    'split': 'training',
    'variant': 'main_split',
    'augmentation': 'base',
    'num_points': 2048,
    'normalize': True,
    'sampling_method': 'all',
    'use_custom_augmentation': True,
}

train_config = {
    'batch_size': 16,
    'min_lr': 1e-6,
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'save_interval': 5,
    'epochs': 100,
    'use_lr_scheduler': True,
    'clip_grad_norm': 10.0,
    'save_views_interval': 1,
    # New parameters for view transformation network
    'view_learning_rate': 1e-3,     
    'view_min_lr': 1e-7,            
    'view_weight_decay': 1e-4,      
    'view_warmup_epochs': 10,       
}

model_config = {
    'num_classes': 15,
    'vit_name': 'vit_b_16',
    'adapter_dim': 32,
    'num_views': 3,
    'img_size': 224,
    'pretrained': True, 
    'dropout_rate': 0.2,
    'diff_renderer': True,
    'view_transform_hidden': 64,
}

trainer = RendererTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda',
    output_dir='./experiment/renderer/run_6' 
)

trainer.train()