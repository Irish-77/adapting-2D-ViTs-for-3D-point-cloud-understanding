from src.train import Pix4PointTrainer


model_config = {
    'model_name': 'Pix4Point',
    'num_classes': 15,
    'pretrained_model':'vit_small_patch16_384.augreg_in21k_ft_in1k', #'vit_base_patch16_384.augreg_in21k_ft_in1k', #
    'pretrained': True,
    'frozen': False,
    'k_neighbors': 16,
    'embed_dim': 384 #768 #
}

dataset_config = {
    'root_dir': './.data/h5_files',
    'split': 'training',
    'variant': 'main_split',
    'augmentation': 'augmentedrot_scale75',
    'num_points': 2048,
    'normalize': True,
    'sampling_method': 'all',
    'use_custom_augmentation': True,
}

train_config = {
    'batch_size': 64,
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'save_interval': 10,
    'epochs': 150,
    't_max': 100,
    'warmup_epochs': 10,
    'min_lr': 1.0e-6,
    'grad_norm_clip': 10
}

trainer = Pix4PointTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda'
)

trainer.train()
