from train import APFTrainer

model_config = {
    'num_classes': 15,
    'in_channels': 4, 
    'vit_name': 'vit_base_patch16_224',
    'pretrained': True,
    'embedding_dim': 768,
    'npoint': 196,
    'nsample': 32,
    'dropout_rate': 0.1,
    'dropout_path_rate': 0.1,
}


dataset_config = {
    # REPLACE WITH CUSTOM DATASET PATH
    'root_dir': '.data/h5_files',
    'variant': 'main_split',
    'augmentation': 'base',
    'background': False,
    'use_newsplit': False,
    'train_num_points': 2048,
    'test_num_points': 1024,
    'sampling_method': 'fps',
    'use_apf_augmentation': True,
    'use_custom_augmentation': False,
    'augmentation_probability': 0.0,
    'use_height': True,
}

train_config = {
    'batch_size': 32,
    'save_interval': 100,
    'epochs': 100,
    # Optimizer
    'label_smoothing': 0.3,
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'warmup_epochs': 10,
    'warmup_lr_init': 1e-3,
}


trainer = APFTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda',
    output_dir='./output/apf', 
)

trainer.train()