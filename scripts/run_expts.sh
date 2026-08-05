pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_default_param" 
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_lr" --lightning_module#base_lr=0.00005
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_teacher_temp" --lightning_module#teacher_temp_min=0.005 --lightning_module#teacher_temp_max=0.01
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_medium_prototypes" --lightning_module#projection_dim=65536
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_prototypes" --lightning_module#projection_dim=16384
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_lr_prototypes" --lightning_module#projection_dim=16384 --lightning_module#base_lr=0.00005
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_lr_tt_prototypes" --lightning_module#projection_dim=16384 --lightning_module#teacher_temp_min=0.005 --lightning_module#teacher_temp_max=0.01 --lightning_module#base_lr=0.00005
pkill -f -9 "python -m scripts.run"
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml --run_name="dinov2_pretrain_primus_exp0_lower_lr_tt_medium_prototypes" --lightning_module#projection_dim=65536 --lightning_module#teacher_temp_min=0.005 --lightning_module#teacher_temp_max=0.01 --lightning_module#base_lr=0.00005
pkill -f -9 "python -m scripts.run"