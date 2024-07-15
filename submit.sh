folder=outfolder_ES
DRN/train $folder pickles_ES --nosemi --idx_name all --target trueE --valid_batch_size 2 --train_batch_size 2 --lr_sched Cyclic --max_lr 0.0001 --n_epochs 100
