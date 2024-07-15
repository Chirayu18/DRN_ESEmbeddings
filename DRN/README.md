# The_DRN_for_HToAA
- To train the model:
```
    folder=outfolder
    ./train $folder AToGG_pickles_fixedSample_tworeco_EEFlag --nosemi --idx_name all --target trueE --in_layers 3 --mp_layers 4 --out_layers 3  --valid_batch_size 100 --train_batch_size 100  --lr_sched Cyclic --max_lr 0.0001 --pool mean --hidden_dim 128 --n_epochs 100 &>> $folder/training.log
```

- Model architecture code:
https://github.com/Chirayu18/DRN/blob/master/models/DynamicReductionNetworkJit.py

- Code used for making inputs(including rescaling the inputs, applying energy threshold on rechits):
https://github.com/Chirayu18/DRN/blob/master/Extractor/preparePickles ( Wrapper code )
https://github.com/Chirayu18/DRN/blob/master/Extractor/Extract_matrix.py ( Main code )

- Code used for training the model:
https://github.com/Chirayu18/DRN/blob/master/train ( Wrapper code )
https://github.com/Chirayu18/DRN/blob/master/Train.py ( Main code )
