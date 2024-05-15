Modified based on the code of 
Membrane Potential Batch Normalization for Spiking Neural Networks
Sparse Training via Boosting Pruning Plasticity with Neuroregeneration
Learning Efficient Convolutional Networks Through Network Slimming

This code contains unstructed pruning and structured pruning experments.

For run our code, need Pytorch and spikingjelly pacakage:

pip install spikingjelly==0.0.0.0.12

For Tiny-ImageNet in unstructured pruning, need rename the data,

cd dataset

download dataset and run:

python tiny_reprocess.py

### Unstructured pruning
For unstructured pruning, 

cd weight_pruning

#### Perfermance Experiments 
For pruning and regeneration, run:

python main.py --dataset cifar100 --arch vgg16 --optimizer sgd --batch_size 128 --learning_rate 3e-1 --timestep 5 -wd 0.0001 --scheduler cosine --gpu 0 -Nf 300 -Np 200 -dt 2000 -r 0.2 -sf 0.90 --save ./model_save/vgg16_s0.90_r0.2 

#### Feature Extraction Experiments 

For feature extraction comparison, run:

python feature_extraction.py --dataset cifar100 --arch vgg16 --batch_size 128 --timestep 5 --gpu 0  --save ./model_save/vgg16_s0.90_r0.2



### Structured pruning
For structured pruning,

cd channel_pruning

#### Perfermance Experiments 
For pretrain the model with L1 sparsity regularization, run:

python -u main.py --timestep 5 --dataset cifar100 --arch vgg16 --epochs 160 --batch-size 128 --lr 3e-1 -sr --s 0.0001 --save ./model_save/vgg16 --gpu 0

For pruning and regeneration, run:

python -u vggprune.py --dataset cifar100 --percent 0.6764 --model ./model_save/vgg16/model_best.pth.tar --save ./model_save/vgg16/s0.6764/ --growth_rate 0.05 --gpu 0

For fine-tune the pruned model:
 
python -u main.py --timestep 5 --dataset cifar100 --arch vgg16 --refine ./model_save/vgg16/s0.6764/pruned_model.pth.tar --epochs 160 --batch-size 128 --lr 3e-1 --save ./model_save/vgg16/s0.6764/ --gpu 0

#### Ablation Experiments 
For feature extraction comparison, run:

python feature_extraction.py --dataset cifar100 --arch resnet19 --batch-size 128 --refine ./model_save/resnet19/s0.658/pruned_model.pth.tar --final_model ./model_save/resnet19/s0.658/model_final_best_pruned.pth.tar --gpu 0

For importance comparison, We need model1 and model2 after fine-tuning. The pruned model,  masks and final model must be in the same path,  run:

python importance_analysis.py --arch resnet19 --save1 ./model_save/resnet19/s0.6325/ --save2 ./model_save/resnet19/s0.658/

For Loss and Accuracy during fine-tune, open --show when fine-tune:

python -u main.py --timestep 5 --dataset cifar100 --arch resnet19 --refine ./model_save/resnet19/s0.658/pruned_model.pth.tar --epochs 160 --batch-size 128 --lr 3e-1 --save ./model_save/resnet19/s0.658/ --gpu 0 --show


