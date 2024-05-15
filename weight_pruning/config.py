import argparse

def get_args():
    parser = argparse.ArgumentParser("weight-pruning")
    parser.add_argument('--data_dir', type=str, default='../dataset/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100]')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')
    parser.add_argument('--arch', type=str, default='vgg16', help='[vgg16, resnet19]')
    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-Nf', "--end_iter", default=300, type=int)
    parser.add_argument('-r', '--regeneration_ratio', type=float, default=0.50, help='The regeneration ratio.')
    parser.add_argument('-dt', '--update_frequency', type=int, default=2000, metavar='dt', help='how many iterations to pruning once')
    parser.add_argument('--init_density', type=float, default=1.0)
    parser.add_argument('--final_density', type=float, default=0.1)
    parser.add_argument('-sf', '--final_sparsity', type=float, default=0.9, help='The sparsity of the overall sparse network.')
    parser.add_argument('--init_prune_epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('-Np', '--final_prune_epoch', type=int, default=200, help='The density of the overall sparse network.')
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--save', type=str, default='./model_save', help='model save path')
    parser.add_argument('--show', action='store_true', default=False,
                    help='show loss and accuracy list')
    parser.add_argument('--proportion', action='store_true', default=False,
                    help='show regeneration survival proportion of each pruning iteration')

    return parser
