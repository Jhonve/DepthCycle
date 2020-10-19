import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3, help="Size of one batch.")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of processes when load data.")
    parser.add_argument("--data_path_file_clean", type=str, default="./DataPathClean.h5", help="Path to data_path H5 file. (Domain A clean data)")
    parser.add_argument("--data_path_file_noise", type=str, default="./DataPathNoise.h5", help="Path to data_path H5 file. (Domain B noise data)")
    parser.add_argument("--zoom_out_scale", type=int, default=2, help="Scale image for training.")

    parser.add_argument("--num_epoch", type=int, default=24, help="Number of train epoch.")
    parser.add_argument("--num_val_batch", type=int, default=0, help="Number of selected batch for validation.")

    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/", help="Path to the checkpoints.")
    parser.add_argument("--val_res_path", type=str, default="./validationRes/", help="Path to the validation results.")
    parser.add_argument("--test_res_path", type=str, default="../../NoiseResults", help="Path to the test results.")
    # parser.add_argument("--current_model", type=str, default="./checkpoints/23_model.t7", help="Latest trainded model.")
    parser.add_argument("--current_model", type=str, default="", help="Latest trainded model.")

    # Original CycleGANs
    ## basic options
    ### basic parameters
    parser.add_argument('--isTrain', type=bool, default=True, help='is train')
    parser.add_argument('--name', type=str, default='TwoBranch', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    ### model parameters
    parser.add_argument('--model', type=str, default='cycleGAN', help='chooses which model to use. [cycleGAN | test]')
    # parser.add_argument('--model', type=str, default='test', help='chooses which model to use. [cycleGAN | test]')
    parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    ### dataset parameters
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    ### additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

    ## train options 
    ### visdom and HTML visualization parameters
    parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    ### network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    ### training parameters
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

    ## test options
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

    # SHEN ADD
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--model_suffix', type=str, default='_A', help='should match direction.')
    
    opt, _ = parser.parse_known_args()

    ## No use in Original CycleGAN
    ## basic options
    # parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    ## train options
    # parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    # parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
    # parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    ## dataset options
    # parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    # parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    # parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    # parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    # parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    # parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    # parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    # parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    # parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
    ## additional options
    # parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    return opt