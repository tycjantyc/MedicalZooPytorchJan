import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss2D, create_loss
from lib.utils.general import prepare_input
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

def segment():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    
    #model_dane = torch.load("../saved_models/UNET2D_checkpoints/UNET2D_08_08___10_25_pioro_/UNET2D_08_08___10_25_pioro__BEST.pth")
    model_dane = torch.load("../saved_models/UNET2D_checkpoints/UNET2D_28_08___12_26_pioro_/UNET2D_28_08___12_26_pioro__BEST.pth")
    model, optimizer = medzoo.create_model(args)
    model.load_state_dict(model_dane["model_state_dict"])
    model.eval()
    
    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    for batch_idx, input_tuple in enumerate(val_generator):

    
        input_tensor, target = prepare_input(input_tuple=input_tuple, args=args)
        output = model(input_tensor)

        criterion = DiceLoss2D(classes = args.classes)
        dice, cos = criterion(output, target)

        output = torch.argmax(output, dim = 1)

        output = output.cpu()
        output = output.detach().numpy()
        output = output.reshape(512, 512)
        output = output.astype(np.uint8)
        


        target = target.cpu()
        target = target.detach().numpy()
        target = target.reshape(512, 512)

        
        
        

        cv2.imshow("okno1", output)
        plt.imsave("okno1.png", output, cmap = "gray")
        plt.imsave("target1.png", target, cmap = "gray")
        cv2.imshow("okno2",target)
        cv2.waitKey(delay=20000)
        
        print("Dice loss:" + str(dice))
        print("Dice coef:" + str(cos))
        break
        
    


    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="pioro")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--samples_train', type=int, default=100)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET2D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'UNET2D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')
    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':
    segment()