import os
# os.environ['TL_BACKEND'] = 'tensorflow' # Just modify this line, easily switch to any framework! PyTorch will coming soon!
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'tensorflow'
import time
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from srgan import SRGAN_g, SRGAN_d
from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, RandomFlipHorizontal, Resize, HWC2CHW
import vgg
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
from PIL import Image
import cv2
tlx.set_device('CPU')

###====================== HYPER-PARAMETERS ===========================###
#batch_size = 8
#n_epoch_init = config.TRAIN.n_epoch_init
#n_epoch = config.TRAIN.n_epoch
## create folders to save result images and trained models
save_dir = "samples"
#tlx.files.exists_or_mkdir(save_dir)
checkpoint_dir = "weights"
#tlx.files.exists_or_mkdir(checkpoint_dir)
#
#hr_transform = Compose([
#    RandomCrop(size=(384, 384)),
#    RandomFlipHorizontal(),
#])
#nor = Compose([Normalize(mean=(127.5), std=(127.5), data_format='HWC'),
#              HWC2CHW()])
#lr_transform = Resize(size=(96, 96))
#
#train_hr_imgs = tlx.vision.load_images(path=config.TRAIN.hr_img_path, n_threads = 32)



G = SRGAN_g()
D = SRGAN_d()
VGG = vgg.VGG19(pretrained=True, end_with='pool4', mode='dynamic')
# automatic init layers weights shape with input tensor.
# Calculating and filling 'in_channels' of each layer is a very troublesome thing.
# So, just use 'init_build' with input shape. 'in_channels' of each layer will be automaticlly set.
G.init_build(tlx.nn.Input(shape=(8, 3, 96, 96)))
D.init_build(tlx.nn.Input(shape=(8, 3, 384, 384)))


def train():
    G.set_train()
    D.set_train()
    VGG.set_eval()
    train_ds = TrainData()
    train_ds_img_nums = len(train_ds)
    train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    lr_v = tlx.optimizers.lr.StepDecay(learning_rate=0.05, step_size=1000, gamma=0.1, last_epoch=-1, verbose=True)
    g_optimizer_init = tlx.optimizers.Momentum(lr_v, 0.9)
    g_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
    d_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
    g_weights = G.trainable_weights
    d_weights = D.trainable_weights
    net_with_loss_init = WithLoss_init(G, loss_fn=tlx.losses.mean_squared_error)
    net_with_loss_D = WithLoss_D(D_net=D, G_net=G, loss_fn=tlx.losses.sigmoid_cross_entropy)
    net_with_loss_G = WithLoss_G(D_net=D, G_net=G, vgg=VGG, loss_fn1=tlx.losses.sigmoid_cross_entropy,
                                 loss_fn2=tlx.losses.mean_squared_error)

    trainforinit = TrainOneStep(net_with_loss_init, optimizer=g_optimizer_init, train_weights=g_weights)
    trainforG = TrainOneStep(net_with_loss_G, optimizer=g_optimizer, train_weights=g_weights)
    trainforD = TrainOneStep(net_with_loss_D, optimizer=d_optimizer, train_weights=d_weights)

    # initialize learning (G)
    n_step_epoch = round(train_ds_img_nums // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patch, hr_patch) in enumerate(train_ds):
            step_time = time.time()
            loss = trainforinit(lr_patch, hr_patch)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, float(loss)))

    # adversarial learning (G, D)
    n_step_epoch = round(train_ds_img_nums // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patch, hr_patch) in enumerate(train_ds):
            step_time = time.time()
            loss_g = trainforG(lr_patch, hr_patch)
            loss_d = trainforD(lr_patch, hr_patch)
            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f}, d_loss: {:.3f}".format(
                    epoch, n_epoch, step, n_step_epoch, time.time() - step_time, float(loss_g), float(loss_d)))
        # dynamic learning rate update
        lr_v.step()

        if (epoch != 0) and (epoch % 10 == 0):
            G.save_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
            D.save_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    #valid_hr_imgs = tlx.vision.load_images(path=config.VALID.hr_img_path )
    ###========================LOAD WEIGHTS ============================###
    G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
    G.set_eval()

    
    imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    #valid_hr_img = valid_hr_imgs[imid]


    valid_hr_img = Image.open("/input.jpg")
    valid_hr_img = valid_hr_img.resize((384,384))
    valid_lr_img = np.asarray(valid_hr_img)

    
    hr_size1 = [valid_hr_img.size[0], valid_lr_img.size[1]]
    valid_lr_img = cv2.resize(valid_lr_img, dsize=(hr_size1[1] // 4, hr_size1[0] // 4))
    valid_lr_img_tensor = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]


    valid_lr_img_tensor = np.asarray(valid_lr_img_tensor, dtype=np.float32)
    valid_lr_img_tensor = np.transpose(valid_lr_img_tensor,axes=[2, 0, 1])
    valid_lr_img_tensor = valid_lr_img_tensor[np.newaxis, :, :, :]
    valid_lr_img_tensor= tlx.ops.convert_to_tensor(valid_lr_img_tensor)
    size = [valid_lr_img.shape[0], valid_lr_img.shape[1]]

    out = tlx.ops.convert_to_numpy(G(valid_lr_img_tensor))
    out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
    out = np.transpose(out[0], axes=[1, 2, 0])
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tlx.vision.save_image(out, file_name='valid_gen.png', path=save_dir)
    tlx.vision.save_image(valid_lr_img, file_name='valid_lr.png', path=save_dir)
    tlx.vision.save_image(valid_hr_img, file_name='valid_hr.png', path=save_dir)
    out_bicu = cv2.resize(valid_lr_img, dsize = [size[1] * 4, size[0] * 4], interpolation = cv2.INTER_CUBIC)
    tlx.vision.save_image(out_bicu, file_name='valid_hr_cubic.png', path=save_dir)


if __name__ == '__main__':
    #import argparse

    #parser = argparse.ArgumentParser()

    #parser.add_argument('--mode', type=str, default='train', help='train, eval')

    #args = parser.parse_args()

    #tlx.global_flag['mode'] = args.mode

    #if tlx.global_flag['mode'] == 'train':
    #    train()
    #elif tlx.global_flag['mode'] == 'eval':
        evaluate()
    #else:
    #    raise Exception("Unknow --mode")
