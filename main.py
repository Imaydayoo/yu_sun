# -*- coding: UTF-8 -*-
import argparse
import os
from sun_solver import SunSolver, test_aop_speed
from data_loader import get_loader
from torch.backends import cudnn
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config):
    print('begin')
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'V_Net', 'MDV_Net', 'BM_U_Net', 'LedNet',
                                 'Deform_U_Net', 'MD_Defm_V_Net', 'FCN', 'MD_UNet', 'BiSeNet', 'U_YS_Net', 'U_YS_Net_64', 'U_YS_Net_16', 'MBU_YS_Net_16',
                                 'RealU_YS_Net_16', 'MDBU_YS_Net_16', 'AG_BU_YS_Net_16', 'ENet', 'ERFNet', 'GACN', 'BiseNetV2', 'fcn_resnet50', 'DeepLabV3']:  # 新增网络后需要修改
        print(
            'ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/V_Net/MDV_Net/BM_U_Net/LedNet/Deform_U_Net/MD_Defm_V_Net/MD_UNet/BiSeNet')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # lr = random.random()*0.0005 + 0.0000005  #lr = random.random()*0.0005 + 0.0000005
    lr = 0.0001
    # lr = 0.00001
    # lr = 0.001
    augmentation_prob = 0.5  # augmentation_prob= random.random()*0.7
    epoch = 199  # epoch = random.choice([100,150,200,250])

    # decay_ratio = random.random()*0.8
    decay_ratio = 0.7
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    # config.train_path = './dataset/train/'
    # config.train_path = './dataset_1/train/train'
    config.train_path = './dataset_5_fold/sun_dataset_1/train/train' # my old
    # config.train_path = './dataset_5_fold_new/sun_dataset_1/train/train' #new
    # config.train_path = './quick_1000/train/train'
    # ('--image_size', type=int, default=512)
    # ('--batch_size', type=int, default=1)
    # ('--num_workers', type=int, default=0)
    # ('--augmentation_prob', type=float, default=0.5)  # 0

    # config.valid_path = './dataset/valid/'
    # config.valid_path = './dataset_1/valid/valid'
    # config.valid_path = './dataset_5_fold/sun_dataset_1/valid/valid'
    config.valid_path = './dataset_5_fold_new/sun_dataset_1/valid/valid'
    # config.valid_path = './quick_1000/valid/valid'
    # ('--test_path', type=str, default=r'D:\py_seg\Landmark-Net\dataset\test/')
    config.model_path = './models/temp_models/'
    config.model_path = config.model_path
    # config.test_path = './dataset_1/valid/valid'
    # config.test_path =  './dataset_5_fold_new/sun_dataset_1/test/test'
    config.test_path =  './dataset_yu/test/test'
    config.num_workers = 0
    config.batch_size = 1  #
    config.num_epochs = 70 # 原来70
    # config.image_size = 512 128
    config.image_size = 512
    config.augmentation_prob = 0.3
    # config.model_type = 'SB_DBU_YS_Net_16'
    # config.model_type = 'AG_DBU_YS_Net_16'
    # config.model_type = 'MD_UNet'
    # config.model_type = 'DBU_YS_Net_16'
    # config.model_type = 'U_Net_d1'
    # config.model_type = 'U_Net_nd1'
    # config.model_type = 'U_Net_d2'
    # config.model_type = 'U_Net_d3'
    # config.model_type = 'U_Net_d4'
    # config.model_type = 'U_Net_d5'
    # config.model_type = 'U_Net4000'
    # config.model_type = 'fcn_resnet50'
    # config.model_type = 'fcn_resnet50_d1'
    # config.model_type = 'fcn_resnet50_d2'
    #dd
    # config.model_type = 'fcn_resnet50_d3'
    # config.model_type = 'fcn_resnet50_d4'
    # config.model_type = 'fcn_resnet50_d5'
    # config.model_type = 'CBAM_U_Net'
    # config.model_type = 'ASPP_U_Net'
    # config.model_type = 'R2U_Net'             #后面要继续看
    # config.model_type = 'multiresunet'
    # config.model_type = 'deeplabv3_resnet50'
    # config.model_type = 'deeplabv3_resnet50d5'
    # config.model_type = 'PraNet' # 暂时未实现
    # config.model_type = 'AttU_Net'
    # config.model_type = 'lraspp'
    # config.model_type = 'dscaunet'
    # config.model_type = 'AG_DBU_YS_Net_16'
    # config.model_type = 'AG_BU_YS_Net_16'
    # AttU_Net U_Net
    # config.model_type = 'U2_Net'
    # config.model_type = 'Res_Unet'
    # config.model_type = 'Multi_U_Net'
    # config.model_type = 'SAU_Net'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention_g123'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention4000'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention_d5'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention_d2'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention_d3'
    # config.model_type = 'My_Unet_no_multi_has_shape_no_attention'
    #config.model_type = 'My_Unet_no_multi_has_shape_has_attention' # mul shape at都有
    # config.model_type = 'My_Unet_has_multi_has_shape_has_aspp' # mul shape aspp都有
    # config.model_type = 'My_Unet_has_multi_has_shape_has_res' # mul shape aspp都有
    # config.model_type = 'My_Unet_has_multi_has_shape_has_attu' # mul shape aspp都有
    # config.model_type = 'My_Unet_has_multi_has_shape_has_deform' # mul shape deform都有
    # config.model_type = 'MSSN_GN' #组归一化
    # config.model_type = 'MSSN_GN_1C' #组归一化+初始1通道
    # config.model_type = 'MSSN_GN_1C_nd1' #组归一化+初始1通道


    # config.model_type = 'UnetX'
    # config.model_type = 'BiseNetV2'
    # config.model_type = 'ENet'
    # config.model_type = 'ERFNet'
    # config.model_type = 'deeplabv3_resnet50_light'
    # config.model_type = 'GACN'
    # config.model_type = 'My_light'
    # config.model_type = 'fast_scnn'
    # config.model_type = 'My_ENet'


    #speed aop
    # config.model_type = 'U_Net4000'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention4000'

    # seg
    # config.model_type = 'ENet'
    # config.model_type = 'My_Unet_has_multi_has_shape_no_attention'
    # config.model_type = 'SAU_Net'
    # config.model_type = 'fcn_resnet50'
    # config.model_type = 'U_Net'
    # config.model_type = 'AG_DBU_YS_Net_16'
    # config.model_type = 'MD_UNet'
    # config.model_type = 'AttU_Net'

    #yu new dataset
    # config.model_type = 'AG_DBU_YS_Net_16_YU'
    # config.model_type = 'U_Net_YU'
    config.model_type = 'AttU_Net_YU'
    config.img_ch = 1
    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)


    # solver.generate_output_aop_75(3)
    # solver.generate_output_aop_75(4)
    # solver.generate_output_aop_75(5)
    # solver.generate_result_75()
    # solver.generate_result() # 用模型输出pic结果，并计算aop
    # solver.generate_paper_pic()# 仿真结果部分
    # solver.generate_result_dataset_no(2)

    #My old dataset
    # solver = SunSolver(config, train_loader, valid_loader, test_loader)
    # solver.train()
    # # # # # # # # 测试集测试
    # solver.generate_output_pic(mode='test') # 生成预测图片和分割结果
    # solver.generate_output_aop_sun_normal(mode='test') # 生成aop结果 改了测试集 椭圆拟合
    # # # solver.generate_output_aop_sun_circle(mode='test') # 生成aop结果 改了测试集 重心圆拟合
    # # #solver.generate_output_aop_sun_both(mode='test') # 生成aop结果 改了测试集 椭圆重心圆选一个
    # # # solver.generate_output_aop_sun_clip(mode='test') # 生成aop结果 改了测试集 椭圆重心圆选一个
    # # solver.generate_output_aop_sun_hull(mode='test') # 生成aop结果 改了测试集 凸包
    # solver.generate_result_test_sun() # 统计aop等指标的均值方差中位数等

    #生成注意力图
    # solver.generate_output_pic_g123()
    #测速
    # solver.test_model_speed()
    # solver.test_aop_speed()
    # test_aop_speed(config.model_type, test_loader)

    #new dataset ou
    # solver = SunSolver(config, train_loader, valid_loader, test_loader)
    # solver.train()
    # solver.sun_get_test_result()
    # # # # # # # # # 测试集测试
    # solver.generate_output_pic(mode='test') # 生成预测图片和分割结果
    # solver.generate_output_aop_sun_normal(mode='test') # 生成aop结果 改了测试集 椭圆拟合
    # solver.generate_output_aop_sun_circle(mode='test') # 生成aop结果 改了测试集 重心圆拟合
    # solver.generate_output_aop_sun_both(mode='test') # 生成aop结果 改了测试集 椭圆重心圆选一个
    # solver.generate_output_aop_sun_clip(mode='test') # 生成aop结果 改了测试集 裁剪
    # solver.generate_output_aop_sun_hull(mode='test') # 生成aop结果 改了测试集 凸包
    # solver.generate_result_test_sun() # 统计aop等指标的均值方差中位数等

    #yu dataset
    solver = SunSolver(config, train_loader, valid_loader, test_loader)
    # # solver.train()
    solver.yu_get_test_result(fold='d2')
    solver.yu_get_generate_output_pic(mode='test', fold='d2')
    solver.yu_get_output_aop_sun_normal(mode='test', fold='d2')
    solver.yu_get_result_test_sun(fold='d2')

    # # # # # # # # # 测试集测试
    # solver.generate_output_pic(mode='test') # 生成预测图片和分割结果
    # solver.generate_output_aop_sun_normal(mode='test') # 生成aop结果 改了测试集 椭圆拟合
    # solver.generate_output_aop_sun_circle(mode='test') # 生成aop结果 改了测试集 重心圆拟合
    # solver.generate_output_aop_sun_both(mode='test') # 生成aop结果 改了测试集 椭圆重心圆选一个
    # solver.generate_output_aop_sun_clip(mode='test') # 生成aop结果 改了测试集 裁剪
    # solver.generate_output_aop_sun_hull(mode='test') # 生成aop结果 改了测试集 凸包
    # solver.generate_result_test_sun() # 统计aop等指标的均值方差中位数等

    # lr_list = [0.0001]
    # model_type_list = ['U_Net', 'My_Unet_has_multi_has_shape_no_attention']
    # train_multi_parameter(config, lr_list, model_type_list, train_loader, valid_loader, test_loader)



def train_multi_parameter(config, lr_list, model_type_list, train_loader, validate_loader, test_loader):

    for lr in lr_list:
        for model_type in model_type_list:
            config.lr = lr
            config.model_type = model_type
            solver = SunSolver(config, train_loader, validate_loader, test_loader)
            solver.train()
            # # 测试集测试
            solver.generate_output_pic(mode='test') # 生成预测图片和分割结果
            solver.generate_output_aop_75(mode='test') # 生成aop结果
            solver.generate_result_test_sun() # 统计aop等指标的均值方差中位数等




    # solver.test_dc()



    # # Train and sample the images
    # if config.mode == 'train':
    #     solver.train()
    # elif config.mode == 'test':
    #     solver.test()
    # elif config.mode == 'test_output':
    #     solver.test_output()
    # elif config.mode == 'test_output_pic':
    #     solver.test_output_pic()
    # elif config.mode == 'test_output_pic_hc':
    #     solver.test_output_pic_hc()
    # elif config.mode == 'saveONNX':
    #     filePath = 'D:/py_seg/U-Net/U-Net_vari/model_Onnx/512-Bisenet.onnx'
    #     model_name = 'near-fetalhead-Bisenet-BiSeNet-199-0.0001-139-0.5000-512.pkl'
    #     solver.saveONNX(filePath, model_name)
    # elif config.mode == 'video_output':
    #     solver.video_output(config.video_testpath, config.video_testmodel, config.video_output_csv_path)


# from mmdet.ops import dcn
# dcn.ModulatedDeformConv
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.5)  # 0
    parser.add_argument('--EM_momentum', type=float, default=0.9)
    parser.add_argument('--EM_iternum', type=int, default=3)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test/test_output/test_output_pic/test_output_pic_hc/saveONNX/video_output')
    parser.add_argument('--model_type', type=str, default='MD_UNet',
                        help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/V_Net/MDV_Net/BM_U_Net/LedNet/Deform_U_Net/MD_Defm_V_Net/FCN/MD_UNet/BiSeNet')
    parser.add_argument('--model_path', type=str, default='./modelsv3')
    parser.add_argument('--train_path', type=str,
                        default=r'D:\picturecut\313data_seg\dataset_3\train\train/')  # './dataset/train/'
    parser.add_argument('--valid_path', type=str,
                        default=r'D:\picturecut\313data_seg\dataset_3\valid\valid/')  # './dataset/valid/'
    parser.add_argument('--test_path', type=str, default=r'D:\py_seg\Landmark-Net\dataset\test/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--video_testpath', type=str, default='D:/py_seg/video7')  # VIDEO
    parser.add_argument('--video_testmodel', type=str, default='multitask_use2vgg1d_cls1use.pkl')  # VIDEO
    parser.add_argument('--video_output_csv_path', type=str, default=r'D:\py_seg\video7/aop_video_nocls.csv')

    parser.add_argument('--cuda_idx', type=int, default=1)

    # parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")

    config = parser.parse_args()
    main(config)

##tensorboardX使用

# tensorboard --logdir ./ --host=127.0.0.1   ##runs目录下控制台指令

# import torch
# from tensorboardX import SummaryWriter
#
# writer = SummaryWriter('runs/unet')
# x = torch.FloatTensor([100])
# y = torch.FloatTensor([500])
#
# for epoch in range(200):
#     x /= 1.5
#     y /= 1.5
#     loss = y - x
#     print(loss)
#     writer.add_histogram('zz/x', x, epoch)
#     writer.add_histogram('zz/y', y, epoch)
#     writer.add_scalar('data/x', x, epoch)
#     writer.add_scalar('data/y', y, epoch)
#     writer.add_scalar('data/loss', loss, epoch)
#     writer.add_scalars('data/scalar_group', {'x': x,
#                                              'y': y,
#                                              'loss': loss}, epoch)
#     writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)
#
# # export scalar data to JSON for external processing
# writer.export_scalars_to_json("./test.json")
# writer.close()

# print('\n'.join([''.join([('Love'[(x-y) % len('Love')] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0 else' ') for x in range(-30, 30)]) for y in range(30, -30, -1)]))
