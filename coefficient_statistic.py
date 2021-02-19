import numpy as np
import nibabel as nib
from medpy import metric
from matplotlib import pyplot as plt
def data_provider_read_volume(volume_path):
    """
    Read one nii volume with name format of case_id.nii
    :param volume_path: the full path of a volume
    :return: the volume data, affine matrix, case_id
    """

    nii_volume = nib.load(volume_path)
    data_volume = nii_volume.get_data()
    return  data_volume
def Get_CBAM_Unet_Dice_PVP_Sensitivity(statistic_start_num,statistic_end_num):
    Dice = []
    PVP = []
    Sensitivity = []
    for ID in range(statistic_start_num,statistic_end_num):
        file_path1 = 'C:/Users/12638/Desktop/SRTP/生物医学创新工程/prediction/prediction/mask_case'+ str(ID) + '.nii'
        file_path2 = 'C:/Users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/seg_volume/data_volume/mask_case' + str(ID) + '.nii'
        Dice.append(metric.binary.dc(data_provider_read_volume(file_path1) == 1, data_provider_read_volume(file_path2) == 1))
        PVP.append(metric.binary.precision(data_provider_read_volume(file_path1) == 1, data_provider_read_volume(file_path2) == 1))
        Sensitivity.append(metric.binary.recall(data_provider_read_volume(file_path1) == 1, data_provider_read_volume(file_path2) == 1))
    return Dice,PVP,Sensitivity
def Get_Unet_Dice_PVP_Sensitivity(statistic_start_num,statistic_end_num):
    Dice = []
    PVP = []
    Sensitivity = []
    for ID in range(statistic_start_num,statistic_end_num):
        file_path3 = 'C:/Users/12638/Desktop/SRTP/生物医学创新工程/unet/unet_2d/prediction/mask_case' + str(ID) + '.nii'
        file_path4 = 'C:/Users/12638/dfsdata2/zhangyao_data/DB/SpineSagT2W/standard_data/seg_volume/data_volume/mask_case' + str(ID) + '.nii'
        Dice.append(metric.binary.dc(data_provider_read_volume(file_path3) == 1, data_provider_read_volume(file_path4) == 1))
        PVP.append(metric.binary.precision(data_provider_read_volume(file_path3) == 1, data_provider_read_volume(file_path4) == 1))
        Sensitivity.append(metric.binary.recall(data_provider_read_volume(file_path3) == 1, data_provider_read_volume(file_path4) == 1))
    return Dice,PVP,Sensitivity
def histogram_plot(coefficient,temp_list):
    plt.subplot(1, 1, 1)
    N = len(temp_list)
    index = np.arange(N)
    index = index + 180
    width = 0.35
    plt.bar(index, temp_list, width, align='center', color="#87CEFA")
    plt.xlabel('Mask_ID')
    plt.ylabel(''+str(coefficient))
    plt.title('Cofficient statistic')
    plt.xticks(index)

    plt.show()
    return 0
if __name__ == '__main__':
        Dice, PVP, Sensitivity = Get_CBAM_Unet_Dice_PVP_Sensitivity(181,196)

        print('F1:',np.mean(PVP)*np.mean(Sensitivity)*2/(np.mean(Sensitivity)+np.mean(PVP)))

