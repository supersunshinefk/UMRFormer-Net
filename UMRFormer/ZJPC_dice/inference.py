import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage


def read_nii(path):
    itk_img=sitk.ReadImage(path)
    spacing=np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img),spacing

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    # rv = label == 1
    # myo = label == 2
    # lv = label == 3
    # pancreas = label == 1
    cancer = label == 1

    
    return cancer
'''    
def hd(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
'''

def hd(pred,gt):
    #labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    #labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    #hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    #hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    #return hausdorffcomputer.GetAverageHausdorffDistance()
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return hd95
    else:
        return 0


def test(fold):
    # path='../DATASET/UMRFormer_raw/UMRFormer_raw_data/Task001_ACDC/'
    path='/disk7/fangkun/Synapse/UMRFormer_raw/UMRFormer_raw_data/UMRFormer_raw_data/Task005_ZJPancreasCancer/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'infersTs/nnTrans_output','*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)
    # Dice_pancreas=[]
    Dice_cancer=[]
    hd95_cancer=[]
    

    file = path + 'inferTs/' + fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/UMRFormer_UNet_dice.txt', 'w')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,spacing= read_nii(label_path)
        infer,spacing= read_nii(infer_path)
        # print(np.max(label))
        # print(np.max(infer))
        # break
        label_cancer=process_label(label)
        infer_cancer=process_label(infer)
        
        # Dice_pancreas.append(dice(infer_pancreas,label_pancreas))
        Dice_cancer.append(dice(infer_cancer,label_cancer))
        hd95_cancer.append(hd(infer_cancer,label_cancer))
        
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')

        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        # fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        fw.write('Dice_cancer: {:.4f}\n'.format(Dice_cancer[-1]))
        fw.write('hd95_cancer: {:.4f}\n'.format(hd95_cancer[-1]))
        fw.write('*'*20+'\n')
         
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    # fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    fw.write('Dice_cancer'+str(np.mean(Dice_cancer))+'\n')
    fw.write('hd95_cancer' + str(np.mean(hd95_cancer)) + '\n')
    fw.write('*'*20+'\n')
    
    dsc=[]
    hd95=[]
    # dsc.append(np.mean(Dice_pancreas))
    dsc.append(np.mean(Dice_cancer))
    hd95.append(np.mean(hd95_cancer))

    

    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    fw.write('hd95:' + str(np.mean(hd95)) + '\n')
    print('done')

if __name__ == '__main__':
    fold='output'
    test(fold)
