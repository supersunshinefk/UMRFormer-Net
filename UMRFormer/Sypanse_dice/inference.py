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
    spleen = label == 1
    pancreas = label == 11
    r_kidney = label == 2
    L_kidney = label == 3
    gallbladder = label == 4
    liver = label == 6
    stomach = label == 7
    aorta = label == 8

    
    return aorta, gallbladder, L_kidney, r_kidney, liver, pancreas, spleen, stomach
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
    # print("fold>>>>>>", fold)
    path='/disk7/fangkun/Synapse/UMRFormer_raw/UMRFormer_raw_data/UMRFormer_raw_data/Task009_MASIMO/'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    # infer_list=sorted(glob.glob(os.path.join(path,'infersTs',fold,'*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'infersTs/UMRFormer_Net_improved_output','*nii.gz')))
    print("loading success...")
    print(label_list)
    print(">>>>>>>>", infer_list)
    Dice_aorta=[]
    Dice_gallbladder=[]
    Dice_L_kidney=[]
    Dice_r_kidney=[]
    Dice_liver=[]
    Dice_pancreas=[]
    Dice_spleen=[]
    Dice_stomach=[]

    file=path + 'inferTs/'+fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice.txt', 'w')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,spacing= read_nii(label_path)
        infer,spacing= read_nii(infer_path)

        label_aorta, label_gallbladder, label_L_kidney, label_r_kidney, label_liver, label_pancreas, label_spleen, label_stomach=process_label(label)
        infer_aorta, infer_gallbladder, infer_L_kidney, infer_r_kidney, infer_liver, infer_pancreas, infer_spleen, infer_stomach=process_label(infer)

        Dice_aorta.append((dice(infer_aorta,label_aorta)))
        Dice_gallbladder.append((dice(infer_gallbladder,label_gallbladder)))
        Dice_L_kidney.append((dice(infer_L_kidney,label_L_kidney)))
        Dice_r_kidney.append((dice(infer_r_kidney,label_r_kidney)))
        Dice_liver.append((dice(infer_liver,label_liver)))
        Dice_pancreas.append((dice(infer_pancreas,label_pancreas)))
        Dice_spleen.append((dice(infer_spleen,label_spleen)))
        Dice_stomach.append((dice(infer_stomach,label_stomach)))
        
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')

        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')

        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_L_kidney: {:.4f}\n'.format(Dice_L_kidney[-1]))
        fw.write('Dice_r_kidney: {:.4f}\n'.format(Dice_r_kidney[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('*'*20+'\n')
         
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    fw.write('Dice_L_kidney'+str(np.mean(Dice_L_kidney))+'\n')
    fw.write('Dice_r_kidney'+str(np.mean(Dice_r_kidney))+'\n')
    fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_L_kidney))
    dsc.append(np.mean(Dice_r_kidney))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_pancreas))
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_stomach))
    

    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    print('done')

if __name__ == '__main__':
    fold='output'
    test(fold)
