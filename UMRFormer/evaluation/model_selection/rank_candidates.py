#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from UMRFormer_Net.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "UMRFormerPlans"

    overwrite_plans = {
        'UMRFormerTrainerV2_2': ["UMRFormerPlans", "UMRFormerPlansisoPatchesInVoxels"], # r
        'UMRFormerTrainerV2': ["UMRFormerPlansnonCT", "UMRFormerPlansCT2", "UMRFormerPlansallConv3x3",
                            "UMRFormerPlansfixedisoPatchesInVoxels", "UMRFormerPlanstargetSpacingForAnisoAxis",
                            "UMRFormerPlanspoolBasedOnSpacing", "UMRFormerPlansfixedisoPatchesInmm", "UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_warmup': ["UMRFormerPlans", "UMRFormerPlansv2.1", "UMRFormerPlansv2.1_big", "UMRFormerPlansv2.1_verybig"],
        'UMRFormerTrainerV2_cycleAtEnd': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_cycleAtEnd2': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_reduceMomentumDuringTraining': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_graduallyTransitionFromCEToDice': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_independentScalePerAxis': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Mish': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Ranger_lr3en4': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_GN': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_momentum098': ["UMRFormerPlans", "UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_momentum09': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_DP': ["UMRFormerPlansv2.1_verybig"],
        'UMRFormerTrainerV2_DDP': ["UMRFormerPlansv2.1_verybig"],
        'UMRFormerTrainerV2_FRN': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_resample33': ["UMRFormerPlansv2.3"],
        'UMRFormerTrainerV2_O2': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ResencUNet': ["UMRFormerPlans_FabiansResUNet_v2.1"],
        'UMRFormerTrainerV2_DA2': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_allConv3x3': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ForceBD': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ForceSD': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_LReLU_slope_2en1': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_lReLU_convReLUIN': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ReLU': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ReLU_biasInSegOutput': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_ReLU_convReLUIN': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_lReLU_biasInSegOutput': ["UMRFormerPlansv2.1"],
        #'UMRFormerTrainerV2_Loss_MCC': ["UMRFormerPlansv2.1"],
        #'UMRFormerTrainerV2_Loss_MCCnoBG': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Loss_DicewithBG': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Loss_Dice_LR1en3': ["UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Loss_Dice': ["UMRFormerPlans", "UMRFormerPlansv2.1"],
        'UMRFormerTrainerV2_Loss_DicewithBG_LR1en3': ["UMRFormerPlansv2.1"],
        # 'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],
        # 'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],
        # 'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],
        # 'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],
        # 'UMRFormerTrainerV2_fp32': ["UMRFormerPlansv2.1"],

    }

    trainers = ['UMRFormerTrainer'] + ['UMRFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'UMRFormerTrainerNewCandidate24_2',
        'UMRFormerTrainerNewCandidate24_3',
        'UMRFormerTrainerNewCandidate26_2',
        'UMRFormerTrainerNewCandidate27_2',
        'UMRFormerTrainerNewCandidate23_always3DDA',
        'UMRFormerTrainerNewCandidate23_corrInit',
        'UMRFormerTrainerNewCandidate23_noOversampling',
        'UMRFormerTrainerNewCandidate23_softDS',
        'UMRFormerTrainerNewCandidate23_softDS2',
        'UMRFormerTrainerNewCandidate23_softDS3',
        'UMRFormerTrainerNewCandidate23_softDS4',
        'UMRFormerTrainerNewCandidate23_2_fp16',
        'UMRFormerTrainerNewCandidate23_2',
        'UMRFormerTrainerVer2',
        'UMRFormerTrainerV2_2',
        'UMRFormerTrainerV2_3',
        'UMRFormerTrainerV2_3_CE_GDL',
        'UMRFormerTrainerV2_3_dcTopk10',
        'UMRFormerTrainerV2_3_dcTopk20',
        'UMRFormerTrainerV2_3_fp16',
        'UMRFormerTrainerV2_3_softDS4',
        'UMRFormerTrainerV2_3_softDS4_clean',
        'UMRFormerTrainerV2_3_softDS4_clean_improvedDA',
        'UMRFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'UMRFormerTrainerV2_3_softDS4_radam',
        'UMRFormerTrainerV2_3_softDS4_radam_lowerLR',

        'UMRFormerTrainerV2_2_schedule',
        'UMRFormerTrainerV2_2_schedule2',
        'UMRFormerTrainerV2_2_clean',
        'UMRFormerTrainerV2_2_clean_improvedDA_newElDef',

        'UMRFormerTrainerV2_2_fixes', # running
        'UMRFormerTrainerV2_BN', # running
        'UMRFormerTrainerV2_noDeepSupervision', # running
        'UMRFormerTrainerV2_softDeepSupervision', # running
        'UMRFormerTrainerV2_noDataAugmentation', # running
        'UMRFormerTrainerV2_Loss_CE', # running
        'UMRFormerTrainerV2_Loss_CEGDL',
        'UMRFormerTrainerV2_Loss_Dice',
        'UMRFormerTrainerV2_Loss_DiceTopK10',
        'UMRFormerTrainerV2_Loss_TopK10',
        'UMRFormerTrainerV2_Adam', # running
        'UMRFormerTrainerV2_Adam_UMRFormerTrainerlr', # running
        'UMRFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'UMRFormerTrainerV2_SGD_lr1en1', # running
        'UMRFormerTrainerV2_SGD_lr1en3', # running
        'UMRFormerTrainerV2_fixedNonlin', # running
        'UMRFormerTrainerV2_GeLU', # running
        'UMRFormerTrainerV2_3ConvPerStage',
        'UMRFormerTrainerV2_NoNormalization',
        'UMRFormerTrainerV2_Adam_ReduceOnPlateau',
        'UMRFormerTrainerV2_fp16',
        'UMRFormerTrainerV2', # see overwrite_plans
        'UMRFormerTrainerV2_noMirroring',
        'UMRFormerTrainerV2_momentum09',
        'UMRFormerTrainerV2_momentum095',
        'UMRFormerTrainerV2_momentum098',
        'UMRFormerTrainerV2_warmup',
        'UMRFormerTrainerV2_Loss_Dice_LR1en3',
        'UMRFormerTrainerV2_NoNormalization_lr1en3',
        'UMRFormerTrainerV2_Loss_Dice_squared',
        'UMRFormerTrainerV2_newElDef',
        'UMRFormerTrainerV2_fp32',
        'UMRFormerTrainerV2_cycleAtEnd',
        'UMRFormerTrainerV2_reduceMomentumDuringTraining',
        'UMRFormerTrainerV2_graduallyTransitionFromCEToDice',
        'UMRFormerTrainerV2_insaneDA',
        'UMRFormerTrainerV2_independentScalePerAxis',
        'UMRFormerTrainerV2_Mish',
        'UMRFormerTrainerV2_Ranger_lr3en4',
        'UMRFormerTrainerV2_cycleAtEnd2',
        'UMRFormerTrainerV2_GN',
        'UMRFormerTrainerV2_DP',
        'UMRFormerTrainerV2_FRN',
        'UMRFormerTrainerV2_resample33',
        'UMRFormerTrainerV2_O2',
        'UMRFormerTrainerV2_ResencUNet',
        'UMRFormerTrainerV2_DA2',
        'UMRFormerTrainerV2_allConv3x3',
        'UMRFormerTrainerV2_ForceBD',
        'UMRFormerTrainerV2_ForceSD',
        'UMRFormerTrainerV2_ReLU',
        'UMRFormerTrainerV2_LReLU_slope_2en1',
        'UMRFormerTrainerV2_lReLU_convReLUIN',
        'UMRFormerTrainerV2_ReLU_biasInSegOutput',
        'UMRFormerTrainerV2_ReLU_convReLUIN',
        'UMRFormerTrainerV2_lReLU_biasInSegOutput',
        'UMRFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'UMRFormerTrainerV2_Loss_MCCnoBG',
        'UMRFormerTrainerV2_Loss_DicewithBG',
        # 'UMRFormerTrainerV2_Loss_Dice_LR1en3',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
        # 'UMRFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
