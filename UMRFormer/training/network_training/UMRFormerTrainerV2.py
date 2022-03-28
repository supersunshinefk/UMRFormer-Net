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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from UMRFormer_Net.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from UMRFormer_Net.training.loss_functions.deep_supervision import MultipleOutputLoss2

from UMRFormer_Net.utilities.to_torch import maybe_to_torch, to_cuda
from UMRFormer_Net.network_architecture.generic_UNet import Generic_UNet

from UMRFormer_Net.network_architecture.UMRFormer_skipconnection import UMRFormer

from UMRFormer_Net.network_architecture.initialization import InitWeights_He
from UMRFormer_Net.network_architecture.neural_network import SegmentationNetwork
from UMRFormer_Net.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from UMRFormer_Net.training.dataloading.dataset_loading import unpack_dataset
from UMRFormer_Net.training.network_training.UMRFormerTrainer import UMRFormerTrainer
from UMRFormer_Net.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from UMRFormer_Net.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class UMRFormerTrainerV2(UMRFormerTrainer):
    """
    Info for Fabian: same as internal UMRFormerTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 0.002
        # self.initial_lr_SGD = 0.02
        # self.initial_lr_adam = 0.00003 # 0.00002~0.0002
        self.weight_decay = 1e-5
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            # self.loss = DC_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}, aggregate="sum", square_dice=False,
                                       # weight_ce=1, weight_dice=1,
                                       # log_dice=False, ignore_label=None)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            
            # ############################################################################
            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        self.network = UMRFormer(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)


        print('I am using the pre_train weight!!')  
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)

        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        # ####################deep supervision###########################
        target = target[0]
        output = output[0]
        
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        # print("target>>>>>>>>>>>>>>", target[0].shape, target[1].shape, target[2].shape, target[3].shape, len(target))
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            
        # ###################deep supervision##########################
        # target = np.array(target)[0]
        # target = torch.Tensor(target)
        
        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():

                output = self.network(data)
                del data
                # output = np.array(output)[0]
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))


            # splits[self.fold]['train'] = np.array(
            # ['pancreas_404', 'pancreas_201', 'pancreas_414', 'pancreas_399', 'pancreas_320', 'pancreas_024',
            #  'pancreas_315', 'pancreas_043', 'pancreas_087', 'pancreas_243', 'pancreas_297', 'pancreas_365',
            #  'pancreas_328', 'pancreas_412', 'pancreas_048', 'pancreas_264', 'pancreas_355', 'pancreas_149',
            #  'pancreas_343', 'pancreas_276', 'pancreas_405', 'pancreas_147', 'pancreas_122', 'pancreas_411',
            #  'pancreas_345', 'pancreas_291', 'pancreas_049', 'pancreas_213', 'pancreas_395', 'pancreas_253',
            #  'pancreas_269', 'pancreas_401', 'pancreas_321', 'pancreas_294', 'pancreas_158', 'pancreas_178',
            #  'pancreas_102', 'pancreas_358', 'pancreas_101', 'pancreas_186', 'pancreas_071', 'pancreas_086',
            #  'pancreas_091', 'pancreas_179', 'pancreas_402', 'pancreas_175', 'pancreas_021', 'pancreas_105',
            #  'pancreas_229', 'pancreas_183', 'pancreas_212', 'pancreas_217', 'pancreas_199', 'pancreas_170',
            #  'pancreas_215', 'pancreas_197', 'pancreas_330', 'pancreas_392', 'pancreas_295', 'pancreas_084',
            #  'pancreas_286', 'pancreas_061', 'pancreas_241', 'pancreas_095', 'pancreas_275', 'pancreas_356',
            #  'pancreas_312', 'pancreas_120', 'pancreas_278', 'pancreas_285', 'pancreas_042', 'pancreas_374',
            #  'pancreas_410', 'pancreas_262', 'pancreas_406', 'pancreas_310', 'pancreas_159', 'pancreas_182',
            #  'pancreas_173', 'pancreas_325', 'pancreas_114', 'pancreas_165', 'pancreas_344', 'pancreas_191',
            #  'pancreas_089', 'pancreas_304', 'pancreas_389', 'pancreas_300', 'pancreas_231', 'pancreas_052',
            #  'pancreas_058', 'pancreas_080', 'pancreas_055', 'pancreas_419', 'pancreas_103', 'pancreas_284',
            #  'pancreas_296', 'pancreas_083', 'pancreas_130', 'pancreas_290', 'pancreas_287', 'pancreas_099',
            #  'pancreas_329', 'pancreas_125', 'pancreas_279', 'pancreas_106', 'pancreas_018', 'pancreas_074',
            #  'pancreas_066', 'pancreas_267', 'pancreas_400', 'pancreas_235', 'pancreas_113', 'pancreas_346',
            #  'pancreas_413', 'pancreas_100', 'pancreas_360', 'pancreas_210', 'pancreas_126', 'pancreas_204',
            #  'pancreas_247', 'pancreas_308', 'pancreas_029', 'pancreas_244', 'pancreas_334', 'pancreas_056',
            #  'pancreas_070', 'pancreas_045', 'pancreas_380', 'pancreas_001', 'pancreas_378', 'pancreas_292',
            #  'pancreas_239', 'pancreas_004', 'pancreas_016', 'pancreas_298', 'pancreas_234', 'pancreas_006',
            #  'pancreas_309', 'pancreas_169', 'pancreas_302', 'pancreas_148', 'pancreas_035', 'pancreas_015',
            #  'pancreas_067', 'pancreas_094', 'pancreas_096', 'pancreas_301', 'pancreas_227', 'pancreas_226',
            #  'pancreas_256', 'pancreas_311', 'pancreas_367', 'pancreas_005', 'pancreas_342', 'pancreas_382',
            #  'pancreas_362', 'pancreas_145', 'pancreas_124', 'pancreas_421', 'pancreas_157', 'pancreas_258',
            #  'pancreas_092', 'pancreas_010', 'pancreas_289', 'pancreas_046', 'pancreas_119', 'pancreas_081',
            #  'pancreas_069', 'pancreas_259', 'pancreas_140', 'pancreas_180', 'pancreas_219', 'pancreas_117',
            #  'pancreas_266', 'pancreas_012', 'pancreas_274', 'pancreas_098', 'pancreas_075', 'pancreas_313',
            #  'pancreas_339', 'pancreas_050', 'pancreas_078', 'pancreas_293', 'pancreas_265', 'pancreas_181',
            #  'pancreas_379']
            # )
            # splits[self.fold]['val'] = np.array(
            # ['pancreas_127', 'pancreas_211', 'pancreas_242', 'pancreas_303', 'pancreas_077', 'pancreas_327',
            #  'pancreas_348', 'pancreas_111', 'pancreas_228', 'pancreas_137', 'pancreas_040', 'pancreas_305',
            #  'pancreas_318', 'pancreas_194', 'pancreas_222', 'pancreas_203', 'pancreas_398', 'pancreas_249',
            #  'pancreas_109', 'pancreas_088', 'pancreas_393', 'pancreas_255', 'pancreas_366', 'pancreas_160',
            #  'pancreas_207', 'pancreas_166', 'pancreas_280', 'pancreas_370', 'pancreas_316', 'pancreas_268',
            #  'pancreas_377', 'pancreas_416', 'pancreas_225', 'pancreas_167', 'pancreas_155', 'pancreas_323',
            #  'pancreas_387', 'pancreas_364', 'pancreas_200', 'pancreas_025', 'pancreas_415', 'pancreas_214',
            #  'pancreas_418', 'pancreas_135', 'pancreas_354', 'pancreas_032', 'pancreas_037']
            # )


            # ZJPancreas/ZJPancreasCancer
            # splits[self.fold]['train'] = np.array(
            # ['JBH_0_VP', 'CGQ_0_VP', 'GGH_0_VP', 'GCM_0_VP', 'gaowenxiang_0_VP', 'chenruixiu_0_VP', 'CSM_0_VP',
            #  'CCZ_0_PV', 'GSZ_0_VP', 'baijurong_0_PV', 'JYX_0_VP', 'huangjinfu_0_PV', 'FYY_0_VP', 'GZQ_0_VP',
            #  'GXM_0_VP', 'CEY_0_VP', 'chenyaoshun_0_PV', 'FCL_0_VP', 'CJQ_0_VP', 'caizhaofeng_0_PV', 'CRG_0_PV',
            #  'CLY_0_VP']
            # )
            # splits[self.fold]['val'] = np.array(
            # ['JYF_0_VP', 'fangshuixian_0_PV', 'duyuping_0_VP', 'GP1_0_VP', 'JXS_0_VP', 'ganyehua_0_PV', 'CHL_0_VP',
            #  'HCQ_0_VP', 'JXT_0_VP', 'GHP_0_VP', 'dingdaoju_0_PV', 'CP1_0_VP']
            # )
            splits[self.fold]['train'] = np.array(
            ['C3L_00395.nii.gz', 'C3L_03628.nii.gz', 'C3L_02109.nii.gz', 'C3L_02890.nii.gz', 'C3L_00625.nii.gz', 'C3N_00957.nii.gz',
             'C3L_03126.nii.gz', 'C3L_02115.nii.gz', 'C3L_03123.nii.gz', 'C3L_01689.nii.gz', 'C3L_01703.nii.gz',
             'C3L_00401.nii.gz', 'C3L_02118.nii.gz', 'C3N_01166.nii.gz', 'C3L_04475.nii.gz', 'C3L_00622.nii.gz',
             'C3L_03624.nii.gz', 'C3L_00599.nii.gz', 'C3L_04848.nii.gz', 'C3L_03632.nii.gz', 'C3N_00511.nii.gz',
             'C3L_02613.nii.gz', 'C3N_00302.nii.gz', 'C3N_01165.nii.gz'])
            splits[self.fold]['val'] = np.array(
             ['C3L_04479.nii.gz','C3L_02606.nii.gz', 'C3N_00249.nii.gz', 'C3L_02112.nii.gz', 'C3N_00512.nii.gz',
              'C3N_00303.nii.gz','C3L_03350.nii.gz', 'C3L_01702.nii.gz', 'C3N_00198.nii.gz'])

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        print("tr_keys", tr_keys)
        print("self.dataset", self.dataset)
        for i in tr_keys:

            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of UMRFormerTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
