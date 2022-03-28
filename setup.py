from setuptools import setup, find_namespace_packages

setup(name='UMRFormer_Net',
      packages=find_namespace_packages(include=["UMRFormer_Net", "UMRFormer_Net.*"]),
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'UMRFormer_convert_decathlon_task = UMRFormer_Net.experiment_planning.UMRFormer_convert_decathlon_task:main',
              'UMRFormer_plan_and_preprocess = UMRFormer_Net.experiment_planning.UMRFormer_plan_and_preprocess:main',
              'UMRFormer_train = UMRFormer_Net.run.run_training:main',
              'UMRFormer_train_DP = UMRFormer_Net.run.run_training_DP:main',
              'UMRFormer_train_DDP = UMRFormer_Net.run.run_training_DDP:main',
              'UMRFormer_predict = UMRFormer_Net.inference.predict_simple:main',
              'UMRFormer_ensemble = UMRFormer_Net.inference.ensemble_predictions:main',
              'UMRFormer_find_best_configuration = UMRFormer_Net.evaluation.model_selection.figure_out_what_to_submit:main',
              'UMRFormer_print_available_pretrained_models = UMRFormer_Net.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'UMRFormer_print_pretrained_model_info = UMRFormer_Net.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'UMRFormer_download_pretrained_model = UMRFormer_Net.inference.pretrained_models.download_pretrained_model:download_by_name',
              'UMRFormer_download_pretrained_model_by_url = UMRFormer_Net.inference.pretrained_models.download_pretrained_model:download_by_url',
              'UMRFormer_determine_postprocessing = UMRFormer_Net.postprocessing.consolidate_postprocessing_simple:main',
              'UMRFormer_export_model_to_zip = UMRFormer_Net.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'UMRFormer_install_pretrained_model_from_zip = UMRFormer_Net.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'UMRFormer_change_trainer_class = UMRFormer_Net.inference.change_trainer:main',
              'UMRFormer_evaluate_folder = UMRFormer_Net.evaluation.evaluator:UMRFormer_evaluate_folder',
              'UMRFormer_plot_task_pngs = UMRFormer_Net.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      
      )
