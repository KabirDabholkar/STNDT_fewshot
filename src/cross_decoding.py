from cross_decoder import CrossDecoder, LatentAnalysisInterface
import sys
module_path = '/home/kabird/STNDT_fewshot' #os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.analyze_utils import init_by_ckpt
from src.dataset import SpikesDataset, DATASET_MODES
from src.validate_nlb_fewshot import DATA_DIR
import os.path as osp
import h5py
from nlb_tools.make_tensors import h5_to_dict
from pathlib import Path
import torch
import numpy as np
import argparse

class STNDT_Analysis(LatentAnalysisInterface):
    def __init__(self,checkpoint_path,variant):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        # train_path = osp.join(DATA_DIR, f"{variant}_train.h5")
        # with h5py.File(train_path, 'r') as h5file:
        #     self.train_dict = h5_to_dict(h5file)
        
        # val_path = osp.join(DATA_DIR, f"{variant}_val.h5")
        # with h5py.File(val_path, 'r') as h5file:
        #     self.val_dict = h5_to_dict(h5file)
        
    def get_latents(self,phase='val'):
        target_path = osp.join(DATA_DIR, f"{self.variant}_target.h5")
        with h5py.File(target_path, 'r') as h5file:
            target_dict = h5_to_dict(h5file)[self.variant]
        
        train_path = osp.join(DATA_DIR, f"{self.variant}_train.h5")
        with h5py.File(train_path, 'r') as h5file:
            train_dict = h5_to_dict(h5file)
        
        val_path = osp.join(DATA_DIR, f"{self.variant}_val.h5")
        with h5py.File(val_path, 'r') as h5file:
            val_dict = h5_to_dict(h5file)


        self.runner, self.spikes, self.rates, self.heldout_spikes, self.forward_spikes = init_by_ckpt(self.checkpoint_path, mode=DATASET_MODES.val if phase == 'val' else DATASET_MODES.train)
        # Get rates for the specified phase
        rates, *_ = self.runner.get_rates(
            checkpoint_path=self.checkpoint_path,
            save_path=None,
            mode=DATASET_MODES.val if phase == 'val' else DATASET_MODES.train
        )
        # print({k:data_dict[k].shape for k in data_dict.keys()})
        # Get spikes from the runner
        
        spikes_heldin = torch.tensor(val_dict[f'eval_spikes_heldin'],device=self.runner.device)
        heldout_spikes = torch.tensor(target_dict[f'eval_spikes_heldout'],device=self.runner.device)
        forward_heldin_spikes = torch.tensor(target_dict[f'eval_spikes_heldin_forward'],device=self.runner.device)
        forward_heldout_spikes = torch.tensor(target_dict[f'eval_spikes_heldout_forward'],device=self.runner.device)
        forward_spikes = torch.cat([forward_heldin_spikes,forward_heldout_spikes],-1)

        
        # Prepare input spikes
        eval_trials = spikes_heldin.shape[0]
        spikes = spikes_heldin
        spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)[:eval_trials]], -1)
        spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)[:eval_trials]], 1)
        
        # Process in batches to avoid memory issues
        max_batch_size = 8
        batch_size = max_batch_size
        success = False
        
        while not success and batch_size > 1:
            try:
                encoder_output = [self.runner.model(
                    spike_split,
                    spike_split,
                    contrast_src1=None,
                    contrast_src2=None,
                    val_phase=True,
                    passthrough=True,
                    return_outputs=True,
                    return_weights=True,
                    return_encoder_output=True,
                )[6].to('cpu') for spike_split in torch.split(spikes, spikes.shape[0]//batch_size)]
                success = True
                encoder_output = [thing.to('cpu') for thing in encoder_output]
            except MemoryError:
                print(f"MemoryError: Reducing batch size to {batch_size // 2}")
                batch_size //= 2
        
        # Concatenate outputs and get final latents
        encoder_output = torch.cat(encoder_output, dim=1)
        last_layer_outputs = encoder_output.permute([1,0,2])
        last_layer_outputs = last_layer_outputs[:,:spikes_heldin.shape[1]]
        eval_factors = last_layer_outputs.detach().cpu().numpy()
        
        # Clean up GPU memory
        del spikes_heldin
        del heldout_spikes
        del forward_heldin_spikes 
        del forward_heldout_spikes
        del forward_spikes
        del spikes
        del encoder_output
        del last_layer_outputs
        del self.runner
        del self.spikes
        del self.rates
        del self.heldout_spikes
        del self.forward_spikes
        torch.cuda.empty_cache()
        
        return eval_factors

    def get_trial_lengths(self,phase='val'):
        return None

    def run_name(self):
        return self.checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross decoding analysis for STNDT models')
    parser.add_argument('--variant', type=str, default='mc_rtt_20', help='Dataset variant to use (default: mc_rtt_20)')
    args = parser.parse_args()

    # Test STNDT_Analysis functionality
    variant = args.variant
    to_save_dir = '/home/kabird/STNDT_fewshot/ray_results/'

    num_models = None

    # Get all checkpoint files
    import glob
    import os
    import pandas as pd

    checkpoint_dir = f'/home/kabird/STNDT_fewshot/ray_results/{variant}_lite/{variant}_lite/'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '**/*.cobps.pth'), recursive=True)

    if num_models is not None:
        checkpoint_files = checkpoint_files[:num_models]

    # Create analysis objects for each checkpoint
    analyses = []
    CD = CrossDecoder(save_dir=Path(to_save_dir)/f'{variant}_lite')
    for ckpt_file in checkpoint_files:
        analysis = STNDT_Analysis(
            checkpoint_path=ckpt_file,
            variant=variant
        )
        # analyses.append(analysis)
        CD.load_analysis(analysis)


    r2_matrix, group_matrix = CD.compute_pairwise_latent_r2(max_trials=500, parallel=True, n_jobs=30)
    
    # Create DataFrame from r2_matrix
    df = pd.DataFrame(1-r2_matrix)  # Using 1-r2 to match previous print statements
    
    # Save to CSV
    
    output_path = os.path.join(to_save_dir, f'{variant}_lite_r2_matrix_2.csv')
    df.to_csv(output_path, index=True)
    print(f"Saved R2 matrix to: {output_path}")
    

    CD.save_decoding_matrices(r2_matrix, group_matrix,save_json_only=False)

    # # eval_factors = analysis.get_latents()

    # # # Test getting trial factors
    # # try:
    # #     eval_factors = analysis.get_trial_factors()
    # #     print("Successfully extracted trial factors with shape:", eval_factors.shape)
    # # except Exception as e:
    # #     print("Error getting trial factors:", str(e))

    # # # Test getting trial lengths
    # # try:
    # #     trial_lengths = analysis.get_trial_lengths()
    # #     print("Successfully got trial lengths:", trial_lengths)
    # # except Exception as e:
    # #     print("Error getting trial lengths:", str(e))

