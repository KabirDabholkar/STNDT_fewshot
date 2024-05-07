# Author: Trung Le
# Ensemble prediction rates from tuned checkpoints

import os
import os.path as osp
from pathlib import Path
import sys
import json
module_path = '/home/kabird/STNDT_fewshot' #os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import time
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils import data
import argparse
from typing import Any
from functools import partial

from nlb_tools.fewshot_utils import result_dict_to_pandas
from nlb_tools.evaluation import evaluate
from nlb_tools.load_and_save_latents import run_nlb_evaluation_protocol
from nlb_tools.make_tensors import save_to_h5, make_train_input_tensors, h5_to_dict
from src.run import prepare_config
from src.runner import Runner
from src.dataset import SpikesDataset, DATASET_MODES
from src.mask import UNMASKED_LABEL
from src.analyze_utils import init_by_ckpt
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from nlb_tools.torch_glm import fit_poisson as fit_poisson_pl
from src.utils import print_gpu_memory_usage, sizeof_fmt

data_path_base = '/home/kabird/datasets'
DATA_DIR = Path(f"{data_path_base}/")
ENSEMBLE_RESULTS_DIR = Path("./ensemble_results/")
RAY_RESULTS_DIR = Path("./ray_results/")


def get_parser():
    r"""
    Gets parsed arguments from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset-name",
        choices=["mc_maze", "mc_maze_large", "mc_maze_medium", "mc_maze_small", "mc_rtt", "area2_bump", "dmfc_rsg"],
        help="name of dataset to perform ensembling",
    )
    return parser

def fit_poisson(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    train_pred = []
    eval_pred = []
    for chan in range(train_out.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_in, train_out[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_in, train_out[:, chan])
        train_pred.append(pr.predict(train_factors_s))
        eval_pred.append(pr.predict(eval_factors_s))
    train_rates_s = np.vstack(train_pred).T
    eval_rates_s = np.vstack(eval_pred).T
    return train_rates_s, eval_rates_s

def fit_poisson_parallel(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""

    pr = MultiOutputRegressor(
        estimator=PoissonRegressor(alpha=alpha, max_iter=500),
        n_jobs=-1
    )
    

    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    
    pr.fit(train_in, train_out)
    train_rates_s = pr.predict(train_factors_s)
    eval_rates_s = pr.predict(eval_factors_s)
    return train_rates_s, eval_rates_s

def fit_dummy_zeros(train_factors_s, eval_factors_s, train_spikes_s):
    """Fit Poisson GLM from factors to spikes and return rate predictions"""
    eval_N, _ = eval_factors_s.shape
    _,output_channels = train_spikes_s.shape
    train_rates_s = np.zeros_like(train_spikes_s)
    eval_rates_s = np.zeros((eval_N,output_channels))
    return train_rates_s, eval_rates_s


def main():
    r"""
    Ensembles rate predictions from model checkpoints returned by hyperparameter search 
    Saves ensembled rate predictions to ENSEMBLE_RESULTS_DIR in h5 format
    """
    parser = get_parser()
    args = parser.parse_args()
    variant = vars(args)['dataset-name']
    # variant = "mc_maze"
    # variant = "area2_bump"
    # variant = "dmfc_rsg"
    # variant = "mc_rtt"

    target_path = osp.join(DATA_DIR, f"{variant}_target.h5")

    with h5py.File(target_path, 'r') as h5file:
        target_dict = h5_to_dict(h5file)

    ray_results_dir = osp.join(RAY_RESULTS_DIR, f"{variant}_lite/{variant}_lite/")
    
    train_path = osp.join(DATA_DIR, f"{variant}_train.h5")
    with h5py.File(train_path, 'r') as h5file:
        train_dict = h5_to_dict(h5file)
    
    val_path = osp.join(DATA_DIR, f"{variant}_val.h5")
    with h5py.File(val_path, 'r') as h5file:
        val_dict = h5_to_dict(h5file)

    train_path_json = osp.join(DATA_DIR, f"{variant}_train.json")
    with open(train_path_json, 'r') as file:
        few_shot_metadata = json.load(file)
    

    eval_report = []
    
    for root, dirs, files in os.walk(ray_results_dir):
        for file in files:
            if file.endswith("pth"):
                ckpt_path = os.path.join(root, file)
                print('evaluating', ckpt_path)
                

                ### get runner
                runner, spikes, rates, heldout_spikes, forward_spikes = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                print('Before running runner\n','Total:',sizeof_fmt(t),
                              '\nReserved:',sizeof_fmt(r),
                              '\nAllocated:',sizeof_fmt(a),
                              '\nFree:',sizeof_fmt(r-a)
                              )
                print_gpu_memory_usage(vars())
                eval_rates, *_ = runner.get_rates(
                    checkpoint_path=ckpt_path,
                    save_path = None,
                    mode = DATASET_MODES.val
                )
                train_rates, *_ = runner.get_rates(
                    checkpoint_path=ckpt_path,
                    save_path = None,
                    mode = DATASET_MODES.train
                )

                # print('model',runner.model.src_decoder)

                eval_rates = eval_rates.cpu()
                train_rates = train_rates.cpu()

                # fig,axs = plt.subplots(1,1)
                # im = axs.imshow(eval_rates[0].T.detach().cpu(),interpolation='none')
                # fig.colorbar(im,ax=axs)
                # fig.savefig('eval_rates_full.png',dpi=250)

                eval_rates, eval_rates_forward = torch.split(eval_rates, [spikes.size(1), eval_rates.size(1) - spikes.size(1)], 1)
                eval_rates_heldin_forward, eval_rates_heldout_forward = torch.split(eval_rates_forward, [spikes.size(-1), heldout_spikes.size(-1)], -1)
                train_rates, _ = torch.split(train_rates, [spikes.size(1), train_rates.size(1) - spikes.size(1)], 1)
                eval_rates_heldin, eval_rates_heldout = torch.split(eval_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)
                train_rates_heldin, train_rates_heldout = torch.split(train_rates, [spikes.size(-1), heldout_spikes.size(-1)], -1)

                output_dict = {
                    variant: {
                        'train_rates_heldin': train_rates_heldin.cpu().numpy(),
                        'train_rates_heldout': train_rates_heldout.cpu().numpy(),
                        'eval_rates_heldin': eval_rates_heldin.cpu().numpy(),
                        'eval_rates_heldout': eval_rates_heldout.cpu().numpy(),
                        'eval_rates_heldin_forward': eval_rates_heldin_forward.cpu().numpy(),
                        'eval_rates_heldout_forward': eval_rates_heldout_forward.cpu().numpy()
                    }
                }

                heldout_spikes = heldout_spikes.to(runner.device)
                forward_spikes = forward_spikes.to(runner.device)
                
                ### get eval factors
                eval_spikes_heldin = torch.tensor(val_dict['eval_spikes_heldin'],dtype=spikes.dtype,device=runner.device)
                eval_trials = eval_spikes_heldin.shape[0]
                spikes = eval_spikes_heldin
                spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes)[:eval_trials]], -1)
                spikes = torch.cat([spikes, torch.zeros_like(forward_spikes)[:eval_trials]], 1)
                print('spikes shape',spikes.shape)
                max_batch_size = 8  # Define the maximum batch size that fits in memory
                batch_size = max_batch_size
                success = False
                
                while not success and batch_size > 1:
                    try:
                        encoder_output = [runner.model(
                            spike_split,
                            spike_split,
                            contrast_src1=None,
                            contrast_src2=None,
                            val_phase=True,
                            passthrough=True,
                            return_outputs=True,
                            return_weights=True,
                            return_encoder_output = True,
                        )[6].to('cpu') for spike_split in torch.split(spikes,spikes.shape[0]//batch_size)]
                        success = True
                        encoder_output = [thing.to('cpu') for thing in encoder_output]
                        
                    except MemoryError:
                        print(f"MemoryError: Reducing batch size to {batch_size // 2}")
                        batch_size //= 2  # Halve the batch size

                
            
                # Concatenate the outputs
                # loss = torch.cat(loss, dim=0)
                # print([e.shape for e in encoder_output])
                encoder_output = torch.cat(encoder_output, dim=1)
                
                    
                
                # print('encoder_output shape',encoder_output.shape)
                last_layer_outputs = encoder_output.permute([1,0,2])
                # last_layer_outputs = batch_layer_outputs[...,-1].detach()
                last_layer_outputs = last_layer_outputs[:,:eval_spikes_heldin.shape[1]]
                eval_factors = np.array(last_layer_outputs.cpu())
                eval_factors_s = eval_factors.reshape(-1,last_layer_outputs.shape[-1])


                # print(runner.model)
                # k = 128
                fewshot_output_dict = {}

                k_range = 2**np.arange(4,11)[:].astype(int)
                # k_range = [int(k) for k in k_range]
                for k in k_range:
                    heldout_spikes_fewshot = heldout_spikes[:k]
                    forward_spikes_fewshot = forward_spikes[:k]

                    
                    for shot_id in few_shot_metadata[f'{k}shot_ids'][:]:
                        fewshot_train_spikes_heldin = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldin']
                        
                        spikes = torch.tensor(fewshot_train_spikes_heldin,dtype=heldout_spikes.dtype).to(runner.device)
                        
                        # Do NOT provide privileged eval info
                        spikes_full = torch.cat([spikes.clone(), heldout_spikes_fewshot], -1)
                        spikes_full = torch.cat([spikes_full, forward_spikes_fewshot], 1)
                        spikes = torch.cat([spikes, torch.zeros_like(heldout_spikes_fewshot)], -1)
                        spikes = torch.cat([spikes, torch.zeros_like(forward_spikes_fewshot)], 1)
                        max_batch_size = 4  # Define the maximum batch size that fits in memory
                        batch_size = max_batch_size
                        success = False
                        
                        while not success and batch_size > 1:
                            try:
                                encoder_output_fewshot = [runner.model(
                                    spike_split,
                                    spike_split,
                                    contrast_src1=None,
                                    contrast_src2=None,
                                    val_phase=True,
                                    passthrough=True,
                                    return_outputs=True,
                                    return_weights=True,
                                    return_encoder_output = True,
                                )[6].to('cpu') for spike_split in torch.split(spikes,split_size_or_sections=int(np.maximum(spikes.shape[0]//batch_size,1)))]
                                success = True
                                encoder_output_fewshot = [thing.to('cpu') for thing in encoder_output_fewshot]
                                
                            except MemoryError:
                                print(f"MemoryError: Reducing batch size to {batch_size // 2}")
                                batch_size //= 2  # Halve the batch size

                        
                    
                        # Concatenate the outputs
                        # loss = torch.cat(loss, dim=0)
                        # print([e.shape for e in encoder_output])
                        encoder_output_fewshot = torch.cat(encoder_output_fewshot, dim=1)
                        
                        # last_layer_outputs = batch_layer_outputs[...,-1].detach()
                        last_layer_outputs = encoder_output_fewshot.permute([1,0,2]).detach().cpu()
                        last_layer_outputs = last_layer_outputs[:,:fewshot_train_spikes_heldin.shape[1]]
                        fewshot_train_factors = np.array(last_layer_outputs.cpu())
                        fewshot_train_factors_s = fewshot_train_factors.reshape(-1,last_layer_outputs.shape[-1])
                        
                        fewshot_train_spikes_heldout = train_dict[f'{k}shot_id{shot_id}_train_spikes_heldout']
                        fewshot_train_spikes_heldout_s = fewshot_train_spikes_heldout.reshape(-1,fewshot_train_spikes_heldout.shape[-1])
                        # print(fewshot_train_factors_s.shape,eval_factors_s.shape,fewshot_train_spikes_heldout_s.shape)
                        # fewshot_train_rates_s, eval_rates_s = fit_dummy_zeros(fewshot_train_factors_s,eval_factors_s,fewshot_train_spikes_heldout_s)
                        # fewshot_train_rates_s, eval_rates_s = fit_poisson(fewshot_train_factors_s,eval_factors_s,fewshot_train_spikes_heldout_s,alpha=0.0)
                        fewshot_train_rates_s, eval_rates_s = fit_poisson_parallel(fewshot_train_factors_s,eval_factors_s,fewshot_train_spikes_heldout_s,alpha=0.1)
                        fewshot_code_name = 'sklearn_parallel'
                        # coefficients = runner.model.src_decoder[0].weight.data.cpu().numpy()[-fewshot_train_spikes_heldout.shape[-1]:,:]
                        # intercepts = runner.model.src_decoder[0].bias.data.cpu().numpy()[-fewshot_train_spikes_heldout.shape[-1]:]
                        # t = torch.cuda.get_device_properties(0).total_memory
                        # r = torch.cuda.memory_reserved(0)
                        # a = torch.cuda.memory_allocated(0)
                        # print('Before poisson fit \n','Total:',sizeof_fmt(t),
                        #       '\nReserved:',sizeof_fmt(r),
                        #       '\nAllocated:',sizeof_fmt(a),
                        #       '\nFree:',sizeof_fmt(r-a)
                        #       )
                        # fewshot_train_rates_s, eval_rates_s = fit_poisson_pl(fewshot_train_factors_s,eval_factors_s,fewshot_train_spikes_heldout_s) #,coefficients=coefficients,intercepts=intercepts,train=False)
                        # fewshot_code_name = 'torch_lightning'
                        # t = torch.cuda.get_device_properties(0).total_memory
                        # r = torch.cuda.memory_reserved(0)
                        # a = torch.cuda.memory_allocated(0)
                        # print('After poisson fit \n','Total:',sizeof_fmt(t),
                        #       '\nReserved:',sizeof_fmt(r),
                        #       '\nAllocated:',sizeof_fmt(a),
                        #       '\nFree:',sizeof_fmt(r-a)
                        #       )
                        eval_rates = eval_rates_s.reshape(*val_dict['eval_spikes_heldout'].shape[:2],-1)
                        
                        # fig,axs = plt.subplots(1,1)
                        # im = axs.imshow(eval_rates[0].T,interpolation='none')
                        # fig.colorbar(im,ax=axs)
                        # fig.savefig('eval_rates_fewshot.png',dpi=250)
                        # fig,axs = plt.subplots(1,2)
                        # im = axs[0].imshow(eval_rates[0].T,interpolation='none')
                        # im = axs[1].imshow(np.array(eval_rates_heldout.detach().cpu())[0].T,interpolation='none')
                        # # im = axs[1].imshow(val_dict['eval_spikes_heldout'][0].T,interpolation='none')
                        # fig.colorbar(im,ax=axs[1])
                        # fig.savefig('eval_rates.png',dpi=250)
                        fewshot_output_dict [f'{k}shot_id{shot_id}_eval_rates_heldout'] = eval_rates #np.array(eval_rates_heldout.detach().cpu()) 
                        
                        del(encoder_output_fewshot,last_layer_outputs,spikes,spikes_full)

                        t = torch.cuda.get_device_properties(0).total_memory
                        r = torch.cuda.memory_reserved(0)
                        a = torch.cuda.memory_allocated(0)
                        print('Total:',sizeof_fmt(t),
                              '\nReserved:',sizeof_fmt(r),
                              '\nAllocated:',sizeof_fmt(a),
                              '\nFree:',sizeof_fmt(r-a)
                              )
                        

                del(runner)
                    
                output_dict[variant] = {
                    **output_dict[variant],
                    **fewshot_output_dict
                }
    
                result_data = evaluate(target_dict, output_dict)
                print('result_dict',result_data)
                df = result_dict_to_pandas(
                    result_data,
                    fewshot_learner=fewshot_code_name,
                    path=ckpt_path
                )
                
                eval_report.append(df)
                
                if True: #len(eval_report)==20:
                    D = pd.concat(eval_report,axis=0).reset_index()
                    D.to_csv('results5.csv')
                    
    D = pd.concat(eval_report,axis=0).reset_index()
    D.to_csv('results5.csv')
    return
                
    
                
def run_model_on_numpy(
        model: Any,
        spikes_heldin: np.ndarray,
        spikes_full_shape: tuple,
        batch_size: int = 6
        ):
    n_trials,small_t,small_n = spikes_heldin.shape
    
    spikes = np.zeros((n_trials,*spikes_full_shape[1:]))
    spikes[:,:small_t,:small_n] = spikes_heldin
    spikes = torch.tensor(spikes)
    print('spikes shape',spikes.shape)
    all_outputs = [model(
                            spike_split.to(model.device),
                            spike_split.to(model.device),
                            contrast_src1=None,
                            contrast_src2=None,
                            val_phase=True,
                            passthrough=True,
                            return_outputs=True,
                            return_weights=True,
                            return_encoder_output = True,
                    ) for spike_split in torch.split(spikes,spikes.shape[0]//batch_size)]
    encoder_output = [batch[6].detach().cpu().numpy().swapaxes(0,1) for batch in all_outputs]
    # print([thing.shape for thing in encoder_output])
    encoder_output = np.concatenate(encoder_output,axis=0)
    rate_predictions = [batch[3].detach().cpu().numpy() for batch in all_outputs]
    rate_predictions = np.concatenate(rate_predictions,axis=0)
    rate_predictions = np.exp(rate_predictions)
    return rate_predictions, encoder_output
    
def main2():
    variant = 'mc_maze'
    ray_results_dir = osp.join(RAY_RESULTS_DIR, f"{variant}_lite/{variant}_lite/")
    
    for root, dirs, files in os.walk(ray_results_dir):
        for file in files:
            if file.endswith("pth"):
                ckpt_path = os.path.join(root, file)
                print('loading checkpoint', ckpt_path)
                runner, _, _, _, _ = init_by_ckpt(ckpt_path, mode=DATASET_MODES.val)

                results_df,results_dict,latents_dict = run_nlb_evaluation_protocol(model=runner.model,run_model_on_numpy_pre=run_model_on_numpy,variant='mc_maze')
                savepath = ckpt_path.replace('.pth','_latents.h5')
                print('Saving latents to',savepath)
                save_to_h5(
                    latents_dict,
                    save_path=savepath, #ckpt_path.split('.')[0]+'_latents.h5',
                    overwrite=True
                )
                # print(df)
                
                
    #             pred,latents = run_model_on_numpy_partial(runner.model,spikes_heldin=eval_spikes_heldin)
    #             print(pred.shape, latents.shape)
    #             return 

if __name__=="__main__":
    # main()
    main2()