import os.path
from datetime import datetime
import json
import random
from argparse import ArgumentParser
from pathlib import Path

from torch.cuda.amp import autocast

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import CIRDataset
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics
from utils import collate_fn, extract_index_features, save_model, RunningAverage
from statistics import mean, geometric_mean, harmonic_mean
from collections import OrderedDict
from models import CIRPlus

base_path = Path(__file__).absolute().parents[1].absolute()


def main():
    if args.debug:
        training_path = Path(base_path / f"models/debug")
        training_path.mkdir(exist_ok=True)
    elif args.output_path:
        training_path = Path(args.output_path)
        training_path.mkdir(exist_ok=True, parents=True)
    else:
        training_start = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        training_path: Path = Path(
            base_path / f"models/{args.dataset}_{args.clip_model_name}_{training_start}")
        training_path.mkdir(exist_ok=True, parents=True)
    print("training_path:", training_path)

    model = CIRPlus(args.blip_model_name, tau=args.tau, transform=args.transform,
                    device=device, plus=args.plus).to(device)
    if args.model_path:
        model.load_ckpt(args.model_path, True)
    preprocess = model.preprocess
    # Define the validation datasets
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    relative_val_dataset = None
    classic_val_dataset = None
    index_features_list = []
    index_features_p_list = []
    index_names_list = []
    val_index_features, val_index_names, val_index_features_p = None, None, None
    if args.dataset == 'cirr':
        relative_val_dataset = CIRDataset(args.dataset, 'val', 'relative', preprocess, args.data_path,
                                          args.dress_types)
        classic_val_dataset = CIRDataset(args.dataset, 'val', 'classic', preprocess, args.data_path,
                                         args.dress_types)
        val_index_features, val_index_features_p, val_index_names = extract_index_features(classic_val_dataset,
                                                                                           model.blip,
                                                                                           device=device)
    elif args.dataset == 'fiq':
        for idx, dress_type in enumerate(args.dress_types):
            idx_to_dress_mapping[idx] = dress_type
            relative_val_dataset = CIRDataset('fiq', 'val', 'relative', preprocess, args.data_path, [dress_type])
            relative_val_datasets.append(relative_val_dataset)
            classic_val_dataset = CIRDataset('fiq', 'val', 'classic', preprocess, args.data_path, [dress_type],
                                             fiq_val_type=0)
            classic_val_datasets.append(classic_val_dataset)
            index_features_and_names = extract_index_features(classic_val_dataset, model.blip,
                                                              device=device)
            index_features_list.append(index_features_and_names[0])
            index_features_p_list.append(index_features_and_names[1])
            index_names_list.append(index_features_and_names[2])
    relative_train_dataset = CIRDataset(args.dataset, 'train', 'relative', preprocess, args.data_path,
                                        args.dress_types, plus=args.plus, llmcap=args.llmcap)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=args.batch_size,
                                       num_workers=4, pin_memory=False, collate_fn=collate_fn, shuffle=True,
                                       drop_last=True)
    param_groups = [
        {
            'params': filter(lambda p: p.requires_grad, model.parameters()),
            'lr': args.learning_rate,
            'betas': (0.9, 0.999),
            'eps': 1e-7,
        }
    ]
    optimizer = optim.AdamW(
        param_groups
    )
    scaler = torch.cuda.amp.GradScaler()

    best_score = 0
    cur_score = 0
    loss_avg = RunningAverage()
    if not args.wo_bank:
        # calculate or load bank features
        if not args.bank_path:
            bank_path = os.path.join(args.output_path, f"{args.dataset}_bank.pth")
        else:
            bank_path = args.bank_path
        model.extract_bank_features(relative_train_dataset, device, bank_path, args.reload_bank)
        # calculate or load bank features for additional data
        refer_bank_path = bank_path.replace("bank", "refer_bank")
        model.extract_refer_bank_features(relative_train_dataset, device, refer_bank_path, args.reload_bank)
        if args.plus:
            model.load_refer_bank(refer_bank_path)
        relative_train_dataset.use_bank = True
    print('Training loop started')
    for epoch in range(args.num_epochs):
        model.blip.eval()
        with tqdm(total=len(relative_train_loader)) as t:
            for idx, batch_data in enumerate(relative_train_loader):
                captions, indexs, target_indexs, target_index_all, reference_index_all = batch_data
                optimizer.zero_grad()
                with autocast():
                    loss_dict = model.forward(captions, indexs, target_index_all,
                                              reference_index_all)
                loss = loss_dict['bank_loss']
                loss_avg.update(loss.item())
                loss_dict['loss'] = loss_avg()
                for loss_name in loss_dict:
                    loss_dict[loss_name] = '{:05.3f}'.format(loss_dict[loss_name])
                t.set_postfix(OrderedDict(loss_dict))
                t.update()
                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        print(f"Epoch [{epoch}] Loss: {loss_avg()}")

        if epoch % args.validation_frequency == 0:
            if args.dataset == 'cirr':
                results = compute_cirr_val_metrics(relative_val_dataset, model.blip, val_index_features,
                                                   val_index_features_p,
                                                   val_index_names)
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'recall_mean': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }
                print(json.dumps(results_dict, indent=4))
                cur_score = results_dict['recall_mean']
                if args.nni:
                    nni.report_intermediate_result({'default': best_score, "recall_mean": results_dict['recall_mean']})
            elif args.dataset == 'fiq':
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                          idx_to_dress_mapping):
                    index_features, index_features_p, index_names = index_features_list[idx], index_features_p_list[
                        idx], index_names_list[idx]

                    recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, model.blip,
                                                                       index_features, index_features_p, index_names)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })
                cur_score = results_dict['average_recall']
                print(json.dumps(results_dict, indent=4))
                if args.nni:
                    nni.report_intermediate_result(
                        {'default': best_score, "average_recall_at10": results_dict['average_recall_at10']})

            if cur_score > best_score:
                best_score = cur_score
                print("current best:", best_score)
                if not args.nni:
                    save_model('best', epoch, model, training_path)
    if args.nni:
        nni.report_final_result({'default': best_score})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['fiq', 'cirr'],
                        help="should be either 'cirr' or 'fiq'")
    parser.add_argument("--num-epochs", default=5, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="/root_path/models/blip/model_base.pth", type=str,
                        help="CLIP model to use, e.g 'RN50', 'RN50x4','ViT-B/16")
    parser.add_argument("--learning-rate", default=5e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--output_path", default='', help='set the output path of models and log')
    parser.add_argument("--tau", default=0.03, type=float)
    parser.add_argument("--dress_types", default='dress,shirt,toptee')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", default='')
    parser.add_argument("--use_bank", action='store_true')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--reload_bank", action='store_true')
    parser.add_argument("--device", default='0')
    parser.add_argument("--bank_path", default='')
    parser.add_argument("--nni", action='store_true')
    parser.add_argument("--plus", action='store_true', help='whether use additional data')
    parser.add_argument("--llmcap", action='store_true', help='whether use llm caption')

    args = parser.parse_args()
    if args.data_path == '':
        if args.dataset == 'fiq':
            args.data_path = 'fashionIQ_dataset'
        else:
            args.data_path = 'cirr_dataset'
    device = torch.device(f'cuda:{args.device}')
    args.dress_types = args.dress_types.split(',')
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if args.nni:
        import nni
        from nni.utils import merge_parameter

        nni_args = nni.get_next_parameter()
        args = merge_parameter(args, nni_args)
    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    main()
