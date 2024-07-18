import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import squarepad_transform, targetpad_transform
from utils import extract_index_features, collate_fn, device
from data_utils import CIRDataset
from models import CIRPlus


def compute_fiq_val_metrics(relative_val_dataset: CIRDataset, model, index_features: torch.tensor,
                            index_names: List[str], device=torch.device('cuda')) -> Tuple[float, float]:
    # Generate predictions
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(model,
                                                                                         relative_val_dataset,
                                                                                         index_names, index_features,
                                                                                         device)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(model, relative_val_dataset: CIRDataset, index_names: List[str],
                                 index_features: torch.tensor, device=torch.device('cuda')):
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=16,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[-1]))

    # Initialize predicted features and target names
    target_names = []
    reference_names_all = []
    distance = None
    captions_all = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions = [model.txt_processors["eval"](caption) for caption in input_captions]
        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(input_captions) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = model.blip_model.inference(reference_image_features.to(device),
                                                        index_features[0].to(device),
                                                        input_captions)
            if distance is None:
                distance = batch_distance
            else:
                if len(batch_distance.shape) == 1:
                    distance = torch.vstack([distance, batch_distance.unsqueeze(0)])
                else:
                    distance = torch.vstack([distance, batch_distance])
            captions_all += input_captions

        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)
    return distance, target_names, reference_names_all, captions_all


def fashioniq_val_retrieval(dress_type: str, model, preprocess: callable):
    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRDataset('fiq', 'val', 'classic', preprocess, args.data_path, [dress_type],
                                     fiq_val_type=args.fiq_val_type)
    index_features, index_names = extract_index_features(classic_val_dataset, model)
    relative_val_dataset = CIRDataset('fiq', 'val', 'relative', preprocess, args.data_path, [dress_type])

    return compute_fiq_val_metrics(relative_val_dataset, model, index_features, index_names)


def compute_cirr_val_metrics(relative_val_dataset: CIRDataset, model, index_features: torch.tensor,
                             index_names: List[str], device=torch.device('cuda')) -> Tuple[
    float, float, float, float, float, float, float]:
    # Generate predictions
    pred_sim, reference_names, target_names, group_members, captions_all = \
        generate_cirr_val_predictions(model, relative_val_dataset, index_names, index_features, device)

    print("Compute CIRR validation metrics")
    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(model, relative_val_dataset: CIRDataset, index_names: List[str],
                                  index_features: torch.tensor, device=torch.device('cuda')):
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize predicted features, target_names, group_members and reference_names
    distance = []
    target_names = []
    group_members = []
    reference_names = []
    captions_all = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [model.txt_processors["eval"](caption) for caption in captions]
        # Compute the predicted features
        with torch.no_grad():
            # text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = model.blip_model.inference(reference_image_features.to(device),
                                                        index_features[0].to(device),
                                                        captions)
            distance.append(batch_distance)
            captions_all += captions

        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    distance = torch.vstack(distance)

    return distance, reference_names, target_names, group_members, captions_all


def cirr_val_retrieval(model, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRDataset('cirr', 'val', 'classic', preprocess, args.data_path)
    index_features, index_names = extract_index_features(classic_val_dataset, model)
    relative_val_dataset = CIRDataset('cirr', 'val', 'relative', preprocess, args.data_path)

    return compute_cirr_val_metrics(relative_val_dataset, model, index_features, index_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'cirr' or 'fiq'")
    parser.add_argument("--combiner", type=str, default='sum',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--blip_model_name", default="blip2_cir_align_prompt", type=str,
                        help="[blip2_cir_cat, blip2_cir_align_prompt]")
    parser.add_argument("--model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--model_path")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--fiq_val_type", default=0, type=int)
    parser.add_argument("--load_origin", action='store_true')
    parser.add_argument("--query_type", type=int, default=1)
    args = parser.parse_args()

    model = CIRPlus(args.blip_model_name)
    model.blip_model.eval()
    input_dim = model.input_dim

    if args.model_path:
        model.load_ckpt(args.model_path, args.load_origin)
    model.blip_model.query_type = args.query_type
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = model.preprocess

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(model, preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fiq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', model,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', model,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', model,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'cirr' or 'fiq")
