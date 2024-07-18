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
    predicted_features, target_names, refer_names = generate_fiq_val_predictions(model, relative_val_dataset,
                                                                                 index_names, index_features, device)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    N = distances.shape[0]
    recall_at10 = 0
    recall_at50 = 0
    for i in range(N):
        sorted_index_name = sorted_index_names[i]
        sorted_index_name = sorted_index_name[sorted_index_name != refer_names[i]]
        if target_names[i] in sorted_index_name[:10]:
            recall_at10 += 1
            recall_at50 += 1
        elif target_names[i] in sorted_index_name[:50]:
            recall_at50 += 1

    # Compute the metrics
    recall_at10 = (recall_at10 / N) * 100
    recall_at50 = (recall_at50 / N) * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(model, relative_val_dataset: CIRDataset, index_names: List[str],
                                 index_features: torch.tensor, device=torch.device('cuda')) -> \
        Tuple[torch.tensor, List[str], List[str]]:
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, model.output_dim)).to(device, non_blocking=True)
    target_names = []
    refer_names = []
    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        # input_captions = [
        #     f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
        #     i in range(0, len(flattened_captions), 2)]
        input_captions = [
            f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        # text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)
        text_inputs = input_captions
        # Compute the predicted features
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = model.combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        refer_names.extend(reference_names)

    return predicted_features, target_names, refer_names


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
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(model, relative_val_dataset, index_names, index_features, device=device)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    if len(index_features.shape) > 2:
        index_features = torch.mean(index_features, dim=1)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
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
                                  index_features: torch.tensor, device=torch.device('cuda')) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, model.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        new_batch_reference_names = []
        try:
            for j, batch_reference_name in enumerate(batch_reference_names):
                batch_reference_names[j] = int(batch_reference_name)
                new_batch_reference_names.append(int(batch_reference_name))
            batch_reference_names = new_batch_reference_names
        except:
            pass
        # Compute the predicted features
        with torch.no_grad():
            text_features = model.encode_text(captions)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = model.combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


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
    parser.add_argument("--clip-model-name", default="ViT-L/14", type=str,
                        help="CLIP model to use, e.g 'RN50', 'RN50x4','ViT-B/16'")
    parser.add_argument("--model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--model_path")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--fiq_val_type", default=0, type=int)
    parser.add_argument("--load_origin", action='store_true')
    args = parser.parse_args()

    model = CIRPlus(args.clip_model_name, combiner=args.combiner)
    model.eval()
    input_dim = model.input_dim
    feature_dim = model.output_dim

    if args.model_path:
        model.load_ckpt(args.model_path, args.load_origin)
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
