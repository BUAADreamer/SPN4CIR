import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRDataset, targetpad_transform, squarepad_transform, base_path
from utils import device, extract_index_features
from models import CIRPlus


def generate_cirr_test_submissions(file_name: str, model, preprocess, txt_processors):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
   :param file_name: file_name of the submission
   :param clip_model: CLIP model
   :param preprocess: preprocess pipeline
   """
    # Define the dataset and extract index features
    classic_test_dataset = CIRDataset('cirr', 'test1', 'classic', preprocess, args.data_path)
    index_features, index_names = extract_index_features(classic_test_dataset, model)
    relative_test_dataset = CIRDataset('cirr', 'test1', 'relative', preprocess, args.data_path)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, model,
                                                                                  index_features, index_names,
                                                                                  txt_processors)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'blip24cir'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{args.submission_name}.json", 'w') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{args.submission_name}.json", 'w') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRDataset, model, index_features: torch.tensor,
                             index_names: List[str], txt_processors):
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: test index features
    :param index_names: test index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_sim, reference_names, group_members, pairs_id, captions_all, name2feat = \
        generate_cirr_test_predictions(model, relative_test_dataset, index_names,
                                       index_features, txt_processors)

    print(f"Compute CIRR prediction dicts")
    # Compute the distances and sort the results
    distances = 1 - predicted_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(model, relative_test_dataset: CIRDataset, index_names: List[str],
                                   index_features: torch.tensor, txt_processors):
    """
    Compute CIRR predictions on the test set
    :param clip_model: CLIP model
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                           features
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=4, pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    group_members = []
    reference_names = []
    distance = []
    captions_all = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data

        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = model.blip_model.inference(reference_image_features.to(device),
                                                        index_features[0].to(device),
                                                        captions)
            distance.append(batch_distance)
            captions_all += captions

        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    distance = torch.vstack(distance)

    return distance, reference_names, group_members, pairs_id, captions_all, name_to_feat


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, default='sprc', help="submission file name")
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str,
                        help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--combiner", type=str, default='sum',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--load_origin", action='store_true')
    parser.add_argument("--query_type", type=int, default=1)
    args = parser.parse_args()
    model = CIRPlus(args.blip_model_name)
    model.eval()
    input_dim = model.input_dim

    if args.model_path:
        if not args.load_origin:
            model.blip_model.init_stage2()
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

    generate_cirr_test_submissions(args.submission_name, model, preprocess, model.txt_processors)
