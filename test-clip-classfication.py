from collections import OrderedDict
import os
from typing import List
import clip
from clip.model import CLIP
import ordered_set
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

from torch.utils import data
import tqdm
import yaml
from tsgan.data import CroppedCOCO

batch = 8
print(f"batch size: {batch}")
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"    
    model, preprocess = clip.load('RN50', device)
    model:CLIP = model
    checkpoint = torch.load("tmp/clip/clip_coco-27.pt", map_location="cpu")
    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
    # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    data_set = CroppedCOCO(
        config_file='configs/coco.yaml',
        is_train=True,
        transform=preprocess,
    )
    save_file = 'tmp/coco-object-dist.yaml'
    with open(save_file, 'w') as file:
        sorted_dict = OrderedDict(sorted(
            data_set.class_detail().items(),
            key=lambda x: x[1],
            reverse=True,
        ))
        yaml.dump({k: v for k, v in sorted_dict.items()}, file, sort_keys=False)
        print(f" --> categories distribution saved to {save_file}")

    # print({v: k for k, v in data_set.class_detail().items()})
    data_loader = data.DataLoader(
        data_set,
        batch_size=batch,
        num_workers=8,
        drop_last=True,
    )
    # print(data_set.categories)
    text_inputs = torch.cat([clip.tokenize(f"a {c}") for c in data_set.categories.keys()]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    pbar = tqdm.tqdm(data_loader)
    correct_count = 0
    total_count = 0
    for batch_data in pbar:

        images: torch.Tensor = batch_data["image"].to(device)
        labels: List[int] = batch_data["predict_id"].tolist()
        categories: List[int] = batch_data["category_id"].tolist()
        

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            top_res = [similarity[i].topk(1) for i in range(batch)]


        predict_result_zip_list=list(zip(
            labels,            
            [int(res.indices) for res in top_res],
            categories,
            [float(res.values) for res in top_res],
        ))
        correct=0
        for pred in predict_result_zip_list:
            if pred[0] == pred[1]:
                correct+=1
        correct_count+=correct
        total_count+=batch
        
        pbar.set_description(f"batch Prec@1: {correct/batch:.3f}; Prec@1: {correct_count/total_count:.3f}")



# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# # Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/Datasets"), download=True, train=False)

# # Prepare the inputs
# image, class_id = cifar100[3637]

# image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)

# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# values, indices = similarity[0].topk(5)

# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

if __name__ == "__main__":
    main()