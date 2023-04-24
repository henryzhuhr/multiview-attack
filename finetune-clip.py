import datetime
import os
import torch
from torch.utils import data
import clip
import tqdm
from tsgan.data.crop_coco import CroppedCOCO

BATCH_SIZE = 256
EPOCH = 200


# https://github.com/openai/CLIP/issues/83
def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    print("clip available models: ", clip.available_models())
    model, preprocess = clip.load("RN50", device=device, jit=False) # Must set jit=False for training

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    data_set = CroppedCOCO(
        config_file='configs/coco.yaml',
        is_train=True,
        transform=preprocess,
    )
    data_loader = data.DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(# Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        model.parameters(),
        lr=5e-6,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.001,# https://github.com/openai/CLIP/issues/150#issuecomment-976076317
    )

    os.makedirs("tmp/clip", exist_ok=True)
    open("tmp/clip/clip_coco.txt", "w").close()
    for epoch in range(EPOCH):
        pbar = tqdm.tqdm(data_loader)
        for batch_data in pbar:
            optimizer.zero_grad()

            images: torch.Tensor = batch_data["image"].to(device)
            labels: torch.Tensor = batch_data["predict_id"].to(device)
            category_text = [f"a {c}" for c in batch_data["category_name"]]
            texts = clip.tokenize(category_text).to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            log_info="Epoch {epoch} loss: {total_loss.item():.4f}"
            pbar.set_description(log_info)

        with open("tmp/clip/clip_coco.txt", "a") as f:
            f.write(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log_info}" 
            )
        if epoch % 5 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, f"tmp/clip/clip_coco-{epoch}.pt"
            )                                                         #just change to your preferred folder/filename


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if __name__ == "__main__":
    main()