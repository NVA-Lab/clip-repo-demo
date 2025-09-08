import autorootcwd
import os, math
import torch, torch.nn as nn, torch.utils.data as data
import lightning as L
from lightning.pytorch.loggers import CSVLogger

import clip
from dataset import get_dataloaders

torch.set_float32_matmul_precision('medium')

# DEFINE THE FINETUNING ROUTINE
class ClipFinetuner(L.LightningModule):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image, text):
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        # # log max and min gradients
        # max_grad = torch.max(torch.stack([torch.max(p.grad.cpu()) if p.grad is not None else torch.tensor(0.0) for p in self.parameters()]))
        # min_grad = torch.min(torch.stack([torch.min(p.grad.cpu()) if p.grad is not None else torch.tensor(0.0) for p in self.parameters()]))

        # self.log_dict({"grad_max": max_grad.item(), "grad_min": min_grad.item()})

        images, tokenized_text = batch # images:(batch, channels, width, height), tokenized_text:(batch, tokenizer_dim)
        tokenized_text = torch.stack(tokenized_text).squeeze(1)

        # get embeddings
        image_features, text_features = self(images, tokenized_text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # PUSH X,Y together and push other vectors away. cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # create targets for binary cross-entropy + binary cross entropy
        targets = torch.eye(logits_per_image.size(0), dtype=torch.float32, device=logits_per_image.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_per_image, targets)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    import os
    os.environ["NCCL_P2P_DISABLE"] = "1" # remove error of ddp
    # LOAD OPEN AI MODEL
    clip_model, preprocess = clip.load("ViT-B/32")

    # LOAD THE DATA - CLIP expects 224x224 images
    train_dataloader, val_dataloader = get_dataloaders(batch_size=8, num_workers=2)

    # SETUP THE FINETUNING
    clip_finetuner = ClipFinetuner(clip_model)

    # CSV Logger 설정
    csv_logger = CSVLogger("logs", name="clip_pretrain")
    
    # PTL TRAINER auto-scales across CPUs, GPUs, etc...
    trainer = L.Trainer(max_steps=1000, log_every_n_steps=2, accelerator='gpu', devices=2, strategy='ddp_find_unused_parameters_true', logger=csv_logger)
    trainer.fit(clip_finetuner, train_dataloader)