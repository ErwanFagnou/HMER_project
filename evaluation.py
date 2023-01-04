import torch

import config
import datasets
from models.vision_transformer import TrOCR


if __name__ == '__main__':
    crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, precompute_padding=config.precompute_padding, batch_padding=config.batch_padding, downscale=config.downscale)
    print(crohme.max_img_h, crohme.max_img_w, crohme.max_label_len)

    # model = TrOCR(crohme).to(config.device)
    model = TrOCR.load_from_checkpoint("lightning_logs/version_23/checkpoints/epoch=25-step=7202.ckpt", dataset=crohme).to(config.device)
    model.eval()

    import matplotlib.pyplot as plt
    for imgs, true_outputs in crohme.test_loaders["TEST14"]:
        for img, true_output in zip(imgs, true_outputs):
            print(img.shape, true_output.shape)
            print("True:", [crohme.id2label[i] for i in true_output.cpu().numpy()])

            inputs = img.unsqueeze(0).to(config.device)
            result = model.generate(inputs)[0]
            print("Pred:", [crohme.id2label[i] for i in result.cpu().numpy()])

            plt.imshow(img)
            plt.show()
        # break

