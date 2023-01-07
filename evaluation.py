import torch
import matplotlib.pyplot as plt

import config
import datasets
from models.vision_transformer import TrOCR, CustomEncoderDecoder


def show_generate(model, loader):

    with torch.no_grad():
        for imgs, true_outputs in loader:
            for img, true_output in zip(imgs, true_outputs):
                print(img.shape, true_output.shape)
                print("True:", [crohme.id2label[i] for i in true_output.cpu().numpy()])

                inputs = img.unsqueeze(0).to(config.device).float()

                # model(inputs, true_output)
                # print(model.result)
                # input("pause")

                result = model.generate(inputs)[0]
                print("Pred:", [crohme.id2label[i] for i in result.cpu().numpy()])

                plt.imshow(img)
                plt.show()


def show_pos_embeddings(model: TrOCR):
    pe = model.encoder.get_position_embeddings().detach().cpu().numpy()
    for i in range(pe.shape[3]):
        plt.imshow(pe[0, :, :, i])
        plt.show()


if __name__ == '__main__':


    crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, precompute_padding=config.precompute_padding, batch_padding=config.batch_padding, downscale=config.downscale)
    print("Train size:", len(crohme.train_loader.dataset))
    print("Test size 2014:", len(crohme.test_loaders['TEST14'].dataset))
    print("Test size 2016:", len(crohme.test_loaders['TEST16'].dataset))
    print("Test size 2019:", len(crohme.test_loaders['TEST19'].dataset))

    # model = TrOCR(crohme).to(config.device)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V1-3jxjprsy/epoch=epoch=414-step=step=57685-val_loss=val_loss=0.1211.ckpt", dataset=crohme).to(config.device)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V2-pcfs4qv7/epoch=499-step=69500-last.ckpt", dataset=crohme).to(config.device)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V3-bm63svkd/epoch=499-step=69500-last.ckpt", dataset=crohme).to(config.device)
    model = CustomEncoderDecoder.load_from_checkpoint("checkpoints/Model_V4-3khbx46l/epoch=54-step=07645-last.ckpt", dataset=crohme).to(config.device)

    model.eval()

    loader = crohme.train_loader
    # loader = crohme.test_loaders["TEST14"]
    show_generate(model, loader)

    # show_pos_embeddings(model)
