import math

import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
from torch import nn

import config
import datasets
from models.vision_transformer import TrOCR, CustomEncoderDecoder

# set matplotlib font to Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"


def pred_to_str(pred):
    s = ""
    for i in pred:
        if i not in [crohme.label2id["<pad>"], crohme.label2id["<sos>"], crohme.label2id["<eos>"]]:
            word = crohme.id2label[i]
            if len(word) > 1:
                word = " " + word + " "
            s += word
    return s


def show_generate(model, dataset: datasets.DatasetManager, loader, **kwargs):
    model.eval()
    with torch.no_grad():
        for img, true_output in loader.dataset:
            # pad image to dataset max size
            padded = torch.zeros((dataset.max_img_h, dataset.max_img_w), dtype=torch.float32)
            padded[:img.shape[0], :img.shape[1]] = img
            inputs = padded

            inputs = inputs.unsqueeze(0).float().to(config.device)

            # model(inputs, true_output)
            # print(model.result)
            # input("pause")
            max_len = max(10, round(true_output.shape[0] * 1.5))
            print(inputs.shape, inputs.mean(), true_output.shape, max_len, num_beams, kwargs)
            result = model.generate(inputs, max_length=max_len, **kwargs)[0]
            print("\nTrue:", pred_to_str(true_output.cpu().numpy()))
            print("Pred:",  pred_to_str(result.cpu().numpy()))

            pred_ids = result.cpu().numpy()
            s1 = pred_to_str(true_output.cpu().numpy())
            s2 = pred_to_str(result.cpu().numpy())

            # im = Image.fromarray((img.numpy() * 255).astype(np.uint8))
            # im.save(f"../reports/final_report/images/img.png")

            plt.imshow(img, cmap="gray")
            plt.title(f"True: {s1}\nPred: {s2}")
            plt.show()

            if input("Show attentions? (y/n)") == "y":

                # with torch.enable_grad():
                #     inputs = torch.autograd.Variable(inputs, requires_grad=True)
                #     model.forward(inputs, result.unsqueeze(0), output_attentions=True)
                #     # attentions = model.result.decoder_attentions
                #     a = model.result.logits.squeeze(0).sum(dim=-1)
                #
                #     for i in range(result.shape[0]):
                #         # attention.backward(torch.ones_like(attention), retain_graph=True)
                #         a[i].backward(torch.ones_like(a[i]), retain_graph=True)
                #         img_attn = inputs.grad.squeeze(0).cpu().numpy()
                #         plt.imshow(np.abs(img_attn), vmin=0)
                #         plt.show()

                n_cols = 4
                attentions = model.result.attentions
                low_res_shape = (padded.shape[0] // model.encoder.total_pooling, padded.shape[1] // model.encoder.total_pooling)

                fig, axs = plt.subplots(math.ceil(len(attentions) / n_cols), n_cols,
                                        figsize=(n_cols * 3, math.ceil(len(attentions) / n_cols) * 3),
                                        constrained_layout=True)

                for i, attention in enumerate(attentions):
                    attention = attention.squeeze(0).cpu().numpy()
                    attention = attention.reshape(*low_res_shape)
                    attention = attention.repeat(model.encoder.total_pooling, axis=0).repeat(model.encoder.total_pooling, axis=1)
                    attention = attention[:img.shape[0], :img.shape[1]]

                    # add attention in red
                    img_attn = np.zeros((img.shape[0], img.shape[1], 3))
                    img2 = 1 - img.float().numpy()
                    img_attn[:, :, 0] = img2
                    img_attn[:, :, 1] = img2 * (1 - attention)
                    img_attn[:, :, 2] = img2 * (1 - attention)

                    img_attn = np.pad(img_attn, ((2, 2), (2, 2), (0, 0)), mode="constant", constant_values=0)
                    ax = axs[i // n_cols, i % n_cols]
                    ax.imshow(img_attn)
                    ax.set_title(f"{crohme.id2label[pred_ids[i+1]]}")
                    ax.axis("off")

                plt.show()

            # break


def show_pos_embeddings(model: TrOCR):
    pe = model.encoder.get_position_embeddings().detach().cpu().numpy()

    n_cols = 2
    n_rows = 4

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1 * 4, n_rows * 1), constrained_layout=True)
    for i in range(n_rows * n_cols):
        ax = axs[i // n_cols, i % n_cols]
        img = pe[0, :, :, i]
        img = img.repeat(4, axis=0).repeat(4, axis=1)
        img = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=-1)
        ax.imshow(img, vmin=-.25, vmax=.25, cmap="seismic")
        # ax.set_title(f"{i}")
        ax.axis("off")

        # bottom colorbar
        # if i == 0:
        #     cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
        #     cbar = plt.colorbar(ax.images[0], cax=cax, orientation="horizontal")
        #     cbar.set_label("Positional embedding value")

    plt.show()


def edit_distance(seq1, seq2):
    """
    Modified from torchaudio.functional.edit_distance, returning the number of substitutions, insertions and deletions.
    """
    # trick: we will be able to know the number of ins, del and sub from the total cost
    max_ops = max(len(seq1), len(seq2)) + 1
    base_cost = 10 * max_ops ** 3
    sub_cost = base_cost + 1
    ins_cost = base_cost + max_ops
    del_cost = base_cost + max_ops**2

    len_sent2 = len(seq2)
    dold = [ins_cost*i for i in range(len_sent2 + 1)]
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i * sub_cost
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + sub_cost
                insertion = dnew[j - 1] + ins_cost
                deletion = dold[j] + del_cost
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    # deduce values
    cost = dold[-1]
    true_cost = cost // base_cost

    residual = cost % base_cost
    num_sub = residual % max_ops

    residual = residual // max_ops
    num_ins = residual % max_ops

    num_del = residual // max_ops

    return true_cost, num_sub, num_ins, num_del


def metrics(dataset, pred_seq, true_seq):
    metrics = dict()

    # remove special tokens
    pred_seq = [i for i in pred_seq if i not in [dataset.label2id["<pad>"], dataset.label2id["<sos>"], dataset.label2id["<eos>"]]]
    true_seq = [i for i in true_seq if i not in [dataset.label2id["<pad>"], dataset.label2id["<sos>"], dataset.label2id["<eos>"]]]

    # lengths
    metrics["pred_len"] = len(pred_seq)
    metrics["true_len"] = len(true_seq)

    # number of errors
    metrics["num_errors"] = sum([p != t for p, t in zip(pred_seq, true_seq)]) + abs(len(pred_seq) - len(true_seq))
    metrics["error_tol_0"] = (metrics["num_errors"] == 0) * 100
    metrics["error_tol_1"] = (metrics["num_errors"] <= 1) * 100
    metrics["error_tol_2"] = (metrics["num_errors"] <= 2) * 100

    # levenshtein distance, with number of substitutions, insertions and deletions
    metrics["Levenshtein"], metrics["WER_sub"], metrics["WER_ins"], metrics["WER_del"] = edit_distance(pred_seq, true_seq)
    metrics["WER"] = metrics["Levenshtein"] / len(true_seq) * 100

    # print("true", pred_to_str(true_seq))
    # print("pred", pred_to_str(pred_seq))
    # print(metrics)

    return metrics


def compute_model_metrics(model: TrOCR, dataset: datasets.DatasetManager, num_beams=None, **generate_kwargs):
    model.eval()
    with torch.no_grad():
        for test_name, test_loader in dataset.test_loaders.items():
            df = pandas.DataFrame(columns=["pred_len", "true_len", "num_errors", "error_tol_0", "error_tol_1", "error_tol_2", "WER", "WER_sub", "WER_ins", "WER_del"])
            # loop over each sample is slower than with batches, but it is better for reproducibility
            for img, output in tqdm.tqdm(test_loader.dataset, desc=f"Computing metrics for {test_name}"):
                # pad image to dataset max size
                padded = torch.zeros((dataset.max_img_h, dataset.max_img_w), dtype=torch.float32)
                padded[:img.shape[0], :img.shape[1]] = img
                img = padded

                inputs = img.unsqueeze(0).float().to(config.device)
                output = [int(i) for i in output if int(i) not in {dataset.label2id["<pad>"], dataset.label2id["<sos>"], dataset.label2id["<eos>"]}]

                max_len = max(10, round(len(output) * 1.5))
                # print(inputs.shape, max_len, output)
                # print(inputs.shape, inputs.mean(), max_len, num_beams, generate_kwargs)
                decoded = model.generate(inputs, max_length=max_len, num_beams=num_beams, **generate_kwargs)

                metrics_dict = metrics(dataset, decoded[0].detach().cpu().numpy(), output)
                df = pandas.concat([df, pandas.DataFrame(metrics_dict, index=[0])], ignore_index=True)
            #     break
            print(test_name)
            print(df.mean().round(2))
            # break


def print_dataset_stats(dataloaders, name=""):
    total = 0
    lengths = []
    img_widths = []
    img_heights = []
    for dataloader in dataloaders:
        for img, output in dataloader.dataset:
            lengths.append(len(output))
            img_widths.append(img.shape[1])
            img_heights.append(img.shape[0])
            total += 1

    print()
    print(name)
    print("Total:", total)
    quartiles = [0, 0.25, 0.5, 0.75, 1]
    for metric_name, metric in zip(["Length", "Image width", "Image height"], [lengths, img_widths, img_heights]):
        print(metric_name, "quartiles:", [round(np.quantile(metric, q), 2) for q in quartiles])


if __name__ == '__main__':

    crohme = datasets.DatasetManager(datasets.CROHME, batch_size=config.batch_size, precompute_padding=config.precompute_padding, batch_padding=config.batch_padding, downscale=config.downscale)

    # print_dataset_stats([crohme.train_loader], "Train")
    # print_dataset_stats([crohme.test_loaders["TEST14"]], "Test14")
    # print_dataset_stats([crohme.test_loaders["TEST16"]], "Test16")
    # print_dataset_stats([crohme.test_loaders["TEST19"]], "Test19")
    print_dataset_stats([crohme.train_loader, crohme.test_loaders["TEST14"], crohme.test_loaders["TEST16"], crohme.test_loaders["TEST19"]], "All")

    # model = TrOCR(crohme)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V1-3jxjprsy/epoch=epoch=414-step=step=57685-val_loss=val_loss=0.1211.ckpt", dataset=crohme)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V2-pcfs4qv7/epoch=499-step=69500-last.ckpt", dataset=crohme)
    # model = TrOCR.load_from_checkpoint("checkpoints/CNN_V3-bm63svkd/epoch=499-step=69500-last.ckpt", dataset=crohme)
    # model = CustomEncoderDecoder.load_from_checkpoint("checkpoints/Model_V4-3khbx46l/epoch=54-step=07645-last.ckpt", dataset=crohme)
    # model = CustomEncoderDecoder.load_from_checkpoint("checkpoints/WAP-2f0odbz4/epoch=499-step=69500-last.ckpt", dataset=crohme)
    # model = CustomEncoderDecoder.load_from_checkpoint("checkpoints/WS-WAP-1ugpg6mc/epoch=499-step=69500-last.ckpt", dataset=crohme)
    model = CustomEncoderDecoder.load_from_checkpoint("checkpoints/WS-WAP-ik0dgt27/epoch=499-step=69500-last.ckpt", dataset=crohme)

    # torch.save(model, "final_models/WAP-2.pt")

    # model = torch.load("final_models/CNN-V1.pt")
    # model = torch.load("final_models/CNN-V2.pt")
    # model = torch.load("final_models/CNN-V3.pt")

    # model = torch.load("final_models/WAP-1_updated.pt")
    # model.decoder.dropout = nn.Dropout(0)

    # model = torch.load("final_models/WAP-2.pt")  # best model

    model = model.to(config.device)
    model.eval()
    num_beams = None  # 10

    # Generate and show some examples
    # loader = crohme.train_loader
    # loader = crohme.test_loaders["TEST14"]
    # show_generate(model, crohme, loader, num_beams=num_beams, output_attentions=True)

    # Show positional embeddings
    show_pos_embeddings(model)

    # Compute metrics
    # compute_model_metrics(model, crohme, num_beams=num_beams)
