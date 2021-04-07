__author__ = "yunbo"
from comet_ml import Experiment, ExistingExperiment
import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
import torch
from data.cloudcast import CloudCast
import lpips
from skimage.metrics import structural_similarity

# from skimage.measure import compare_ssim
# import skimage.measure
from core.utils import preprocess, metrics
from tqdm import tqdm
from utils import init_net, upload_images

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PyTorch video prediction model - PredRNN")

# training/test
parser.add_argument("--is_training", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")

# data
parser.add_argument("--dataset_name", type=str, default="mnist")
parser.add_argument(
    "--train_data_paths",
    type=str,
    default="data/moving-mnist-example/moving-mnist-train.npz",
)
parser.add_argument(
    "--valid_data_paths",
    type=str,
    default="data/moving-mnist-example/moving-mnist-valid.npz",
)
parser.add_argument("--save_dir", type=str, default="checkpoints/mnist_predrnn")
parser.add_argument("--gen_frm_dir", type=str, default="results/mnist_predrnn")
parser.add_argument("--input_length", type=int, default=10)
parser.add_argument("--total_length", type=int, default=20)
parser.add_argument("--img_width", type=int, default=128)
parser.add_argument("--img_channel", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-pn", "--projectname", default="predrnn", type=str, help="comet-ml project name",
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument(
    "--cometid", type=str, default="", help="the comet id to resume exps",
)
parser.add_argument(
    "-rs", "--randomseed", type=int, default=2021, help="random seed for the training",
)
parser.add_argument(
    "-v", "--verbose", type=int, default=0, help="print lots of details",
)
# model
parser.add_argument("--model_name", type=str, default="predrnn")
parser.add_argument("--pretrained_model", type=int, default=0)
parser.add_argument("--num_hidden", type=str, default="64,64,64,64")
parser.add_argument("--filter_size", type=int, default=5)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--patch_size", type=int, default=4)
parser.add_argument("--layer_norm", type=int, default=1)
parser.add_argument("--decouple_beta", type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument("--reverse_scheduled_sampling", type=int, default=0)
parser.add_argument("--r_sampling_step_1", type=float, default=25000)
parser.add_argument("--r_sampling_step_2", type=int, default=50000)
parser.add_argument("--r_exp_alpha", type=int, default=5000)
# scheduled sampling
parser.add_argument("--scheduled_sampling", type=int, default=1)
parser.add_argument("--sampling_stop_iter", type=int, default=50000)
parser.add_argument("--sampling_start_value", type=float, default=1.0)
parser.add_argument("--sampling_changing_rate", type=float, default=0.00002)
parser.add_argument(
    "-nbv",
    "--n_batch_val",
    type=int,
    default=0,
    help="If 1, only eval n batches during the validation phase to save time.",
)
# optimization
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--reverse_input", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=36)
parser.add_argument("--max_iterations", type=int, default=80000)
parser.add_argument("--display_interval", type=int, default=100)
parser.add_argument("--test_interval", type=int, default=5000)
parser.add_argument("--snapshot_interval", type=int, default=5)
parser.add_argument("--num_save_samples", type=int, default=10)
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument(
    "--milestones",
    type=list,
    default=[100, 150, 200, 250],
    help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150, 200, 250]",
)
# visualization of memory decoupling
parser.add_argument("--visual", type=int, default=0)
parser.add_argument("--visual_path", type=str, default="./decoupling_visual")

args = parser.parse_args()
print(args)

random_seed = args.randomseed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if args.pretrained_model:
    if not args.nocomet:
        comet_exp = ExistingExperiment(previous_experiment=args.cometid)
    else:
        comet_exp = None
    # load statedict here
else:
    # start logging info in comet-ml
    if not args.nocomet:
        comet_exp = Experiment(workspace=args.workspace, project_name=args.projectname)
        # comet_exp.log_parameters(flatten_opts(args))
    else:
        comet_exp = None


def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(
            -float(itr - args.r_sampling_step_1) / args.r_exp_alpha
        )
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (
            itr - args.r_sampling_step_1
        )
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((args.batch_size, args.input_length - 1))
    r_true_token = r_random_flip < r_eta

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1)
    )
    true_token = random_flip < eta

    ones = np.ones(
        (
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        )
    )
    zeros = np.zeros(
        (
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        )
    )

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(
        real_input_flag,
        (
            args.batch_size,
            args.total_length - 2,
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        ),
    )
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros(
        (
            args.batch_size,
            args.total_length - args.input_length - 1,
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        )
    )
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1)
    )
    true_token = random_flip < eta
    ones = np.ones(
        (
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        )
    )
    zeros = np.zeros(
        (
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        )
    )
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(
        real_input_flag,
        (
            args.batch_size,
            args.total_length - args.input_length - 1,
            args.img_width // args.patch_size,
            args.img_width // args.patch_size,
            args.patch_size ** 2 * args.img_channel,
        ),
    )
    return eta, real_input_flag


def test(model, configs, itr, loader):

    loss_fn_alex = lpips.LPIPS(net="alex").to(args.device)
    # res_path = os.path.join(configs.gen_frm_dir, str(itr))
    # os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []
    t_test = tqdm(loader, leave=False, total=2)
    image_output = []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (
            configs.batch_size,
            configs.total_length - mask_input - 1,
            configs.img_width // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size ** 2 * configs.img_channel,
        )
    )

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, : configs.input_length - 1, :, :] = 1.0
    with torch.no_grad():
        for i, inputVar in enumerate(t_test):
            if batch_id < args.n_batch_val or (not args.n_batch_val):
                if args.n_batch_val and args.verbose:
                    print(
                        "Warning, you activated the feature that only eval on one batch to save time on eval. This may introduce uncertainty of val losses!"
                    )
                batch_id = batch_id + 1
                test_ims = inputVar.to(args.device)
                # print("inputs", inputs.shape)
                test_dat = preprocess.reshape_patch(test_ims.cpu(), configs.patch_size)
                # print("test_dat:", test_dat.shape)
                # print("real_input_flag:", real_input_flag.shape)
                img_gen = model.test(test_dat, real_input_flag)

                img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
                output_length = configs.total_length - configs.input_length
                img_gen_length = img_gen.shape[1]
                img_out = img_gen[:, -output_length:]

                # MSE per frame
                for i in range(output_length):
                    x = test_ims[:, i + configs.input_length, :, :, :].to(args.device)
                    gx = torch.tensor(img_out[:, i, :, :, :]).to(args.device)
                    gx = torch.maximum(gx, torch.zeros(gx.shape).to(args.device))
                    gx = torch.minimum(gx, torch.ones(gx.shape).to(args.device))
                    mse = torch.square(x - gx).sum()
                    img_mse[i] += mse
                    avg_mse += mse
                    # cal lpips
                    # img_x = torch.zeros(
                    #     [configs.batch_size, 3, configs.img_width, configs.img_width]
                    # ).to(args.device)
                    if configs.img_channel == 3:
                        img_x = torch.movedim(x, -1, 1)
                    elif configs.img_channel == 1:
                        img_x = torch.movedim(x, -1, 1).repeat(1, 3, 1, 1)
                    else:
                        raise ValueError(configs.img_channel)
                    # img_gx = torch.zeros(
                    #     [configs.batch_size, 3, configs.img_width, configs.img_width]
                    # ).to(args.device)
                    if configs.img_channel == 3:
                        img_gx = torch.movedim(gx, -1, 1)
                    elif configs.img_channel == 1:
                        img_gx = torch.movedim(gx, -1, 1).repeat(1, 3, 1, 1)
                    else:
                        raise ValueError(configs.img_channel)
                    lp_loss = loss_fn_alex(img_x, img_gx)
                    lp[i] += torch.mean(lp_loss).item()

                    real_frm = np.uint8(x.cpu() * 255)
                    pred_frm = np.uint8(gx.cpu() * 255)

                    psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                    for b in range(configs.batch_size):
                        # score = 10
                        # original method is depricated
                        score, _ = structural_similarity(
                            pred_frm[b], real_frm[b], full=True, multichannel=True
                        )
                        ssim[i] += score

            # save prediction examples
            if batch_id <= configs.num_save_samples:
                # path = os.path.join(res_path, str(batch_id))
                # os.mkdir(path)
                for i in range(configs.total_length):
                    # name = "gt" + str(i + 1) + ".png"
                    # file_name = os.path.join(path, name)
                    img_gt = torch.tensor(test_ims[0, i, :, :, :] * 255)
                    image_output.append(
                        torch.movedim(img_gt, -1, 0)
                        .unsqueeze(0)
                        .repeat(1, 3, 1, 1)
                        .cpu()
                    )
                    # print("img_gt", img_gt.shape)
                for i in range(img_gen_length):
                    # name = "pd" + str(i + 1 + configs.input_length) + ".png"
                    # file_name = os.path.join(path, name)
                    img_pd = torch.tensor(img_gen[0, i, :, :, :])
                    img_pd = torch.maximum(img_pd, torch.zeros(img_pd.shape))
                    img_pd = torch.minimum(img_pd, torch.ones(img_pd.shape))
                    img_pd = torch.tensor(img_pd * 255)
                    image_output.append(
                        torch.movedim(img_pd, -1, 0)
                        .unsqueeze(0)
                        .repeat(1, 3, 1, 1)
                        .cpu()
                    )
                    # print("img_pd", img_pd.shape)

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print("mse per seq: " + str(avg_mse))
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print("ssim per frame: " + str(np.mean(ssim)))
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print("psnr per frame: " + str(np.mean(psnr)))
    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print("lpips per frame: " + str(np.mean(lp)))
    if args.verbose:
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / (batch_id * configs.batch_size))
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i])
        for i in range(configs.total_length - configs.input_length):
            print(psnr[i])
        for i in range(configs.total_length - configs.input_length):
            print(lp[i])
    print("Validation phase complete!")
    return avg_mse, np.mean(ssim), np.mean(psnr), image_output


def cloud_cast_wrapper(model):
    trainFolder = CloudCast(
        is_train=True,
        root="data/",
        n_frames_input=10,
        n_frames_output=10,
        batchsize=args.batch_size,
    )

    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    testFolder = CloudCast(
        is_train=False,
        root="data/",
        n_frames_input=10,
        n_frames_output=10,
        batchsize=args.batch_size,
    )
    # number of workers will need to be changed
    testLoader = torch.utils.data.DataLoader(
        testFolder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    # device may need to change
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = tqdm(trainLoader, leave=False, total=2)
    for epoch in range(0, int(args.epochs)):
        print("epoch: ", epoch)
        train_loss = 0
        best_psnr = -1
        best_ssim = -1
        # val_loss = 0
        # val_ssim = 0
        # val_psnr = 0
        if epoch == 0:
            avg_mse, avg_ssim, avg_psnr, image_output = test(
                model, args, epoch, testLoader
            )
            comet_exp.log_metric("ValLoss", avg_mse.item(), epoch=epoch)
            comet_exp.log_metric("SSIM", avg_ssim, epoch=epoch)
            comet_exp.log_metric("PSNR", avg_psnr, epoch=epoch)
            upload_images(
                image_output,
                epoch,
                exp=comet_exp,
                im_per_row=2,
                rows_per_log=int(len(image_output) / 2),
            )
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                model.save_best("PSNR")
                print("New Best PSNR found and saved at " + str(epoch))
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                model.save_best("SSIM")
                print("New Best SSIM found and saved at " + str(epoch))
        for i, inputVar in enumerate(t):
            if args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(i)
            ims = preprocess.reshape_patch(inputVar.cpu(), args.patch_size)
            loss = model.train(ims, real_input_flag)
            train_loss += loss.item()
        comet_exp.log_metric("TrainLoss", train_loss / len(trainLoader), epoch=epoch)
        model.save(epoch)
        avg_mse, avg_ssim, avg_psnr, image_output = test(model, args, epoch, testLoader)
        comet_exp.log_metric("ValLoss", avg_mse.item(), epoch=epoch)
        comet_exp.log_metric("SSIM", avg_ssim, epoch=epoch)
        comet_exp.log_metric("PSNR", avg_psnr, epoch=epoch)
        upload_images(
            image_output,
            epoch,
            exp=comet_exp,
            im_per_row=2,
            rows_per_log=int(len(image_output) / 2),
        )
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model.save_best("PSNR")
            print("New Best PSNR found and saved at " + str(epoch))
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            model.save_best("SSIM")
            print("New Best SSIM found and saved at " + str(epoch))


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print("Initializing models")

model = Model(args)

#
eta = args.sampling_start_value

if args.dataset_name == "cloud_cast":
    print("Training cloud cast")
    cloud_cast_wrapper(model)
