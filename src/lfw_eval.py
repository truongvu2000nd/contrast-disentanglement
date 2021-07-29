from __future__ import print_function
import torch
import sys
from PIL import Image
from torchvision import transforms
import tqdm
import cv2
import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import os
import six

from utils.matlab_cp2tform import get_similarity_transform_for_cv2
from utils.benchmark_helpers import compose_transforms
from utils.utils import str2bool
from models.vggface import vgg_face
from torchsummary import summary

torch.backends.cudnn.bencmark = True
MODEL_DIR = os.path.expanduser("~/data/models/pytorch/mcn_imports/")


def alignment(src_img, src_pts, output_size=(96, 112)):
    """Warp a face image so that its features align with a canoncial
    reference set of landmarks. The alignment is performed with an
    affine warp

    Args:
        src_img (ndarray): an HxWx3 RGB containing a face
        src_pts (ndarray): a 5x2 array of landmark locations
        output_size (tuple): the dimensions (oH, oW) of the output image

    Returns:
        (ndarray): an (oH x oW x 3) warped RGB image.
    """
    ref_pts = [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ]
    src_pts = np.array(src_pts).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, output_size)
    return face_img


def KFold(num_pairs, n_folds, shuffle=False):
    folds = []
    base = list(range(num_pairs))
    for i in range(n_folds):
        test = base[i * num_pairs // n_folds: (i + 1) * num_pairs // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def modify_to_return_embeddings(net, model_name):
    """Modify the structure of the network (if necessary), to ensure that it
    returns embeddings (the features from the penultimate layer of the network,
    just before the classifier)

    Args:
         net (nn.Module): the network to be modified
         model_name (str): the name of the network

    Return:
        (nn.Module): the modified network

    NOTE:
        We use `nn.Sequential` to simluate Identity (i.e. no-op).
    """
    if model_name in ["vgg_face_dag", "vgg_m_face_bn_dag"]:
        net.fc8 = torch.nn.Sequential()
    else:
        msg = "{} not yet supported".format(model_name)
        raise NotImplementedError(msg)
    return net


def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(MODEL_DIR, model_name + ".py")
    weights_path = os.path.join(MODEL_DIR, model_name + ".pth")
    if six.PY3:
        import importlib.util

        spec = importlib.util.spec_from_file_location(model_name,
                                                      model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    net = modify_to_return_embeddings(net, model_name)
    return net


def extract_features(net, ims):
    """Extract penultimate features from network

    Args:
        net (nn.Module): the network to be used to compute features
        ims (torch.Tensor): the data to be processed

    NOTE:
        Pretrained networks often vary in the manner in which their outputs
        are returned.  For example, some return the penultimate features as
        a second argument, while others need to be modified directly and will
        return these features as their only output.
    """
    outs = net(ims)
    if isinstance(outs, list):
        outs = outs[1]
    features = outs.data
    return features


def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER) from the predictions and scores.

    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): the confidence of the prediction that the
            given sample is a positive.

    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold

    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.

       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch sphereface lfw")
    parser.add_argument("--net", default="sphere20a", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--model_name", default="data", type=str)
    parser.add_argument("--model_path", default="model.pth", type=str)
    parser.add_argument('--cuda', default=True,
                        type=str2bool, help='enable cuda')

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    net = vgg_face(args.model_path).to(device)
    net.eval()
    net.feature = True

    landmark = {}
    with open(os.path.join(args.data_dir, "lfw_landmark.txt")) as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        ll = line.replace("\n", "").split("\t")
        landmark[ll[0]] = [int(k) for k in ll[1:]]

    with open(os.path.join(args.data_dir, "pairs.txt")) as f:
        next(f)
        pairs_lines = f.readlines()

    # orig_pairs = 6000
    # if args.limit:
    #     num_pairs = min(orig_pairs, args.limit)
    # else:
    #     num_pairs = orig_pairs
    num_pairs = 6000

    # -----------Aligned---------------
    summary(net, (3, 224, 224))
    predicts = []
    for i in tqdm.tqdm(range(num_pairs)):
        p = pairs_lines[i].replace("\n", "").split("\t")

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))

        im1 = cv2.imread(os.path.join(args.data_dir, "lfw", name1))
        img1_aligned = alignment(im1, landmark[name1])
        im2 = cv2.imread(os.path.join(args.data_dir, "lfw", name2))
        img2_aligned = alignment(im2, landmark[name2])

        # convert images to PIL to use builtin transforms
        # import matplotlib.pyplot as plt
        img1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1)
        img2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

        meta = net.meta
        preproc_transforms = compose_transforms(
            meta=meta,
            center_crop=False
        )

        imglist = [img1, img2]

        for i in range(len(imglist)):
            imglist[i] = preproc_transforms(imglist[i])

        ims = torch.stack(imglist, dim=0)
        with torch.no_grad():
            ims = imgs.to(device)
            features = extract_features(net, ims)
            f1, f2 = features[0].squeeze(), features[1].squeeze()
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            pred = [name1, name2, cosdistance.item(), sameflag]
            predicts.append(pred)

    accuracy = []
    thd = []
    folds = KFold(num_pairs=num_pairs, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(predicts)
    eers = []
    eer_thresholds = []
    for idx, (train, test) in tqdm.tqdm(enumerate(folds)):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        scores = [float(x[2]) for x in predicts[test]]
        labels = [int(x[3]) for x in predicts[test]]
        eer, thresh = compute_eer(labels=labels, scores=scores)
        eers.append(eer)
        eer_thresholds.append(thresh)

    print("Aligned images:")
    msg = "LFWACC={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    msg = "EER={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(eers), np.std(eers), np.mean(eer_thresholds)))

    # Add blanks to prevent tqdm from swallowing the summary
    print("")

    # ------------Unaligned-------------
    summary(net, (3, 224, 224))
    predicts = []
    for i in tqdm.tqdm(range(num_pairs)):
        p = pairs_lines[i].replace("\n", "").split("\t")

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))

        im1 = cv2.imread(os.path.join(args.data_dir, "lfw", name1))
        # img1_aligned = alignment(im1, landmark[name1])
        im2 = cv2.imread(os.path.join(args.data_dir, "lfw", name2))
        # img2_aligned = alignment(im2, landmark[name2])

        # convert images to PIL to use builtin transforms
        # import matplotlib.pyplot as plt
        img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1)
        img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

        meta = net.meta
        preproc_transforms = compose_transforms(
            meta=meta,
            center_crop=False
        )

        imglist = [img1, img2]

        for i in range(len(imglist)):
            imglist[i] = preproc_transforms(imglist[i])

        ims = torch.stack(imglist, dim=0)
        with torch.no_grad():
            ims = imgs.to(device)
            features = extract_features(net, ims)
            f1, f2 = features[0].squeeze(), features[1].squeeze()
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            pred = [name1, name2, cosdistance.item(), sameflag]
            predicts.append(pred)

    accuracy = []
    thd = []
    folds = KFold(num_pairs=num_pairs, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(predicts)
    eers = []
    eer_thresholds = []
    for idx, (train, test) in tqdm.tqdm(enumerate(folds)):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        scores = [float(x[2]) for x in predicts[test]]
        labels = [int(x[3]) for x in predicts[test]]
        eer, thresh = compute_eer(labels=labels, scores=scores)
        eers.append(eer)
        eer_thresholds.append(thresh)

    print("Unaligned images:")
    msg = "LFWACC={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    msg = "EER={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(eers), np.std(eers), np.mean(eer_thresholds)))

    # Add blanks to prevent tqdm from swallowing the summary
    print("")

    # ------------Not include top-------------
    predicts = []
    net.include_top = False
    summary(net, (3, 224, 224))

    for i in tqdm.tqdm(range(num_pairs)):
        p = pairs_lines[i].replace("\n", "").split("\t")

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
            name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))

        im1 = cv2.imread(os.path.join(args.data_dir, "lfw", name1))
        # img1_aligned = alignment(im1, landmark[name1])
        im2 = cv2.imread(os.path.join(args.data_dir, "lfw", name2))
        # img2_aligned = alignment(im2, landmark[name2])

        # convert images to PIL to use builtin transforms
        # import matplotlib.pyplot as plt
        img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1)
        img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

        meta = net.meta
        preproc_transforms = compose_transforms(
            meta=meta,
            center_crop=False
        )

        imglist = [img1, img2]

        for i in range(len(imglist)):
            imglist[i] = preproc_transforms(imglist[i])

        ims = torch.stack(imglist, dim=0)
        with torch.no_grad():
            ims = imgs.to(device)
            features = extract_features(net, ims)
            f1, f2 = features[0].squeeze(), features[1].squeeze()
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            pred = [name1, name2, cosdistance.item(), sameflag]
            predicts.append(pred)

    accuracy = []
    thd = []
    folds = KFold(num_pairs=num_pairs, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(predicts)
    eers = []
    eer_thresholds = []
    for idx, (train, test) in tqdm.tqdm(enumerate(folds)):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        scores = [float(x[2]) for x in predicts[test]]
        labels = [int(x[3]) for x in predicts[test]]
        eer, thresh = compute_eer(labels=labels, scores=scores)
        eers.append(eer)
        eer_thresholds.append(thresh)

    print("Not include top:")
    msg = "LFWACC={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    msg = "EER={:.4f} std={:.4f} thd={:.4f}"
    print(msg.format(np.mean(eers), np.std(eers), np.mean(eer_thresholds)))

    # Add blanks to prevent tqdm from swallowing the summary
    print("")
