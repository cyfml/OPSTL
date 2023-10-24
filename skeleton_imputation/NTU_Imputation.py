import sys
import cv2
from config import *
from model import *
import pickle
import torch
import numpy as np
import random
from tqdm import tqdm
from module.gcn.st_gcn import Model
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from numpy.lib.format import open_memmap

croped_size = 50
C = 3
V = 25
M = 2
toolbar_width = 30


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)


def _y_transmat(thetas):
    tms = np.zeros((0, 3, 3))
    thetas = thetas * np.pi / 180
    for theta in thetas:
        tm = np.zeros((3, 3))
        tm[0, 0] = np.cos(theta)
        tm[0, 2] = -np.sin(theta)
        tm[1, 1] = 1
        tm[2, 0] = np.sin(theta)
        tm[2, 2] = np.cos(theta)
        tm = tm[np.newaxis, :, :]
        tms = np.concatenate((tms, tm), axis=0)
    return tms


def parallel_skeleton(ins_data):
    right_shoulder = ins_data[:, :, 8]  # 9th joint
    left_shoulder = ins_data[:, :, 4]  # 5tf joint
    vec = right_shoulder - left_shoulder
    vec[1, :] = 0
    # print(vec.shape)
    l2_norm = np.sqrt(np.sum(np.square(vec), axis=0))
    theta = vec[0, :] / (l2_norm + 0.0001)
    # print(l2_norm)
    thetas = np.arccos(theta) * (180 / np.pi)
    isv = np.sum(vec[2, :])
    if isv >= 0:
        thetas = -thetas
    # print (thetas)
    y_tms = _y_transmat(thetas)
    # print(y_tms)
    new_skel = np.zeros(shape=(0, 25, 3))
    # print(new_skel.shape)
    ins_data = ins_data.transpose(1, 2, 0)
    # print(ins_data.shape, new_skel.shape)
    for ind, each_s in enumerate(ins_data):
        # print(each_s.shape)
        r = np.reshape(each_s, newshape=(25, 3))
        r = np.transpose(r)
        r = np.dot(y_tms[ind], r)
        r_t = np.transpose(r)
        r_t = np.reshape(r_t, newshape=(1, -1, 3))
        # print(new_skel.shape, r_t.shape)
        new_skel = np.concatenate((new_skel, r_t), axis=0)
    return new_skel, ins_data


@staticmethod
def real_resize(data_numpy, length, crop_size):
    C, T, V, M = data_numpy.shape
    new_data = np.zeros([C, crop_size, V, M])
    for i in range(M):
        tmp = cv2.resize(
            data_numpy[:, :length, :, i].transpose([1, 2, 0]),
            (V, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tmp = tmp.transpose([2, 0, 1])
        new_data[:, :, :, i] = tmp
    return new_data.astype(np.float32)


def print_toolbar(rate, annotation=""):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(" ")
        else:
            sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\r")


def end_toolbar():
    sys.stdout.write("\n")


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        self.data = np.load(self.data_path)
        # N C T V M
        N, C, T, V, M = self.data.shape
        self.size = N

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple:
        data = np.array(self.data[index])
        return index, data


class Complete_Joints:
    @ex.capture
    def __init__(
        self,
        original_train_data_nan: str,
        original_test_data_nan: str,
        missing_joints_train: str,
        missing_joints_test: str,
        train_list: str,
        test_list: str,
    ):
        self.original_train_data_nan = original_train_data_nan
        self.original_test_data_nan = original_test_data_nan
        self.missing_joints_train = missing_joints_train
        self.missing_joints_test = missing_joints_test
        self.train_list = train_list
        self.test_list = test_list
        self.load_data()
        self.load_model()

    @ex.capture
    def load_data(self, batch_size):
        self.dataset = dict()
        self.data_loader = dict()
        self.data = np.load(self.train_list)
        self.test = np.load(self.test_list)
        self.original_train_nan_data = np.load(self.original_train_data_nan)
        self.original_test_nan_data = np.load(self.original_test_data_nan)
        self.dataset["complete_train"] = DataSet(self.train_list)
        self.dataset["complete_test"] = DataSet(self.test_list)

        self.data_loader["complete_train"] = torch.utils.data.DataLoader(
            dataset=self.dataset["complete_train"],
            batch_size=batch_size,
            num_workers=32,
            shuffle=False,
        )

        self.data_loader["complete_test"] = torch.utils.data.DataLoader(
            dataset=self.dataset["complete_test"],
            batch_size=batch_size,
            num_workers=32,
            shuffle=False,
        )

        with open(self.missing_joints_train, "rb") as f:
            self.missing_joints_train = pickle.load(f)
            self.missing_joints_train = np.array(self.missing_joints_train)

        with open(self.missing_joints_test, "rb") as f:
            self.missing_joints_test = pickle.load(f)
            self.missing_joints_test = np.array(self.missing_joints_test)

    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    @ex.capture
    def load_model(
        self,
        weight_path,
        in_channels,
        hidden_channels,
        hidden_dim,
        dropout,
        graph_args,
        edge_importance_weighting,
        label_num,
    ):
        self.encoder = Model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
        )
        self.encoder = self.encoder.cuda()
        self.load_weights(self.encoder, weight_path)

    @ex.capture
    def Kmeans(self, cluster_num, output_path, split, dataset, n_neighbors):
        self.get_embedding(mode="train")
        f = open("{}/{}-kmeans-{}.pkl".format(output_path, split, "train"), "rb")
        x = pickle.load(f)
        x = x.detach().cpu().numpy()
        kmeans_model = KMeans(
            n_clusters=cluster_num, max_iter=100, random_state=0, n_init="auto"
        )
        self.fitted_model = kmeans_model.fit(x)
        labels = self.fitted_model.labels_

        output_folder = "{}/{}_completed_frame50/{}".format(output_path, dataset, split)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fp = open_memmap(
            "{}/{}_completed_frame50/{}/train_position.npy".format(
                output_path, dataset, split
            ),
            dtype="float32",
            mode="w+",
            shape=(len(labels), C, croped_size, V, M),
        )

        imputer = KNNImputer(
            n_neighbors=n_neighbors, weights="distance", keep_empty_features=True
        )
        self.knn_models = []
        for i in range(cluster_num):
            print_toolbar(
                i * 1.0 / cluster_num,
                "({:>5}/{:<5}) Processing {}-{} missing joints data in {}th cluster: ".format(
                    i + 1, cluster_num, split, "train", i + 1
                ),
            )
            indexes = np.argwhere(labels == i).reshape(-1)
            if len(indexes) == 0:
                continue
            X = self.original_train_nan_data[indexes].reshape(indexes.shape[0], -1)
            missing_joints = self.missing_joints_train[indexes]
            knn_model = imputer.fit(X)
            self.knn_models.append(knn_model)
            X = knn_model.transform(X)
            X = X.reshape(indexes.shape[0], C, -1, V, M)
            X = np.where(X == np.nan, 0, X)
            for i, index in enumerate(indexes):
                frame_length = len(missing_joints[i])
                dC, dT, dV, dM = X[i, :, :, :, :].shape
                center = X[i, :, :, 20, :].reshape([dC, dT, 1, dM])
                X[i, :, :, :, :] = X[i, :, :, :, :] - center
                new_data = self.real_resize(X[i, :, :, :, :], frame_length, croped_size)
                for n in range(2):
                    tmp_new, tmp_old = parallel_skeleton(new_data[:, :, :, n])
                    new_data[:, :, :, n] = tmp_new.transpose(2, 0, 1)
                fp[index, :, :, :, :] = new_data
        end_toolbar()

    @ex.capture
    def predict(self, output_path, split, dataset, cluster_num):
        self.get_embedding(mode="test")
        f = open("{}/{}-kmeans-{}.pkl".format(output_path, split, "test"), "rb")
        x = pickle.load(f)
        x = x.detach().cpu().numpy()

        fitted_model = self.fitted_model
        test_label = fitted_model.predict(x)

        test_data = self.original_test_nan_data
        missing_joints = self.missing_joints_test

        fp = open_memmap(
            "{}/{}_completed_frame50/{}/val_position.npy".format(
                output_path, dataset, split
            ),
            dtype="float32",
            mode="w+",
            shape=(test_data.shape[0], 3, croped_size, V, M),
        )

        for i in range(cluster_num):
            print_toolbar(
                i * 1.0 / cluster_num,
                "({:>5}/{:<5}) Processing {}-{} missing joints data in {}th cluster: ".format(
                    i + 1, cluster_num, split, "val", i + 1
                ),
            )
            indexes = np.argwhere(test_label == i).reshape(-1)
            if len(indexes) == 0:
                continue
            X = self.original_test_nan_data[indexes].reshape(indexes.shape[0], -1)
            missing_joints = self.missing_joints_test[indexes]
            X = self.knn_models[i].transform(X)
            X = X.reshape(indexes.shape[0], C, -1, V, M)
            X = np.where(X == np.nan, 0, X)
            for i, index in enumerate(indexes):
                frame_length = len(missing_joints[i])
                dC, dT, dV, dM = X[i, :, :, :, :].shape
                center = X[i, :, :, 20, :].reshape([dC, dT, 1, dM])
                X[i, :, :, :, :] = X[i, :, :, :, :] - center
                new_data = real_resize(X[i, :, :, :, :], frame_length, croped_size)
                for n in range(2):
                    tmp_new, tmp_old = parallel_skeleton(new_data[:, :, :, n])
                    new_data[:, :, :, n] = tmp_new.transpose(2, 0, 1)
                fp[index, :, :, :, :] = new_data
        end_toolbar()

    @ex.capture
    def get_embedding(self, hidden_dim, output_path, split, mode: str):
        self.encoder.eval()
        if mode == "train":
            self.embeddings = torch.FloatTensor(
                len(self.dataset["complete_train"]), hidden_dim
            ).cuda()
        else:
            self.embeddings = torch.FloatTensor(
                len(self.dataset["complete_test"]), hidden_dim
            ).cuda()
        loader = self.data_loader["complete_{}".format(mode)]
        for index, data in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                embedding = self.encoder(data).squeeze()
                self.embeddings[index, :] = embedding
        with open("{}/{}-kmeans-{}.pkl".format(output_path, split, mode), "wb") as f:
            pickle.dump(self.embeddings, f)

    def start(self):
        self.Kmeans()
        self.predict()


@ex.automain
def main():
    p = Complete_Joints()
    p.start()
