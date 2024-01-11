import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

import numpy as np
import random
import h5py
from torchvision import models
from sklearn.metrics import confusion_matrix
import argparse
import os


class GRUNet(nn.Module):
    def __init__(self, emb_dim, input_dim=1, hidden_dim1=128, hidden_dim2=64, feature_dim=32, dropout=0.5):
        super(GRUNet, self).__init__()

        # First GRU layer
        self.gru1 = nn.GRU(input_size=input_dim,
                           hidden_size=hidden_dim1, batch_first=True)

        # Second GRU layer
        self.gru2 = nn.GRU(input_size=hidden_dim1,
                           hidden_size=hidden_dim2, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Dense layers for feature extraction
        self.fc1 = nn.Linear(hidden_dim2, feature_dim)
        self.fc2 = nn.Linear(feature_dim, emb_dim)  # This is the feature layer

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        # We take the output from the last time step of the GRU
        x = self.dropout(x[:, -1, :])
        x = self.fc1(x)
        x = self.fc2(x)  # This is the output from the feature layer

        # x = F.softmax(x, dim=1)

        return x


class TriDataset(data.Dataset):
    """Custom Dataset for loading and processing the data."""

    def __init__(self, features, labels, source, transform=None):
        self.features = features
        self.labels = labels
        self.source = source
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        anchor = self.features[index]
        anchor_label = self.labels[index]

        # positive
        pos_index = random.randint(0, len(self.labels) - 1)
        while self.labels[pos_index] != anchor_label:
            pos_index = random.randint(0, len(self.labels) - 1)
        positive = self.features[pos_index]
        positive_label = self.labels[pos_index]

        # negative
        neg_index = random.randint(0, len(self.labels) - 1)
        while self.labels[neg_index] == anchor_label:
            neg_index = random.randint(0, len(self.labels) - 1)
        negative = self.features[neg_index]
        negative_label = self.labels[neg_index]

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return torch.from_numpy(anchor).float(), \
            torch.from_numpy(positive).float(), \
            torch.from_numpy(negative).float(), \
            torch.tensor(anchor_label, dtype=torch.long), \
            torch.tensor(positive_label, dtype=torch.long), \
            torch.tensor(negative_label, dtype=torch.long)

    def get_source(self):
        return self.source


class CombinedTriNet(nn.Module):
    def __init__(self, act, grunet, embdistance, prob, emd_dim, feature_dim=64):
        super(CombinedTriNet, self).__init__()
        self.act = act
        self.grunet = grunet
        self.embdistance = embdistance
        self.prob = prob

        self.fc1 = nn.Linear(emd_dim*4, feature_dim)

    def forward(self, x, act_dim):

        x_activation, x_rank, x_embdis, x_prob = x[:, :, :, :act_dim], x[:, :, :, act_dim:(
            act_dim+1)], x[:, :, :, (act_dim+1):(act_dim+11)], x[:, :, :, (act_dim+11):]

        x1 = self.act(x_activation)
        x2 = self.grunet(x_rank)
        x3 = self.embdistance(x_embdis)
        x4 = self.prob(x_prob)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.fc1(x))
        embedding = F.normalize(x, p=2, dim=1)

        return embedding


def train_and_evaluate_model(model, train_loader, test_loader, support_loader, save_path, act_dim, squeeze_dim=1, epochs=30):

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    highest_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (anchor, positive, negative, _, _, _) in enumerate(train_loader):

            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            positive = positive.unsqueeze(squeeze_dim).cuda()
            negative = negative.unsqueeze(squeeze_dim).cuda()

            optimizer.zero_grad()
            anchor_embedding = model(anchor, act_dim)
            positive_embedding = model(positive, act_dim)
            negative_embedding = model(negative, act_dim)
            loss = criterion(anchor_embedding,
                             positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
        test_accuracy = test_model(model, test_loader, support_loader, act_dim, squeeze_dim=1)
        if test_accuracy > highest_acc:
            highest_acc = test_accuracy
            torch.save(model, save_path)

def test_model(model, test_loader, support_loader, act_dim, squeeze_dim=1):

    model.eval()
    correct = 0
    y_pred = []
    y_true = []

    support_set_labels = []
    support_set_output = []
    with torch.no_grad():
        for i, (support_data, _, _, support_label, _, _) in enumerate(support_loader):
            support_data = support_data.unsqueeze(squeeze_dim).cuda()
            if i == 0:
                support_set_output = model(support_data, act_dim)
                support_set_labels = support_label
            else:
                support_set_output = torch.cat(
                    (support_set_output, model(support_data, act_dim)), dim=0)
                support_set_labels = torch.cat(
                    (support_set_labels, support_label), dim=0)

    # compare the distance between the embedding of the test image and the embedding of the support set
    with torch.no_grad():
        for i, (anchor, _, _, anchor_label, _, _) in enumerate(test_loader):
            anchor = anchor.unsqueeze(squeeze_dim).cuda()
            anchor_embedding = model(anchor, act_dim)
            anchor_embedding = anchor_embedding.squeeze()
            dist = F.pairwise_distance(
                anchor_embedding, support_set_output, p=2)
            pred = support_set_labels[torch.argmin(dist, -1)]
            y_pred.append(int(pred))
            y_true.append(int(anchor_label))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = conf_matrix.ravel()
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print(f"True Positives: {TP / (TP + FN)}, False Positives: {FP / (FP + TN)}, True Negatives: {TN / (TN + FP)}, False Negatives: {FN / (TP + FN)}")
    print(f"Accuracy: {accuracy}")
    return accuracy

def load_data(dataset_name, model_name='gpt2-xl'):
    largest_data_num = {'gpt2-xl': 10000, 'Llama-2-7b-hf': 10000, 'vicuna-7b-v1.5': 10000,
                        'stablelm-tuned-alpha-7b': 10000, 'Llama-2-13b-chat-hf': 8000, 'vicuna-13b-v1.5': 8000}
    for i, current_dataset_name in enumerate(dataset_name):
        print('./features/{}/all_data_{}.h5'.format(model_name, current_dataset_name))
        with h5py.File('./features/{}/all_data_{}.h5'.format(model_name, current_dataset_name), 'r') as f:
            activation_values = torch.tensor(
                f['all_activation_values'][:].astype(np.float32))
            final_output_rank = torch.tensor(
                f['all_final_output_rank'][:].astype(np.float32))
            word_id_topk_rank = torch.tensor(
                f['all_word_id_topk_rank'][:].astype(np.float32))
            topk_rank_prob = torch.tensor(
                f['all_topk_rank_prob'][:].astype(np.float32))
            label = torch.tensor(f['all_label'][:])
            source = torch.ones_like(label) * i
            print(activation_values.shape)
            if current_dataset_name == dataset_name[0]:
                if activation_values.shape[0] > largest_data_num[model_name]:
                    all_activation_values = torch.cat((activation_values[:int(
                        largest_data_num[model_name]/2)], activation_values[-int(largest_data_num[model_name]/2):]), dim=0)
                    print(all_activation_values.shape[0])
                    all_final_output_rank = torch.cat((final_output_rank[:int(
                        largest_data_num[model_name]/2)], final_output_rank[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_word_id_topk_rank = torch.cat((word_id_topk_rank[:int(
                        largest_data_num[model_name]/2)], word_id_topk_rank[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_topk_rank_prob = torch.cat((topk_rank_prob[:int(
                        largest_data_num[model_name]/2)], topk_rank_prob[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_label = torch.cat((label[:int(
                        largest_data_num[model_name]/2)], label[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_source = torch.cat((source[:int(
                        largest_data_num[model_name]/2)], source[-int(largest_data_num[model_name]/2):]), dim=0)
                else:
                    all_activation_values = activation_values
                    print(all_activation_values.shape[0])
                    all_final_output_rank = final_output_rank
                    all_word_id_topk_rank = word_id_topk_rank
                    all_topk_rank_prob = topk_rank_prob
                    all_label = label
                    all_source = source
            else:
                if activation_values.shape[0] > largest_data_num[model_name]:
                    all_activation_values = torch.cat((all_activation_values, activation_values[:int(
                        largest_data_num[model_name]/2)], activation_values[-int(largest_data_num[model_name]/2):]), dim=0)
                    print(all_activation_values.shape[0])
                    all_final_output_rank = torch.cat((all_final_output_rank, final_output_rank[:int(
                        largest_data_num[model_name]/2)], final_output_rank[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_word_id_topk_rank = torch.cat((all_word_id_topk_rank, word_id_topk_rank[:int(
                        largest_data_num[model_name]/2)], word_id_topk_rank[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_topk_rank_prob = torch.cat((all_topk_rank_prob, topk_rank_prob[:int(
                        largest_data_num[model_name]/2)], topk_rank_prob[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_label = torch.cat((all_label, label[:int(
                        largest_data_num[model_name]/2)], label[-int(largest_data_num[model_name]/2):]), dim=0)
                    all_source = torch.cat((all_source, source[:int(
                        largest_data_num[model_name]/2)], source[-int(largest_data_num[model_name]/2):]), dim=0)
                else:
                    all_activation_values = torch.cat(
                        (all_activation_values, activation_values), dim=0)
                    print(all_activation_values.shape[0])
                    all_final_output_rank = torch.cat(
                        (all_final_output_rank, final_output_rank), dim=0)
                    all_word_id_topk_rank = torch.cat(
                        (all_word_id_topk_rank, word_id_topk_rank), dim=0)
                    all_topk_rank_prob = torch.cat(
                        (all_topk_rank_prob, topk_rank_prob), dim=0)
                    all_label = torch.cat((all_label, label), dim=0)
                    all_source = torch.cat((all_source, source), dim=0)

    all_final_output_rank = all_final_output_rank[:, :, np.newaxis]

    # all_data = all_activation_values
    # print(all_data.shape[0])

    return all_activation_values, all_final_output_rank, all_word_id_topk_rank, all_topk_rank_prob, all_label, all_source


def split_set(data, label, source, support_size, ratio=[0.8, 0.2]):

    all_dataset = TriDataset(data, label, source)
    train_size = int(data.shape[0] * ratio[0])
    test_size = data.shape[0] - train_size - support_size
    print('train size: {}, test size: {}, support size: {}'.format(
        train_size, test_size, support_size))
    train_data, test_data, support_data = torch.utils.data.random_split(
        all_dataset, [train_size, test_size, support_size])

    return train_data, test_data, support_data


def main(model_name, split_list, support_size, emb_dim=24):
    random_seed = 0

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    dataset_name_dict = {'gpt2-xl': ['year_olympicscity', 'city_country', 'athlete_sport', 
                    'final_company', 'final_nobel_birth_country', 'movie_name_director', 
                    'river_country', 'final_nobel_year', 'movie_name_year', 'multi', 'movie_name_writer', 'book_author', 'final_nobel_category', 'pantheon_country', 'pantheon_occupation', 'athlete_country'], 'Llama-2-7b-hf': ['athlete_country', 'artwork_artist', 'city_country', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'movie_name_director', 'river_country', 'final_nobel_year', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi'], 'vicuna-7b-v1.5': ['year_olympicscity', 'city_country', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'movie_name_director', 'river_country', 'final_nobel_year', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi', 'athlete_country', 'movie_name_writer'], 'Llama-2-13b-chat-hf': ['athlete_country', 'artwork_artist', 'city_country', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'movie_name_director', 'river_country', 'final_nobel_year', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi'], 'stablelm-tuned-alpha-7b': ['artwork_artist', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'river_country', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi'], 'vicuna-13b-v1.5':['year_olympicscity', 'city_country', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'movie_name_director', 'river_country', 'final_nobel_year', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi', 'athlete_country', 'movie_name_writer']
                    }
    act_dim_dict = {'gpt2-xl': 6400, 'Llama-2-7b-hf': 11008, 'vicuna-7b-v1.5': 11008,
                    'Llama-2-13b-chat-hf': 13824, 'vicuna-13b-v1.5': 13824, 'stablelm-tuned-alpha-7b': 24576}

    act_dim = act_dim_dict[model_name]
    dataset_name = dataset_name_dict[model_name]

    print("In distribution dataset: ", dataset_name)

    if not os.path.exists('./features/{}/combined_trinet_test.pth'.format(model_name)):
        all_activation_values, all_final_output_rank, all_word_id_topk_rank, all_topk_rank_prob, all_label, all_source = load_data(
            dataset_name, model_name)
        all_data = np.concatenate((all_activation_values, all_final_output_rank,
                                   all_word_id_topk_rank, all_topk_rank_prob), axis=2)

        train_data, test_data, support_data = split_set(
            all_data, all_label, all_source, support_size, split_list)
        train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
        support_loader = data.DataLoader(
            support_data, batch_size=64, shuffle=False)
        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)

        with h5py.File('./features/{}/test_support_data.h5'.format(model_name), 'w') as f:
            f.create_dataset('test_data_features', data=test_data.dataset.features[test_data.indices])
            f.create_dataset('test_data_label', data=test_data.dataset.labels[test_data.indices])
            f.create_dataset('test_data_source', data=test_data.dataset.source[test_data.indices])
            f.create_dataset('support_data_features', data=support_data.dataset.features[support_data.indices])
            f.create_dataset('support_data_label', data=support_data.dataset.labels[support_data.indices])
            f.create_dataset('support_data_source', data=support_data.dataset.source[support_data.indices])

        # init model
        act_resnet_model = models.resnet18(
            pretrained=False, num_classes=emb_dim).cuda()
        act_resnet_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()

        grunet_model = GRUNet(emb_dim=emb_dim).train().cuda()

        emb_dist_resnet_model = models.resnet18(
            pretrained=False, num_classes=emb_dim).train().cuda()
        emb_dist_resnet_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()

        prob_resnet_model = models.resnet18(
            pretrained=False, num_classes=emb_dim).train().cuda()
        prob_resnet_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()

        combined_model = CombinedTriNet(
            act_resnet_model, grunet_model, emb_dist_resnet_model, prob_resnet_model, emb_dim).cuda()

        # train the model
        train_and_evaluate_model(combined_model, train_loader, test_loader,
                                 support_loader, './features/{}/combined_trinet_test.pth'.format(model_name), act_dim)
    else:
        with h5py.File('./features/{}/test_support_data.h5'.format(model_name), 'r') as f:
            test_data_features = f['test_data_features'][:]
            test_data_label = f['test_data_label'][:]
            test_data_source = f['test_data_source'][:]
            support_data_features = f['support_data_features'][:]
            support_data_label = f['support_data_label'][:]
            support_data_source = f['support_data_source'][:]
        test_data = TriDataset(test_data_features, test_data_label, test_data_source)
        support_data = TriDataset(support_data_features, support_data_label, support_data_source)
        test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
        support_loader = data.DataLoader(
            support_data, batch_size=64, shuffle=False)
        combined_model = torch.load('./features/{}/combined_trinet_test.pth'.format(model_name))

        test_model(combined_model, test_loader,
                   support_loader, act_dim, squeeze_dim=1)

    for num in range(len(dataset_name)):
        print(dataset_name[num])
        current_index = (test_data.source == num)
        print(np.where(current_index)[0].shape)
        current_test_data = data.Subset(
            test_data, np.where(current_index)[0])
        current_test_loader = data.DataLoader(
            current_test_data, batch_size=1, shuffle=False)
        test_model(combined_model, current_test_loader,
                    support_loader, act_dim, squeeze_dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and Evaluate Neural Network')
    parser.add_argument('--model_name', type=str,
                        default='gpt2-xl', help='Name of the model to use')
    parser.add_argument('--split_list', type=float, nargs='+',
                        default=[0.8, 0.2], help='List of splits for the dataset')
    parser.add_argument('--support_size', type=int,
                        default=500, help='Size of the support set')

    args = parser.parse_args()
    main(args.model_name, args.split_list, args.support_size)
