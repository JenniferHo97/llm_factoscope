import numpy as np
import h5py
import os
import torch
import ecco
import json

random_seed = 0

def append_to_hdf5(file_path, dataset_name, new_data):
    # new_data = np.expand_dims(new_data, axis=0)
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))


def load_dataset(model_name, data_name):
    correct_file_path = './features/{}/{}_dataset/correct_data.h5'.format(model_name, data_name)
    if not os.path.exists(correct_file_path):
        print(correct_file_path)
        return None
    with h5py.File(correct_file_path, 'r') as f:
        correct_data = f['correct_activation_values'][:]
        correct_rank = f['correct_final_output_rank'][:]
        correct_word_id_topk_rank = f['correct_word_id_topk_rank'][:]
        correct_topk_rank_prob = f['correct_topk_rank_prob'][:]

    false_file_path = './features/{}/{}_dataset/false_data.h5'.format(model_name, data_name)
    if not os.path.exists(false_file_path):
        print(false_file_path)
        return None
    with h5py.File(false_file_path, 'r') as f:
        false_data = f['false_activation_values'][:]
        false_rank = f['false_final_output_rank'][:]
        false_word_id_topk_rank = f['false_word_id_topk_rank'][:]
        false_topk_rank_prob = f['false_topk_rank_prob'][:]
        
    unrelative_file_path = './features/{}/{}_dataset/unrelative_data.h5'.format(model_name, data_name)
    if os.path.exists(unrelative_file_path):
        with h5py.File(unrelative_file_path, 'r') as f:
            unrelative_data = f['unrelative_activation_values'][:]
            unrelative_rank = f['unrelative_final_output_rank'][:]
            unrelative_word_id_topk_rank = f['unrelative_word_id_topk_rank'][:]
            unrelative_topk_rank_prob = f['unrelative_topk_rank_prob'][:]
    else:
        unrelative_data = []
        unrelative_rank = []
        unrelative_word_id_topk_rank = []
        unrelative_topk_rank_prob = []
            
    len_data = correct_data.shape[0]
    if len(false_data) > int(len_data / 2) and len(unrelative_data) > int(len_data / 2):
        random_indices_false = np.random.choice(false_data.shape[0], int(len_data / 2), replace=False)
        random_indices_unrelated = np.random.choice(unrelative_data.shape[0], int(len_data / 2), replace=False)
        false_data = false_data[random_indices_false]
        false_rank = false_rank[random_indices_false]
        false_word_id_topk_rank = false_word_id_topk_rank[random_indices_false]
        false_topk_rank_prob = false_topk_rank_prob[random_indices_false]
        unrelative_data = unrelative_data[random_indices_unrelated]
        unrelative_rank = unrelative_rank[random_indices_unrelated]
        unrelative_word_id_topk_rank = unrelative_word_id_topk_rank[random_indices_unrelated]
        unrelative_topk_rank_prob = unrelative_topk_rank_prob[random_indices_unrelated]
    elif len(false_data) > int(len_data / 2) and len(unrelative_data) <= int(len_data / 2):
        if false_data.shape[0] > (len_data-len(unrelative_data)):
            random_indices_false = np.random.choice(false_data.shape[0], len_data-len(unrelative_data), replace=False)
            false_data = false_data[random_indices_false]
            false_rank = false_rank[random_indices_false]
            false_word_id_topk_rank = false_word_id_topk_rank[random_indices_false]
            false_topk_rank_prob = false_topk_rank_prob[random_indices_false]
    elif len(false_data) <= int(len_data / 2) and len(unrelative_data) > int(len_data / 2):
        random_indices_unrelated = np.random.choice(unrelative_data.shape[0], len_data-len(false_data), replace=False)
        unrelative_data = unrelative_data[random_indices_unrelated]
        unrelative_rank = unrelative_rank[random_indices_unrelated]   
        unrelative_word_id_topk_rank = unrelative_word_id_topk_rank[random_indices_unrelated]
        unrelative_topk_rank_prob = unrelative_topk_rank_prob[random_indices_unrelated]
    
    if len(unrelative_data) != 0:
        false_data = np.concatenate((false_data, unrelative_data), axis=0)
        false_rank = np.concatenate((false_rank, unrelative_rank), axis=0)
        false_word_id_topk_rank = np.concatenate((false_word_id_topk_rank, unrelative_word_id_topk_rank), axis=0)
        false_topk_rank_prob = np.concatenate((false_topk_rank_prob, unrelative_topk_rank_prob), axis=0)
    return correct_data, correct_rank, correct_word_id_topk_rank, correct_topk_rank_prob, false_data, false_rank, false_word_id_topk_rank, false_topk_rank_prob

def load_data(model_name, data_name):
    correct_file_path = './features/{}/{}_dataset/correct_data.h5'.format(model_name, data_name)
    if not os.path.exists(correct_file_path):
        print(correct_file_path)
        return None
    with h5py.File(correct_file_path, 'r') as f:
        correct_data = f['correct_activation_values'][:]

    false_file_path = './features/{}/{}_dataset/false_data.h5'.format(model_name, data_name)
    if not os.path.exists(false_file_path):
        print(false_file_path)
        return None
    with h5py.File(false_file_path, 'r') as f:
        false_data = f['false_activation_values'][:]
        
    unrelative_file_path = './features/{}/{}_dataset/unrelative_data.h5'.format(model_name, data_name)
    if os.path.exists(unrelative_file_path):
        with h5py.File(unrelative_file_path, 'r') as f:
            unrelative_data = f['unrelative_activation_values'][:]
    else:
        unrelative_data = []
            
    len_data = correct_data.shape[0]
    if len(false_data) > int(len_data / 2) and len(unrelative_data) > int(len_data / 2):
        random_indices_false = np.random.choice(false_data.shape[0], int(len_data / 2), replace=False)
        random_indices_unrelated = np.random.choice(unrelative_data.shape[0], int(len_data / 2), replace=False)
        false_data = false_data[random_indices_false]
        unrelative_data = unrelative_data[random_indices_unrelated]
    elif len(false_data) > int(len_data / 2) and len(unrelative_data) <= int(len_data / 2):
        if false_data.shape[0] > len_data-len(unrelative_data):
            random_indices_false = np.random.choice(false_data.shape[0], len_data-len(unrelative_data), replace=False)
            false_data = false_data[random_indices_false]
    elif len(false_data) <= int(len_data / 2) and len(unrelative_data) > int(len_data / 2):
        random_indices_unrelated = np.random.choice(unrelative_data.shape[0], len_data-len(false_data), replace=False)
        unrelative_data = unrelative_data[random_indices_unrelated]
    
    if len(unrelative_data) != 0:
        false_data = np.concatenate((false_data, unrelative_data), axis=0)
    return correct_data, false_data

def process_activation_data(all_data, mean, std):
    mean = np.mean(all_data)
    std = np.std(all_data)
    all_data = (all_data - mean) / std
    return all_data

def process_rank_data(all_rank):
    a = -1
    all_rank = 1 / (a * (all_rank - 1) + 1 + 1e-7)
    return all_rank

def process_word_id_topk_rank_data(all_word_id_topk_rank, model_emb, file_path):
    batch, layer, n_words = all_word_id_topk_rank.shape
    print(batch, layer, n_words)
    data_distane = None
    
    for b in range(batch):
        layer_distance = torch.zeros((1, all_word_id_topk_rank.shape[-1]))
        for l in range(layer-1):
            words0 = all_word_id_topk_rank[b, l, :]
            words1 = all_word_id_topk_rank[b, l+1, :]
            words0 = torch.tensor(words0).unsqueeze(0)
            words1 = torch.tensor(words1).unsqueeze(0)
            
            emb0 = model_emb(words0)
            emb1 = model_emb(words1)
            
            distances = torch.cosine_similarity(emb0, emb1, dim=2)
            
            if layer_distance is None:
                layer_distance = distances
            else:
                layer_distance = torch.cat((layer_distance, distances), dim=0)
        
        append_to_hdf5(file_path, 'all_word_id_topk_rank', layer_distance.unsqueeze(0).detach().cpu().numpy())  
  
def save_processed_data(file_path, data_dict):
    np.savez(file_path, **data_dict)
    print(f"Data successfully saved to '{file_path}'")

def main():
    np.random.seed(random_seed)
    # load dataset
    dataset_name = ['movie_name_writer', 'year_olympicscity', 'athlete_country', 'artwork_artist', 'city_country', 'athlete_sport', 'book_author', 'final_company', 'final_nobel_birth_country', 'final_nobel_category', 'final_song_artist', 'movie_name_director', 'river_country', 'final_nobel_year', 'movie_name_year', 'pantheon_country', 'pantheon_occupation', 'multi']

    llama2_model_config = {    
        'embedding': "model.embed_tokens.weight",
        'type': 'causal',
        'activations': ['mlp\.act_fn'],
        'token_prefix': 'Ġ',
        'partial_token_prefix': ''
    }

    MODELS_DIR = "./llm_models"
    model_name = "Llama-2-7b-hf"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
    lm = ecco.from_pretrained(os.path.join(MODELS_DIR, model_name), model_config=llama2_model_config, activations=False)
    lm.model.config._name_or_path = model_name

    # calculate mean
    sum_val = 0
    counter = 0
    for current_dataset_name in dataset_name:
        correct_data, false_data = load_data(model_name, current_dataset_name)
        sum_val += np.sum(correct_data)
        sum_val += np.sum(false_data)
        counter += correct_data.shape[0] * correct_data.shape[1] * correct_data.shape[2]
        counter += false_data.shape[0] * false_data.shape[1] * false_data.shape[2]
    mean = sum_val / counter
    
    # calculate std
    sum_val = 0
    for current_dataset_name in dataset_name:
        correct_data, false_data = load_data(model_name, current_dataset_name)
        sum_val += np.sum(np.square(correct_data - mean))
        sum_val += np.sum(np.square(false_data - mean))
    std = np.sqrt(sum_val / counter)
    print('mean: {}, std: {}'.format(mean, std))
    # save mean and std in json file
    with open('./features/{}/mean_std.json'.format(model_name), 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    for current_dataset_name in dataset_name:
        file_path = './features/{}/all_data_{}.h5'.format(model_name, current_dataset_name)
        correct_data, correct_rank, correct_word_id_topk_rank, correct_topk_rank_prob, false_data, false_rank, false_word_id_topk_rank, false_topk_rank_prob = load_dataset(model_name, current_dataset_name)
        print('[length] dataset: {}, correct: {}, false: {}'.format(current_dataset_name, correct_data.shape[0], false_data.shape[0]))
        # if current_dataset_name == dataset_name[0]:
        correct_all_data = correct_data
        false_all_data = false_data
        correct_all_rank = correct_rank
        false_all_rank = false_rank
        correct_all_word_id_topk_rank = correct_word_id_topk_rank
        false_all_word_id_topk_rank = false_word_id_topk_rank
        correct_all_topk_rank_prob = correct_topk_rank_prob
        false_all_topk_rank_prob = false_topk_rank_prob

        all_data = np.concatenate((correct_all_data, false_all_data), axis=0)
        all_rank = np.concatenate((correct_all_rank, false_all_rank), axis=0)
        all_word_id_topk_rank = np.concatenate((correct_all_word_id_topk_rank, false_all_word_id_topk_rank), axis=0)
        all_topk_rank_prob = np.concatenate((correct_all_topk_rank_prob, false_all_topk_rank_prob), axis=0)
        all_label = np.concatenate((np.ones(correct_all_data.shape[0]), np.zeros(false_all_data.shape[0])), axis=0)
        
        all_data = process_activation_data(all_data, mean, std)
        all_rank = process_rank_data(all_rank)
        append_to_hdf5(file_path, 'all_activation_values', all_data)
        append_to_hdf5(file_path, 'all_final_output_rank', all_rank)
        append_to_hdf5(file_path, 'all_topk_rank_prob', all_topk_rank_prob)
        del all_data
        del all_rank
        del all_topk_rank_prob
        
        if model_name == "gpt2-xl":
            process_word_id_topk_rank_data(all_word_id_topk_rank, lm.model.transformer.wte, file_path)
        else:
            process_word_id_topk_rank_data(all_word_id_topk_rank, lm.model.model.embed_tokens.cpu(), file_path)
        
        append_to_hdf5(file_path, 'all_label', all_label)
        del all_word_id_topk_rank  

if __name__ == "__main__":
    main()
