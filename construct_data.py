import warnings
import ecco
import os
from transformers import set_seed
import torch
import json
import numpy as np
import re
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')

def append_to_hdf5(file_path, dataset_name, new_data):
    new_data = np.expand_dims(new_data, axis=0)
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))
            
if __name__ == '__main__':
    llama2_model_config = {    
        'embedding': "model.embed_tokens.weight",
        'type': 'causal',
        'activations': ['mlp\.act_fn'],
        'token_prefix': 'Ġ',
        'partial_token_prefix': ''
    }

    MODELS_DIR = ""
    model_name = "Llama-2-7b-hf" 
    lm = ecco.from_pretrained(os.path.join(MODELS_DIR, model_name), model_config=llama2_model_config, activations=True)
    lm.model.config._name_or_path = model_name

    file_paths = []
    for root, dirs, files in os.walk("./dataset_train"):
        for file in files:
            if file.endswith(".json"):
                file_paths.append(os.path.join(root, file))

    file_name = []
    for file_path in file_paths:
        file_name.append(file_path.split('/')[-1][:-13])
        
    topk = 10

    correct_counter = 0
    correct_prompt = []
    unrelative_counter = 0
    unrelative_prompt = []
    false_counter = 0
    false_prompt = []

    for file_index, current_file_path in enumerate(file_paths):
        if not os.path.exists("./features/{}/{}_dataset".format(model_name,file_name[file_index])):
            os.makedirs("./features/{}/{}_dataset".format(model_name,file_name[file_index]))
        total_correct = 0
        correct_prompt = []
        unrelative_prompt = []
        false_prompt = []
        # Load the JSON data
        with open(current_file_path, 'r') as file:
            data = json.load(file)
    
        for i, entry in enumerate(data):
            if total_correct >= 10000:
                break
            # if file_index != (len(file_paths) - 1):
            input_text = entry['prompt']  
            if isinstance(entry['answer'], str) is False:
                continue
            if len(entry['answer'].split(' ')) > 1:
                target = entry['answer'].split(' ')[0]
            else:      
                target = entry['answer']

            index = entry['index']
            
            source = current_file_path.split('/')[-1]
            len_input = len(input_text.split(' '))
            output = lm.generate(input_text, generate=10, do_sample=False, output_hidden_states=True)
            if model_name == "gpt2-xl":
                if len(output.output_text.split(' ')) < (len_input + 1):
                    continue
                first_token = output.output_text.split(' ')[len_input]
            else:
                if len(output.output_text.split(' ')) < (len_input + 2):
                    continue
                first_token = output.output_text.split(' ')[len_input + 1]
            if re.search('\.|\n|\,|\?|\!|\;|\:', first_token) is not None:
                first_token = first_token[:re.search('\.|\n|\,|\?|\!|\;|\:', first_token).span()[0]]
                if re.search('\.|\n|\,|\?|\!|\;|\:', target) is not None:
                    target = target[:re.search('\.|\n|\,|\?|\!|\;|\:', target).span()[0]]
            new_data = {
                'index': index,
                'prompt': input_text,
                'answer': target,
                'output_first_token': first_token,
                'source': source
            }
            ranking_data = output.rankings(inputs=input_text, printJson=True)
            hidden_states = output.decoder_hidden_states[0][:, -1, :]
            
            logits = torch.softmax(output.lm_head(output.to(hidden_states)), dim=-1)
            # Sort by score (ascending)
            sorted, indices = torch.sort(logits)
            location = indices[:, -topk:]
            prob = sorted[:, -topk:]
            
            first_token = first_token.lower()
            target = target.lower()
            if first_token == target:
                correct_counter += 1    
                total_correct += 1
                correct_prompt.append(new_data)
                with open("./features/{}/{}_dataset/correct_data.json".format(model_name,file_name[file_index]), "w", encoding="utf-8") as f:
                    json.dump(correct_prompt, f, ensure_ascii=False, indent=4)
                append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,file_name[file_index]), 'correct_activation_values', output.activations['decoder'][0,:,:,output.n_input_tokens])
                append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,file_name[file_index]), 'correct_final_output_rank', ranking_data['rankings'][:,0])
                append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,file_name[file_index]), 'correct_word_id_topk_rank', location.cpu().detach().numpy())
                append_to_hdf5('./features/{}/{}_dataset/correct_data.h5'.format(model_name,file_name[file_index]), 'correct_topk_rank_prob', prob.cpu().detach().numpy())
            elif first_token != target and first_token != 'the' and first_token != 'a' and first_token != 'an' and first_token != 'this' and first_token != 'that' and first_token != 'these' and first_token != 'those' and first_token != 'my' and first_token != 'your' and first_token != 'his' and first_token != 'her' and first_token != 'its' and first_token != 'our' and first_token != 'their' and first_token != 'few' and first_token != 'little' and first_token != 'much' and first_token != 'many' and first_token != 'lot' and first_token != 'most' and first_token != 'some' and first_token != 'any' and first_token != 'enough' and first_token != 'all' and first_token != 'both' and first_token != 'half' and first_token != 'either' and first_token != 'neither' and first_token != 'each' and first_token != 'every' and first_token != 'other' and first_token != 'another' and first_token != 'such' and first_token != 'what' and first_token != 'rather' and first_token != 'quite' and first_token != 'same' and first_token != 'different' and first_token != 'such' and first_token != 'when' and first_token != 'while' and first_token != 'who' and first_token != 'whom' and first_token != 'which' and first_token != 'where' and first_token != 'why' and first_token != 'how' and first_token != 'i' and first_token != 'you' and first_token != 'he' and first_token != 'she' and first_token != 'it' and first_token != 'we' and first_token != 'they' and first_token != 'me' and first_token != 'him' and first_token != 'her' and first_token != 'us' and first_token != 'them' and first_token != 'myself' and first_token != 'yourself' and first_token != 'himself' and first_token != 'herself' and first_token != 'itself' and first_token != 'ourselves' and first_token != 'themselves' and first_token != 'to' and first_token != 'of' and first_token != 'not' and first_token != 'at' and first_token != '"':
                false_counter += 1
                false_prompt.append(new_data)
                with open("./features/{}/{}_dataset/false_data.json".format(model_name,file_name[file_index]), "w", encoding="utf-8") as f:
                        json.dump(false_prompt, f, ensure_ascii=False, indent=4)
                append_to_hdf5('./features/{}/{}_dataset/false_data.h5'.format(model_name,file_name[file_index]), 'false_activation_values', output.activations['decoder'][0,:,:,output.n_input_tokens])
                append_to_hdf5('./features/{}/{}_dataset/false_data.h5'.format(model_name,file_name[file_index]), 'false_final_output_rank', ranking_data['rankings'][:,0])
                append_to_hdf5('./features/{}/{}_dataset/false_data.h5'.format(model_name,file_name[file_index]), 'false_word_id_topk_rank', location.cpu().detach().numpy())
                append_to_hdf5('./features/{}/{}_dataset/false_data.h5'.format(model_name,file_name[file_index]), 'false_topk_rank_prob', prob.cpu().detach().numpy())
            else:
                unrelative_counter += 1    
                unrelative_prompt.append(new_data)
                with open("./features/{}/{}_dataset/unrelative_data.json".format(model_name,file_name[file_index]), "w", encoding="utf-8") as f:
                        json.dump(unrelative_prompt, f, ensure_ascii=False, indent=4)
                append_to_hdf5('./features/{}/{}_dataset/unrelative_data.h5'.format(model_name,file_name[file_index]), 'unrelative_activation_values', output.activations['decoder'][0,:,:,output.n_input_tokens])
                append_to_hdf5('./features/{}/{}_dataset/unrelative_data.h5'.format(model_name,file_name[file_index]), 'unrelative_final_output_rank', ranking_data['rankings'][:,0])
                append_to_hdf5('./features/{}/{}_dataset/unrelative_data.h5'.format(model_name,file_name[file_index]), 'unrelative_word_id_topk_rank', location.cpu().detach().numpy())
                append_to_hdf5('./features/{}/{}_dataset/unrelative_data.h5'.format(model_name,file_name[file_index]), 'unrelative_topk_rank_prob', prob.cpu().detach().numpy())
                

    print(correct_counter)
    print(unrelative_counter)
    print(false_counter)
