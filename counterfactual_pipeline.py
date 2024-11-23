from functools import partial
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_parsing import ArgumentParser
from sae_auto_interp.autoencoders import AutoencoderConfig, AutoencoderLatents
from sae_auto_interp.config import CacheConfig, FeatureConfig, ExperimentConfig
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample

import torch
import torch.nn.functional as F
#from sae_auto_interp.counterfactuals.pipeline import main

from nnsight import LanguageModel
from simple_parsing import ArgumentParser
import torch
from sae_auto_interp.config import CacheConfig
import os
import numpy as np


l0_dict_mlp = {
    "16k": {0:50,1:56,2:33,3:55,4:66,5:46,6:46,7:47,8:55,9:40,10:49,11:34,12:42,13:40,14:41,15:45,16:37,17:41,
            18:36,19:38,20:41,21:34,22:34,23:73,24:32,25:72,26:57,27:52,28:50,29:49,30:51,31:43,32:44,33:48,
            34:47,35:46,36:47,37:53,38:45,39:43,40:37,41:58
    }
}
l0_dict_res = {
    "16k": {0:35,1:69,2:67,3:37,4:37,5:37,6:47,7:46,8:51,9:51,10:57,11:32,12:33,13:34,14:35,15:34,16:39,17:38,18:37,19:35,
            20:36,21:36,22:35,23:35,24:34,25:34,26:35,27:36,28:37,29:38,30:37,31:35,32:34,33:34,34:34,35:34,36:34,37:34,38:34,39:34,40:32,41:52
    },
    "131k": {0:30, 1:33, 2:36, 3:46, 4:51, 5:51, 6:66, 7:38, 8:41, 9:42, 10:47, 11:49, 12:52, 13:30, 14:56, 15:55, 16:35, 17:35, 18:34, 19:32, 
             20:34, 21:33, 22:32, 23:32, 24:55, 25:54, 26:32, 27:33, 28:32, 29:33, 30:32, 31:52, 32:51, 33:51, 34:51, 35:51, 36:51, 37:53, 38:53, 39:54, 40:49, 41:45
    }
}


# This function depends on the model and the autoencoder, so I put it here
def prepare_examples( record,model,ae,hookpoint):
    maximum_activation = record.max_activation
    latent_number = record.feature.feature_index
    cut_examples,activation_values = _prepare_examples(record.train)
    layer = int(hookpoint.split("_")[-1])
    clamp_tokens = []
    clean_tokens = []
    cut_text = []
    for example,activation_value in zip(cut_examples,activation_values):
        # clean
        with torch.no_grad():
            # with model.trace(example):
            #     if "mlp" in hookpoint:
            #         residual_stream = model.model.layers[layer].output.save()
            #     else:
            #         residual_stream = model.model.layers[layer].output[0].save()
            #     encoded_activations = ae.ae.encode(residual_stream[0,-1,:]).save()
            #     # this is to because sometimes the latent will not be active if the prompt is not the same size, this ensures that there is a maximum, and zero activation 
            #     encoded_activations[latent_number] = activation_value
            #     decoded_activations = ae.ae.decode(encoded_activations)
            #     if "mlp" in hookpoint:
            #         model.model.layers[layer].output[0,-1,:] = decoded_activations
            #     else:
            #         model.model.layers[layer].output[0][0,-1,:] = decoded_activations
            #     clean_output = model.output.save()
            
            #max_clamp
            with model.trace(example):
                if "mlp" in hookpoint:
                    residual_stream = model.model.layers[layer].output.save()
                else:
                    residual_stream = model.model.layers[layer].output[0].save()
                encoded_activations = ae.ae.encode(residual_stream[0,-1,:]).save()
                encoded_activations[latent_number] = maximum_activation
                decoded_activations = ae.ae.decode(encoded_activations)
                if "mlp" in hookpoint:
                    model.model.layers[layer].output[0,-1,:] = decoded_activations
                else:
                    model.model.layers[layer].output[0][0,-1,:] = decoded_activations
                max_clamp_output = model.output.save()
            
            #zero_clamp
            with model.trace(example):
                encoded_activations[latent_number] = 0
                decoded_activations = ae.ae.decode(encoded_activations)
                if "mlp" in hookpoint:
                    model.model.layers[layer].output[0,-1,:] = decoded_activations
                else:
                    model.model.layers[layer].output[0][0,-1,:] = decoded_activations
                zero_clamp_output = model.output.save()
            
            #clean_logits = clean_output.logits
            max_clamp_logits = max_clamp_output.logits
            zero_clamp_logits = zero_clamp_output.logits
            #values,top_100_logits = clean_logits[0,-1,:].topk(100)
            #diff_max_clamp = max_clamp_logits[0,-1,top_100_logits]-clean_logits[0,-1,top_100_logits]
            #diff_zero_clamp = clean_logits[0,-1,top_100_logits]-zero_clamp_logits[0,-1,top_100_logits]
            big_diff = max_clamp_logits[0,-1,:]-zero_clamp_logits[0,-1,:]
            # get top 5 tokens
            values,top_100_diff = big_diff.topk(100)
            decoded_big_100 = model.tokenizer.batch_decode(top_100_diff,skip_special_tokens=True)
            # max_clamp_tokens.append((decoded_max_20,values.tolist()))
            # values,top_20_diff = diff_zero_clamp.topk(20)
            # decoded_zero_20 = model.tokenizer.batch_decode(top_100_logits[top_20_diff],skip_special_tokens=True)
            # zero_clamp_tokens.append((decoded_zero_20,values.tolist()))
            # values,top_20_diff = big_diff.topk(20)
            # decoded_big_20 = model.tokenizer.batch_decode(top_100_logits[top_20_diff],skip_special_tokens=True)
            clamp_tokens.append((decoded_big_100,values.tolist()))
            cut_text.append(model.tokenizer.decode(example))
    logit_lens = model.lm_head.weight.data @ ae.ae.W_dec[latent_number,:]
    top10_values,top_10_tokens = logit_lens.topk(50)
    bottom10_values,bottom_10_tokens = (-logit_lens).topk(10)
    top10_decoded = model.tokenizer.batch_decode(top_10_tokens,skip_special_tokens=True)
    bottom10_decoded = model.tokenizer.batch_decode(bottom_10_tokens,skip_special_tokens=True)
    #print(top10_decoded)
    #print(bottom10_decoded)
    return cut_examples,cut_text,clamp_tokens
            

def _prepare_examples(examples):
    cut_examples = []
    activation_values = []
    for example in examples:
        activations = example.activations
        tokens = example.tokens
        # find the index of the max activation
        max_activation_idx = np.argmax(activations)
        cut = tokens[:max_activation_idx]
        if len(cut) > 16:
            cut_examples.append(cut)
            activation_values.append(activations[max_activation_idx])
    if len(cut_examples) < 10:
        print(f"Only {len(cut_examples)} examples were used because they were too short")
    return cut_examples,activation_values
  

def main(args):

    latents = args.latents
    layer = args.layer
    size = args.size
    model = args.model
    start_feature = args.start_feature
    n_features = args.n_features
    top = args.top
    assert latents in ["sae", "random"]
    subject_name = "google/gemma-2-9b"
    
    path_root = ""
    # save_dir = PATH_ROOT / f"counterfactual_results/{run_prefix}_{subject_name.split('/')[-1]}"
    # save_path = save_dir / "generations.json"
    # config_save_path = save_dir / "config.json"
    # save_dir.mkdir(parents=True, exist_ok=True)
    # if raise_if_exists:
    #     assert not (save_dir / "generations_scores.json").exists() or "debug" in run_prefix, f"Save path {save_path} already exists"
    # with open(config_save_path, "w") as f:
    #     json.dump(config, f)

    module = f".model.layers.{layer}"
    cache_dir = f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/{model}/{size}"
    if size == "16k":
        width = 16384
    elif size == "131k":
        width = 131072

    feature_dict = {f"{module}": torch.arange(start_feature, start_feature + n_features)}
    feature_cfg = FeatureConfig(width=width, n_splits=5, max_examples=100000, min_examples=200)
    if args.top:
        train_type = "top"
    else:   
        train_type = "quantiles"
    experiment_cfg = ExperimentConfig(n_random=0, example_ctx_len=32, n_quantiles=10, n_examples_test=0, n_examples_train=100, train_type=train_type, test_type="quantiles")

    dataset = FeatureDataset(
            raw_dir=cache_dir,
            cfg=feature_cfg,
            modules=[module],
            features=feature_dict,  # type: ignore
    )

    constructor=partial(
                default_constructor,
                tokens=dataset.tokens,  # type: ignore
                n_random=experiment_cfg.n_random, 
                ctx_len=experiment_cfg.example_ctx_len, 
                max_examples=feature_cfg.max_examples
            )

    sampler=partial(sample,cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    
    subject = LanguageModel("google/gemma-2-9b", device_map="cuda", dispatch=True,torch_dtype="float16")
    if latents == "sae":
        config = AutoencoderConfig(
            model_name_or_path="google/gemma-scope-9b-pt-res",
            autoencoder_type="CUSTOM",
            device="cuda",
            kwargs={"custom_name": "gemmascope"}
        )
        autoencoder = AutoencoderLatents.from_pretrained(config,hookpoint=f"layer_11/width_131k/average_l0_49")
        autoencoder.ae = autoencoder.ae.to(torch.float16)
    else:
        raise ValueError(f"Latents type {latents} not supported")
    
    for record in loader:
        cut_examples,cut_text,clamp_tokens = prepare_examples(record,subject,autoencoder,f"layer_11")
        change_values = {}
        big_change_values = {}
        token_counter = {}
        big_token_counter = {}
        counter = 0
        for sentence in clamp_tokens:
            for token,value in zip(sentence[0],sentence[1]):
                if token not in change_values:
                    change_values[token] = []
                    token_counter[token] = 0
                change_values[token].append(value)
                token_counter[token] += 1
            counter += 1
        # for sentence in zero_clamp_tokens:
        #     for token,value in zip(sentence[0],sentence[1]):
        #         if token not in change_values:
        #             change_values[token] = []
        #             token_counter[token] = 0
        #         change_values[token].append(value)
        #         token_counter[token] += 1
        # for sentence in big_clamp_tokens:
        #     for token,value in zip(sentence[0],sentence[1]):
        #         if token not in big_change_values:
        #             big_change_values[token] = []
        #             big_token_counter[token] = 0
        #         big_change_values[token].append(value)
        #         big_token_counter[token] += 1
        for token in change_values:
            if token_counter[token] > 1:
                change_values[token] = sum(change_values[token])/counter
            else:
                change_values[token] = 0
       
        # get the top 10 tokens with the highest change
        sorted_change_values = {token:value for token,value in sorted(change_values.items(),key=lambda x:abs(x[1]),reverse=True)}
        sorted_big_change_values = {token:value for token,value in sorted(big_change_values.items(),key=lambda x:abs(x[1]),reverse=True)}
        positive_change_values = {token:value for token,value in sorted_change_values.items() if value > 0}
        negative_change_values = {token:value for token,value in sorted_change_values.items() if value < 0}
        positive_big_change_values = {token:value for token,value in sorted_big_change_values.items() if value > 0}
        negative_big_change_values = {token:value for token,value in sorted_big_change_values.items() if value < 0}
        top_50_change_positive = list(positive_change_values.keys())[:25]
        top_50_change_negative = list(negative_change_values.keys())[:25]
        # top_50_values_positive = list(positive_change_values.values())[:50]
        # top_50_values_negative = list(negative_change_values.values())[:50]
        # top_50_big_change_positive = list(positive_big_change_values.keys())[:50]
        # top_50_big_change_negative = list(negative_big_change_values.keys())[:50]
        # top_50_big_values_positive = list(positive_big_change_values.values())[:50]
        # top_50_big_values_negative = list(negative_big_change_values.values())[:50]
        # get the top 10 more frequent
        #sorted_token_counter = {token:count for token,count in sorted(token_counter.items(),key=lambda x:x[1],reverse=True)}
        #top_20_frequent = list(sorted_token_counter.keys())[:20]
        print(top_50_change_positive)
        #print(top_10_values_positive)
        print(top_50_change_negative)
        #print(top_10_values_negative)
        #print(top_50_big_change_positive)
        #print(top_10_big_values_positive)
        #print(top_50_big_change_negative)
        #print(top_10_big_values_negative)

        # for text,clamp_token in zip(cut_text,clamp_tokens):
        #     print(text)
        #     print only if the change value is greater than change_value*0.5
        #     wanted_tokens = []
        #     wanted_values = []
        #     counter = 0
        #     for max_clamp_token,value in zip(max_clamp_token[0],max_clamp_token[1]):
        #         wanted_tokens_max.append(max_clamp_token)
        #         wanted_values_max.append(value)
        #         counter += 1
        #         if counter > 10:
        #             break
        #     counter = 0
        #     for zero_clamp_token,value in zip(zero_clamp_token[0],zero_clamp_token[1]):
        #         wanted_tokens_zero.append(zero_clamp_token)
        #         wanted_values_zero.append(value)
        #         counter += 1
        #         if counter > 10:
        #             break
        #     counter = 0
        #     for clamp_token,value in zip(clamp_token[0],clamp_token[1]):
        #         if clamp_token in top_50_change_positive:
        #             wanted_tokens.append(clamp_token)
        #             wanted_values.append(value)
        #             counter += 1
        #             if counter > 5:
        #                 break
        #     print(wanted_tokens)
        #     print(wanted_values)
        #     print("-"*100)
        break

    







if __name__ == "__main__":
    parser = ArgumentParser()
    #ctx len 256
    parser.add_argument("--latents", type=str, default="sae")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--size", type=str, default="131k")
    parser.add_argument("--model", type=str, default="gemma")
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--top", action="store_true")
    args = parser.parse_args()
    
    main(args)
