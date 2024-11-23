from nnsight import LanguageModel
from sae_auto_interp.autoencoders import AutoencoderConfig, AutoencoderLatents
import torch

def generate_with_steering(model, query, maximum_activation, layer, latent_number, autoencoder,number_tokens):
    all_encoded_activations = []
    with model.generate(query, max_new_tokens=number_tokens,temperature=1,do_sample=True):
        for i in range(number_tokens):
            residual_stream = model.model.layers[layer].output[0].save()
            encoded_activations = autoencoder.ae.encode(residual_stream[0,-1,:]).save()
            all_encoded_activations.append(encoded_activations)
            encoded_activations[latent_number] = maximum_activation
            decoded_activations = autoencoder.ae.decode(encoded_activations)
            model.model.layers[layer].output[0][0,-1,:] = decoded_activations
            model.model.layers[layer].next()
        out = model.generator.output.save()
    text = model.tokenizer.decode(out[0])
    print(text)
    return text, all_encoded_activations

model = LanguageModel("google/gemma-2-9b", device_map="cuda", dispatch=True,torch_dtype="float16")
config = AutoencoderConfig(
    model_name_or_path="google/gemma-scope-9b-pt-res",
    autoencoder_type="CUSTOM",
    device="cuda",
    kwargs={"custom_name": "gemmascope"}
)
layer = 11
autoencoder = AutoencoderLatents.from_pretrained(config,hookpoint=f"layer_11/width_131k/average_l0_49")
autoencoder.ae = autoencoder.ae.to(torch.float16)
query = "I'm thinking about"
#query = "Tell me a calm story"
#prompt = [{"role": "user", "content": query}]

#tokenized_prompt = model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
latent_number = 99
_,_ = generate_with_steering(model, query, 250, layer, latent_number, autoencoder,30)
_,_ = generate_with_steering(model, query, 0, layer, latent_number, autoencoder,20)