import copy
import json
import time

import torch
from transformers.generation import GenerationConfig
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen_fei import QWenLMHeadModel
from .modeling_Mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .choices import mc_sim_7b_63
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download




class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(self.base_model_name_or_path, trust_remote_code=True)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.transformer.h[-1].attn.c_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path,trust_remote_code=True).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=="QWenLMHeadModel":
            base_model = QWenLMHeadModel.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            logits_processor=None,
            **kwargs,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            forward_transformer_start_time = time.perf_counter()
            outputs = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                return_dict=True,
                use_cache=True)
            forward_transformer_end_time = time.perf_counter()
            # print(f"forward_transformer time cost: {forward_transformer_end_time - forward_transformer_start_time:.6f} s")
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                next_token_logits = orig[:, -1]
                next_token_scores = logits_processor(None, next_token_logits)
                prob = torch.nn.functional.softmax(next_token_scores, dim=1)
                next_tokens = torch.multinomial(prob, 1)
            else:
                next_tokens = torch.argmax(orig[:, -1])
                next_tokens = next_tokens[None, None]
            input_ids = torch.cat((input_ids, next_tokens.to(input_ids.device)), dim=1)
            # # Clone the output hidden states
            # attention_mask = torch.cat(
            # [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            ea_logits = self.ea_layer.topK_genrate(hidden_states,
                                                   input_ids,
                                                   self.base_model.lm_head,
                                                   logits_processor)
            if output_orig:
                return ea_logits, outputs, orig, hidden_states, next_tokens
            return ea_logits, hidden_states, next_tokens
        else:
            if output_orig:
                return outputs, orig, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            tokenizer, 
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            tree_choices=mc_sim_7b_63,
            verbose=False
    ):
        prepare_start_time = time.perf_counter()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
            print("get logits_processor and use multinomial")
        else:
            logits_processor = None
            print("no logits_processor and use argmax")
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.transformer.h[-1].attn.c_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
        new_token = 0
        iter_num = 0
        total_accept_length = 0
        prepare_end_time = time.perf_counter()
        if verbose:
            print(f"prepare time cost: {prepare_end_time - prepare_start_time:.6f} s")
        for idx in range(max_length):
            round_start_time = time.perf_counter()
            # print(f"sample_token: {tokenizer.decode(sample_token[0])}")
            # print(f"input_ids: {input_ids.shape}")
            generate_start_time = time.perf_counter()
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
            generate_end_time = time.perf_counter()
            if verbose:
                print(f"generate time cost: {generate_end_time - generate_start_time:.6f} s")
            # print(f"past_key_value shape: {past_key_values[0][0].shape}")
            decoding_start_time = time.perf_counter()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
            decoding_end_time = time.perf_counter()
            if verbose:
                print(f"decoding time cost: {decoding_end_time - decoding_start_time:.6f} s")
            verify_start_time = time.perf_counter()
            best_candidate, accept_length, sample_p = evaluate_posterior(
                tokenizer,
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )
            verify_end_time = time.perf_counter()
            if verbose:
                print(f"verify time cost: {verify_end_time - verify_start_time:.6f} s")
            # print(f"sps: {tokenizer.decode(candidates[best_candidate][1 : accept_length + 1])}, accept_length: {accept_length}")
            update_start_time = time.perf_counter()
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )
            update_end_time = time.perf_counter()
            if verbose:
                print(f"update time cost: {update_end_time - update_start_time:.6f} s")
            round_end_time = time.perf_counter()
            if verbose:
                print(f"one round time cost: {round_end_time - round_start_time:.6f} s")
            iter_num += 1
            total_accept_length += accept_length
            if self.tokenizer.im_start_id in input_ids[0, input_len:].tolist():
                print(f"hit rate: {total_accept_length / iter_num}")
                return input_ids
            if self.tokenizer.im_end_id in input_ids[0, input_len:].tolist():
                if verbose:
                    print(f"iter_num: {iter_num}")
                    print(f"token_num: {input_ids.size()[1]}")
                    print(f"hit rate: {total_accept_length / iter_num}")
                return input_ids
            if new_token > max_new_tokens:
                print(f"hit rate: {total_accept_length / iter_num}")
                return input_ids
            if input_ids.shape[1] > max_length:
                print(f"hit rate: {total_accept_length / iter_num}")
                return input_ids

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            tree_choices=mc_sim_7b_63,
            max_length=2048,
            verbose=False
    ):
        prepare_start_time = time.perf_counter()
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.transformer.h[-1].attn.c_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        prepare_end_time = time.perf_counter()
        if verbose:
            print(f"prepare time cost: {prepare_end_time - prepare_start_time:.6f} s")
        iter_num = 0
        for idx in range(max_length):
            round_start_time = time.perf_counter()
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            
            # print(self.tokenizer.decode(input_id[0]))
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            
            round_end_time = time.perf_counter()
            if verbose:
                print(f"one round time cost: {round_end_time - round_start_time:.6f} s")
            iter_num += 1
            if self.tokenizer.im_start_id in input_ids[0, input_len:].tolist():
                return input_ids
            if self.tokenizer.im_end_id in input_ids[0, input_len:].tolist():
                if verbose:
                    print(f"iter_num: {iter_num}")
                    print(f"token_num: {input_ids.size()[1]}")
                return input_ids
            if input_ids.shape[1] > max_length:
                return input_ids

