import logging
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from transformers import BartModel, BartForConditionalGeneration, \
    T5Model, T5ForConditionalGeneration, \
    LEDModel, LEDForConditionalGeneration, \
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, \
    LlamaModel, LlamaForCausalLM, \
    MODEL_WITH_LM_HEAD_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

from typing import TypeVar, Generic

from index_building import Datastore, DatastoreBatch

import os
import json
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = logging.getLogger('Unlimiformer')
logger.setLevel(20)

ModelType = TypeVar('ModelType')
class Unlimiformer(Generic[ModelType]):
    def __init__(self, model: ModelType, 
            layer_begin=-1, layer_end=None,
            unlimiformer_head_num=None, 
            exclude_attention=False, 
            model_encoder_max_len=None,
            chunk_overlap=0,
            verbose=False, save_heatmap=False, 
            tokenizer=None, unlimiformer_training=False,
            use_datastore=False, 
            flat_index=False,
            test_datastore=False, reconstruct_embeddings=False, 
            gpu_datastore=False, gpu_index=False,
            index_devices=(0,), datastore_device=0, 
            num_anchors=2, num_templates=2, 
            token_count_file="token_counts.json", 
            data_file = "combined_facts.txt",
            num_extract=20, split_data=True
            ):
        super().__init__()
        self.csv_unlimiformer = True
        if self.csv_unlimiformer:
            self.not_first_encoding_pass = False
            self.num_anchors = num_anchors
            self.num_templates = num_templates
            self.num_retrieved = 0
            self.is_second_test_decoding_step = False
            self.token_count_file = token_count_file
            with open(self.token_count_file, "r") as f:
                self.fact_lengths = json.loads(f.read())["fact_lengths"]
            self.anchor_length = sum(self.fact_lengths[:self.num_anchors])
            self.update_begin = self.fact_lengths[0]
            self.update_end = sum(self.fact_lengths)
            self.num_extract = 20
            # self.num_extract = sum(self.fact_lengths[7:-7][::1])
            self.cover_end = sum(self.fact_lengths[-7:])
            self.cover_start = sum(self.fact_lengths[1:7])
            with open("dataset/persons.txt", "r") as f:
                self.person_list = f.read().splitlines()
            with open("dataset/actions.txt", "r") as f:
                self.action_list = f.read().splitlines()
            
        self.model = model
        model.unlimiformer = self
        self.layer_begin = layer_begin
        self.layer_end = layer_end
        self.specific_head = unlimiformer_head_num
        self.exclude_attention = exclude_attention
        self.actual_model_window_size = None
        self.model_encoder_max_len = model_encoder_max_len
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.save_heatmap = save_heatmap
        self.tokenizer = tokenizer
        self.unlimiformer_training = unlimiformer_training

        self.use_datastore = use_datastore
        self.flat_index = flat_index
        self.reconstruct_embeddings = reconstruct_embeddings
        self.gpu_datastore = gpu_datastore
        self.gpu_index = gpu_index
        if torch.cuda.is_available() and gpu_index:
            self.index_devices = [torch.device(f'cuda:{i}') for i in index_devices]
        else:
            self.index_devices = [torch.device('cpu')]
        self.datastore_device = torch.device(f'cuda:{datastore_device}' if torch.cuda.is_available() and gpu_datastore else 'cpu')
        self.test_datastore = test_datastore # flag for debugging

        self.device = torch.device(f'cuda:{datastore_device}' if torch.cuda.is_available() else 'cpu')
        self.activation_capturer = None
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.hook_handles = []
        self.is_input_encoding_pass = False
        self.is_first_test_decoding_step = False
        self.prev_tokens = None
        self.last_beam_idx = None
        self.heatmap = None
        self.cur_decoder_layer_index = None
        self.datastore = None

        self.break_into(model)

    def break_into(self, model):
        self.actual_model_window_size = self.window_size()
        if self.model_encoder_max_len is None:
            self.model_encoder_max_len = self.actual_model_window_size
        self.window_margin = int(self.model_encoder_max_len * self.chunk_overlap / 2)
        self.num_heads = model.config.num_attention_heads
        if self.specific_head is None:
            self.head_nums = Ellipsis # torch.arange(0, self.num_heads, device=self.device)
        else:
            self.head_nums = self.specific_head
        self.hooks_injected = False
        self.training_hooks_injected = False
        self.original_forward_func = model.forward

        # Activate Unlimiformer when calling model.eval(), deactivate for model.train()
        self.original_model_eval_func = model.eval
        model.eval = self.pre_eval_hook
        self.original_model_train_func = model.train
        model.train = self.pre_train_hook

    def pre_eval_hook(self):
        self.remove_training_hooks(self.model)
        self.inject_hooks(self.model)
        self.original_model_eval_func()

    def pre_train_hook(self, mode=True):
        # mode=True means model.train() is called
        # mode=False means model.eval() is called
        torch.cuda.empty_cache()
        if mode is True:
            self.break_out(self.model)
            if self.unlimiformer_training:
                self.inject_training_hooks(self.model)
        self.original_model_train_func(mode)
        
    def inject_hooks(self, model):
        if self.hooks_injected:
            return
        # Inject our activation_capturer to capture the activations at every forward pass
        attention_layers_to_capture = self.activation_to_capture(self.layer_begin, self.layer_end)
        self.activation_capturer = []
        for layer in attention_layers_to_capture:
            if type(layer) is list:
                layer_capturers = []
                for k_or_v in layer:
                    capturer = ActivationCapturer(k_or_v, capture_input=False)
                    layer_capturers.append(capturer)
                    self.register_hook(k_or_v, capturer)
                self.activation_capturer.append(layer_capturers)
            else:
                capturer = ActivationCapturer(layer, capture_input=False)
                self.register_hook(layer, capturer)
                self.activation_capturer.append(capturer)

        # Inject our main function after the main attention function
        attention_layers_to_run = self.attention_op_to_run(self.layer_begin, self.layer_end)
        for layer in attention_layers_to_run:
            self.register_hook(layer, self.attention_forward_hook)

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            decoder_layer_cross_attention = self.cross_attention(decoder_layer)
            self.original_decoder_layer_cross_attn_forward_funcs.append(decoder_layer_cross_attention.forward)
            decoder_layer_cross_attention.forward = self.create_cross_attn_pre_forward_hook(decoder_layer_cross_attention.forward, i)

        # Inject our hook function in the beginning of generation.
        # When the "model.generate()" will be called, it will first call our "reset_generation()" function, 
        # and only then call "model.generate()"
        self.original_generate_func = model.generate
        model.generate = self.pre_generate_hook

        model.forward = self.pre_forward_hook
        
        self.original_reorder_cache_func = model._reorder_cache
        model._reorder_cache = self.reorder_cache_hook
        self.hooks_injected = True

    def inject_training_hooks(self, model):
        if self.training_hooks_injected:
            return
        # self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        
        self.original_decoder_layer_self_attn_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            attention = self.self_attention(decoder_layer)
            self.original_decoder_layer_self_attn_forward_funcs.append(attention.forward)
            attention.forward = self.create_self_attn_pre_forward_hook(attention.forward)

        self.original_decoder_layer_cross_attn_forward_funcs = []
        for i, decoder_layer in enumerate(decoder_layers_to_run):
            decoder_layer_cross_attention = self.cross_attention(decoder_layer)
            self.original_decoder_layer_cross_attn_forward_funcs.append(decoder_layer_cross_attention.forward)
            decoder_layer_cross_attention.forward = self.create_cross_attn_pre_forward_hook(decoder_layer_cross_attention.forward, i)

        self.original_decoder_layer_forward_funcs = []
        for decoder_layer in decoder_layers_to_run:
            self.original_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_decoder_layer_func(decoder_layer.forward, decoder_layer)

        self.inject_hooks_for_unaffected_layers(model, decoder_layers_to_run)

        attention_layers_to_run = self.attention_op_to_run(self.layer_begin, self.layer_end)
        for layer in attention_layers_to_run:
            self.register_hook(layer, self.train_attention_forward_hook)

        self.training_hooks_injected = True

    def inject_hooks_for_unaffected_layers(self, model, decoder_layers_to_run):
        self.original_non_injected_decoder_layer_forward_funcs = []
        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None) 
            if l not in decoder_layers_to_run]
        for decoder_layer in non_injected_decoder_layers:
            self.original_non_injected_decoder_layer_forward_funcs.append(decoder_layer.forward)
            decoder_layer.forward = self.create_noninjected_decoder_layer_func(decoder_layer.forward, decoder_layer)

    def create_self_attn_pre_forward_hook(self, original_self_attn_forward_func):
        def self_attention_pre_forward_hook(*args, **kwargs):
            kwargs['past_key_value'] = None
            return original_self_attn_forward_func(*args, **kwargs)
        
        return self_attention_pre_forward_hook

    def create_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
        def checkpointed_decoder_layer(
                hidden_states: torch.Tensor,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                position_bias=None,
                encoder_decoder_position_bias=None,
                use_cache=True):

           
            def forward_with_all_keys(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, past_key_value, 
                    output_attentions, use_cache, long_inputs, long_inputs_mask,
                    position_bias, encoder_decoder_position_bias):
                
                key, value = self.create_key_value(long_inputs, decoder_layer)
                decoder_layer_args = self.create_decoder_layer_args(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    use_cache=use_cache,
                    key=key,value=value)
                return decoder_layer_original_forward_func(**decoder_layer_args)

            return torch.utils.checkpoint.checkpoint(
                forward_with_all_keys, hidden_states, attention_mask, 
                encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                cross_attn_layer_head_mask, None, 
                output_attentions, use_cache, self.long_inputs_encoded, self.long_inputs_mask,
                position_bias, encoder_decoder_position_bias)

        return checkpointed_decoder_layer

    def create_noninjected_decoder_layer_func(self, decoder_layer_original_forward_func, decoder_layer):
        def checkpointed_decoder_layer(
                hidden_states: torch.Tensor,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                position_bias=None,
                encoder_decoder_position_bias=None,
                use_cache=True):

           
            def forward_with_all_keys(hidden_states, attention_mask, 
                    encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                    cross_attn_layer_head_mask, past_key_value, 
                    output_attentions, use_cache, long_inputs, long_inputs_mask,
                    position_bias, encoder_decoder_position_bias):
                
                decoder_layer_args = self.create_decoder_layer_args(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    use_cache=use_cache, key=None, value=None)
                return decoder_layer_original_forward_func(**decoder_layer_args)

            return torch.utils.checkpoint.checkpoint(
                forward_with_all_keys, hidden_states, attention_mask, 
                encoder_hidden_states, encoder_attention_mask, layer_head_mask, 
                cross_attn_layer_head_mask, None, 
                output_attentions, use_cache, self.long_inputs_encoded, self.long_inputs_mask,
                position_bias, encoder_decoder_position_bias)

        return checkpointed_decoder_layer

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self, model):
        self.prompt_keys = []
        self.prompt_values = []
        self.prompt_attention_mask = []
        self.generated_input_ids = []
        torch.cuda.empty_cache()
        if not self.hooks_injected:
            return

        for h in self.hook_handles:
            h.remove()
        model.generate = self.original_generate_func
        model.forward = self.original_forward_func
        model._reorder_cache = self.original_reorder_cache_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func
        self.hooks_injected = False

    def remove_training_hooks(self, model):
        self.long_inputs_encoded, self.long_inputs_mask = None, None
        if not self.training_hooks_injected:
            return
        for h in self.hook_handles:
            h.remove()
        model.forward = self.original_forward_func

        decoder_layers_to_run = self.attention_layer_to_run(self.layer_begin, self.layer_end)
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_self_attn_forward_funcs):
            self.self_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_cross_attn_forward_funcs):
            self.cross_attention(decoder_layer).forward = original_func
        for decoder_layer, original_func in zip(decoder_layers_to_run, self.original_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        non_injected_decoder_layers = [l for l in self.attention_layer_to_run(0, None) 
            if l not in decoder_layers_to_run]
        for decoder_layer, original_func in zip(non_injected_decoder_layers, self.original_non_injected_decoder_layer_forward_funcs):
            decoder_layer.forward = original_func

        self.training_hooks_injected = False

    def reset_memory(self, input_ids, attention_mask):
        if self.use_datastore:
            if self.is_encoder_decoder:
                self.datastore = [DatastoreBatch(dim=self.model.config.hidden_size, batch_size=input_ids.shape[0], flat_index=self.flat_index, 
                    gpu_index=self.gpu_index, index_device=self.index_devices[0])]
                self.hidden_states = [[]]
            else:
                self.datastore = [DatastoreBatch(dim=self.model.config.hidden_size, batch_size=input_ids.shape[0], flat_index=self.flat_index, 
                    gpu_index=self.gpu_index, index_device=self.index_devices[i % len(self.index_devices)]) 
                    for i in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
                self.hidden_states = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
                if self.csv_unlimiformer:
                    self.inputs_head = [None for _ in range(40)]
                    self.tokens_head = [None for _ in range(40)]
                    self.selected_heads = [[] for _ in range(40)]
            torch.cuda.empty_cache()
        self.prompt_input_ids = input_ids
        self.input_ids_size = input_ids.shape[-1]
        self.prompt_keys, self.prompt_values = None, None
        self.prev_tokens = [None for _ in range(len(self.original_decoder_layer_cross_attn_forward_funcs))]
        self.last_beam_idx = None
        self.cur_layer_key_value_placeholder = None
        self.is_input_encoding_pass = True
        if self.is_encoder_decoder:
            dummy_labels = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        else:
            dummy_labels = None
        if self.save_heatmap:
            if self.heatmap is not None:
                print(f'Generated: {self.tokenizer.decode(self.generated_input_ids[0])}')
                self.plot_heatmap(self.heatmap[0].detach().cpu().numpy())
            self.heatmap = torch.tensor([], dtype=torch.float, device=input_ids.device)
        self.generated_input_ids = torch.tensor([], dtype=torch.long, device=input_ids.device)

        self.prompt_keys = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
        self.prompt_values = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
        self.prompt_attention_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            logger.info(f'Encoding {context_start_ind} to {context_end_ind} out of {input_ids.shape[-1]}')
            if self.csv_unlimiformer:
                chunk = input_ids[:, context_start_ind:context_end_ind].to(self.device)
                chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind].to(self.device)

                chunk_position_ids = []
                prev_end = 0
                const_dist = 0
                for fact in range(len(self.fact_lengths)):
                    chunk_position_ids.append(torch.arange(prev_end, prev_end + self.fact_lengths[fact]))
                    prev_end += const_dist + self.fact_lengths[fact]
                    if fact % 10 == 0:
                        const_dist += 0
                chunk_position_ids = torch.cat(chunk_position_ids, dim=0).unsqueeze(0).to(self.device)
                # self.chunk_position_ids = chunk_position_ids[:, 0:update_end_ind - update_start_ind]
                self.chunk_position_ids = chunk_position_ids[:, update_start_ind:update_end_ind]

                logger.info(f'{(self.tokenizer.decode(chunk[0]))}')
            else:
                chunk = input_ids[:, context_start_ind:context_end_ind].to(self.device)
                chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind].to(self.device)
                chunk_position_ids = None
            with torch.inference_mode():
                _ = self.model(chunk, attention_mask=chunk_attention_mask, labels=dummy_labels, position_ids=chunk_position_ids)
            if self.use_datastore:
                # TODO: verify with BART as well
                # hidden_states_to_index = [hidden_states.encoder_last_hidden_state] # list of length 1 of (batch, chunked_source_len, dim)
                hidden_states_to_index = [
                    layer_capturer.captured for layer_capturer in self.activation_capturer
                ] 
                # hidden_states_to_index = list(hidden_states.hidden_states)[:-1][self.layer_begin:self.layer_end]
                to_add = [state[:, update_start_ind:update_end_ind].detach() for state in hidden_states_to_index]
                to_apply_mask = chunk_attention_mask[:, update_start_ind:update_end_ind]
                # to_apply_mask = to_apply_mask.log().to(to_add[0].dtype)
                to_apply_mask = to_apply_mask.to(to_add[0].dtype)
                if not self.reconstruct_embeddings:
                    to_add_embeddings = to_add
                    if not self.gpu_datastore:
                        to_add_embeddings = [states.cpu() for states in to_add_embeddings]
                        to_apply_mask = to_apply_mask.cpu()
                    for i, layer_states in enumerate(to_add_embeddings):
                        layer_states = layer_states * to_apply_mask.unsqueeze(-1)
                        self.hidden_states[i].append(layer_states.to(self.datastore_device))

                if self.csv_unlimiformer:
                    logger.info(f'{self.tokenizer.decode(chunk[0][update_start_ind:update_end_ind])}')
                    self.num_retrieved += len(to_apply_mask[0])
                    self.not_first_encoding_pass = True
                    self.input_ids = chunk[0][update_start_ind:update_end_ind]
                # list of len layers, inside it there is a list of len batch, each item is (masked_time, dim)
                # for i, to_add_layer in enumerate(to_add):
                #     keys = [key[mask.bool()] for key, mask in zip(to_add_layer, to_apply_mask)]
                #     self.datastore[i].add_keys(keys)
            if (not self.use_datastore) or self.test_datastore:
                layers_kv = [
                    self.process_key_value(layer_capturer) # (batch, head, time, dim)
                    for layer_capturer in self.activation_capturer
                ] # list of pairs of (batch, head, time, dim)

                # list of (batch, head, chunked_time, dim)
                key = [layer[0][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                value = [layer[1][:, :, update_start_ind:update_end_ind] for layer in layers_kv]
                chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

                # key = torch.stack(key, dim=0) # (num_layers, batch, head, time, dim)
                # value = torch.stack(value, dim=0) # (num_layers, batch, head, time, dim)

                for i, (layer_key, layer_value) in enumerate(zip(key, value)):
                    self.prompt_keys[i].append(layer_key) # (num_layers, batch, head, chunked_source_len, dim)
                    self.prompt_values[i].append(layer_value) # (num_layers, batch, head, chunked_source_len, dim)
                self.prompt_attention_mask.append(chunk_attention_mask) # (batch, chunked_source_len)
        
        if self.use_datastore:
            # keys are all in datastore already!
            if not self.reconstruct_embeddings:
                # self.hidden_states = [torch.cat(layer_hidden_states, axis=1) for layer_hidden_states in self.hidden_states]
                concat_hidden_states = []
                for i in range(len(self.hidden_states)):
                    concat_hidden_states.append(torch.cat(self.hidden_states[i], axis=1))
                    self.hidden_states[i] = None
                self.hidden_states = concat_hidden_states
            for datastore, layer_hidden_states in zip(self.datastore, self.hidden_states):
                datastore.train_index(layer_hidden_states)
        if (not self.use_datastore) or self.test_datastore:
            for i, (layer_keys, layer_values) in enumerate(zip(self.prompt_keys, self.prompt_values)):
                self.prompt_keys[i] = torch.cat(layer_keys, dim=-2)
                self.prompt_values[i] = torch.cat(layer_values, dim=-2)
            # self.prompt_keys = torch.cat(self.prompt_keys, dim=-2) # (num_layers, batch, head, source_len, dim)
            # self.prompt_values = torch.cat(self.prompt_values, dim=-2) # (num_layers, batch, head, source_len, dim)
            self.prompt_attention_mask = torch.cat(self.prompt_attention_mask, dim=-1) # (batch, source_len)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.actual_model_window_size], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.actual_model_window_size:], skip_special_tokens=True)}')
            print()

    def chunked_encode_input(self, input_ids, attention_mask):
        long_inputs_encoded = []
        long_inputs_mask = []
        window_indices = self.window_indices(input_ids.shape[-1])

        self.is_input_encoding_pass = True
        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            chunk = input_ids[:, context_start_ind:context_end_ind]
            chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind]
            output = self.model.base_model.encoder(chunk, attention_mask=chunk_attention_mask, return_dict=True, output_hidden_states=True)
            encoder_last_hidden_state = output.last_hidden_state # (batch, time, dim)
            
            # list of (batch, head, chunked_time, dim)
            encoder_last_hidden_state = encoder_last_hidden_state[:, update_start_ind:update_end_ind] # (batch, chunked_time, dim)
            chunk_attention_mask = chunk_attention_mask[:, update_start_ind:update_end_ind] # (batch, chunked_time)

            long_inputs_encoded.append(encoder_last_hidden_state) # (batch, chunked_source_len, dim)
            long_inputs_mask.append(chunk_attention_mask) # (batch, chunked_source_len)

        long_inputs_encoded = torch.cat(long_inputs_encoded, dim=1) # (batch, source_len, dim)
        long_inputs_mask = torch.cat(long_inputs_mask, dim=1) # (batch, source_len)

        self.is_input_encoding_pass = False
        if self.verbose:
            print(f'Input: '
                f'{self.tokenizer.decode(input_ids[0][:self.actual_model_window_size], skip_special_tokens=True)} ||| '
                f'{self.tokenizer.decode(input_ids[0][self.actual_model_window_size:], skip_special_tokens=True)}')
            print()
        return long_inputs_encoded, long_inputs_mask

    def window_indices(self, total_seq_len):
        # Copied from SLED (Ivgy et al., 2022)
        # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
        if not self.csv_unlimiformer and total_seq_len <= self.model_encoder_max_len:
            return [(0, total_seq_len, 0, total_seq_len)]
        else:
            results = []
            # if self.chunk_overlap == 0:
            #     stride = self.model_encoder_max_len
            if self.csv_unlimiformer:
                segment_lengths = self.fact_lengths

                results = []
                context_start = 0
                context_end = segment_lengths[0] + 1
                
                results.append((0, None, self.update_begin, self.update_end))
                return results

            stride = self.model_encoder_max_len - 2 * self.window_margin
            context_start = update_start_ind = 0
            context_end = self.model_encoder_max_len
            if self.is_encoder_decoder:
                update_end_ind = context_end - self.window_margin
            else:
                update_end_ind = context_end
            # first window always should update from the beginning
            results.append((context_start, context_end, update_start_ind, update_end_ind))  

            while context_end < total_seq_len:
                context_end = min(total_seq_len, context_end + stride)
                context_start = (
                    context_start + stride if context_end < total_seq_len else total_seq_len - self.model_encoder_max_len
                )
                update_start_ind = max(update_start_ind + stride, update_end_ind)
                # last window always should update until the end
                update_end_ind = (
                    min(total_seq_len, update_end_ind + stride) if context_end < total_seq_len else total_seq_len
                )

                cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                 update_end_ind - context_start

                results.append((cs, ce, us, ue))
            return results

    # format copied from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py
    def suffix(self):
        return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with short responses according to the question. \n<</SYS>>\n\nBelow is a list of facts in the format (Fact: <PERSON> is <ACTION>).\nFact: Emma is Singing, '
        
    # format copied from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py
    def suffix2(self, i=0):
        if i%2:
            return 'Based on the above list of facts, can you tell me the action associated with the person: ' + self.person_list[i//2] + '?[/INST]'
        else:
            return 'Based on the above list of facts, can you tell me the person associated with the action: ' + self.action_list[i//2] + '?[/INST]'
        
    def pre_generate_hook(self, input_ids, **kwargs):
        if 'attention_mask' not in kwargs:
            kwargs['attention_mask'] = torch.ones_like(input_ids)
        self.reset_memory(input_ids, kwargs['attention_mask'])
        new_kwargs = kwargs
        if 'attention_mask' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
            new_kwargs['attention_mask'] = kwargs['attention_mask'][:, :self.actual_model_window_size].to(self.device)
        new_kwargs['use_cache'] = True
        if self.is_encoder_decoder:
            input_ids_prefix = input_ids[:, :self.actual_model_window_size]
        else:
            if self.csv_unlimiformer:
                vals_pred = []
                for i in [23]:
                # for i in [15, 23, 34, 40, 59, 71, 88, 93]:
                # for i in [13, 14, 50, 51, 87, 88]:
                # for i in [13, 14, 15, 16, 17, 18, 19, 20, 21]:
                # for i in range(7, len(self.fact_lengths), 4):
                    for j in [i*2, i*2+1]:
                        self.curr_key = j
                        input_ids_prefix = self.tokenizer.encode(self.suffix(), add_special_tokens=False, return_tensors="pt")
                        new_kwargs["attention_mask"] = torch.cat((torch.zeros(1, self.num_extract + self.cover_end + self.cover_start), torch.ones(1, len(input_ids_prefix[0]))), dim = 1).to(self.device)
                        input_ids_prefix = torch.cat((torch.ones(1, self.num_extract + self.cover_end + self.cover_start).to(torch.int64), input_ids_prefix), dim = 1).to(self.device)
                        vals_pred.append(self.original_generate_func(input_ids_prefix, **new_kwargs))
                        logger.info(f"{self.selected_heads}")
                        self.selected_heads = [[] for _ in range(40)]
                return vals_pred
            input_ids_prefix = input_ids[:, -self.actual_model_window_size:]	
        input_ids_prefix = input_ids_prefix.to(self.device)
        return self.original_generate_func(input_ids_prefix, **new_kwargs)

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.set_gradient_checkpointing(False)
        if not self.is_input_encoding_pass:
            if self.model.training:
                # self.reset_memory(input_ids, attention_mask)
                self.long_inputs_encoded, self.long_inputs_mask = self.chunked_encode_input(input_ids=input_ids, attention_mask=attention_mask)
                input_ids = input_ids[:, :self.actual_model_window_size]
                attention_mask = attention_mask[:, :self.actual_model_window_size] if attention_mask is not None else None
                # input_ids = input_ids[:, :self.model_encoder_max_len]
                # labels = labels[:, :self.model_encoder_max_len] if labels is not None else None
            else:
                if kwargs.get('past_key_values') is None:
                    self.is_first_test_decoding_step = True
                    if self.csv_unlimiformer:
                        self.question_len = (attention_mask[0] == 1).sum(dim=0)
                        self.is_second_test_decoding_step = False
                        self.num_generated = 0
                        kwargs["position_ids"] = torch.cat((torch.zeros(self.num_extract + self.cover_end + self.cover_start), torch.arange(0, self.question_len))).unsqueeze(0).to(self.device)

                if self.csv_unlimiformer and self.is_second_test_decoding_step:
                    input_ids_suffix = self.tokenizer.encode(self.suffix2(self.curr_key), add_special_tokens=False, return_tensors="pt")
                    self.curr_suffix_len = len(input_ids_suffix[0])
                    if self.num_generated == (len(input_ids_suffix[0]) + 1):
                        self.is_second_test_decoding_step = False
                    else:
                        input_ids = input_ids_suffix[:, self.num_generated - 1].unsqueeze(0).to(self.device)
                        self.query_pos_id = self.chunk_position_ids[0][-1] + int(kwargs["position_ids"][0])
                        kwargs["position_ids"] = torch.arange(self.query_pos_id, self.query_pos_id + 1).unsqueeze(0).to(self.device)

                if self.csv_unlimiformer and not self.is_second_test_decoding_step and not self.is_first_test_decoding_step:
                    self.query_pos_id = self.chunk_position_ids[0][-1] + int(kwargs["position_ids"][0])
                    kwargs["position_ids"] = torch.arange(self.query_pos_id, self.query_pos_id + 1).unsqueeze(0).to(self.device)

                if input_ids is not None:
                    if self.is_first_test_decoding_step:
                        self.input_ids_size += len(input_ids[0])
                    else:
                        self.input_ids_size += 1
                    if self.csv_unlimiformer:
                        self.num_generated += 1

                if kwargs.get('decoder_input_ids') is not None:
                    self.generated_input_ids = torch.cat([self.generated_input_ids, kwargs['decoder_input_ids']], axis=-1)
            logger.info(f'"Pre Forward Hook", {self.tokenizer.decode(input_ids[0])}')
        result = self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
        
        if self.csv_unlimiformer: 
            if not self.is_input_encoding_pass and not self.is_first_test_decoding_step:
                logits = result.logits
                top_k_values, top_k_indices = torch.topk(logits, k=10, dim=-1)
                top_k_token_ids = top_k_indices.tolist()[0][0]
                top_k_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in top_k_token_ids]
                logger.info(f'{top_k_tokens}')
                logger.info(f'{top_k_values}')

            if self.is_first_test_decoding_step:
                self.is_second_test_decoding_step = True
        self.is_first_test_decoding_step = False
        return result

    def create_cross_attn_pre_forward_hook(self, original_cross_attn_forward_func, cur_layer_num):
        def attention_pre_forward_hook(hidden_states, attention_mask=None, *args, **kwargs):
            self.cur_decoder_layer_index = cur_layer_num
            if kwargs.get('past_key_value') is not None:
                # it's a tuple, and we convert it to a list to be able to perform assignment 
                # and modify its items from our attention_forward_hook
                self.cur_layer_key_value_placeholder = \
                    kwargs['past_key_value'] = list(kwargs['past_key_value']) # (batch, head, time, attn_dim)

            batch_size, tgt_len, dim = hidden_states.shape
            if self.model.training:
                # from: (batch, tgt_len, dim) to: (batch * tgt_len, 1, dim)
                hidden_states = hidden_states.reshape(-1, 1, hidden_states.shape[-1])
                # from: (batch, 1, tgt_len, dim) to: (batch * tgt_len, 1, 1, dim)
                attention_mask = attention_mask.reshape(-1, 1, 1, attention_mask.shape[-1])

                attn_output, attn_weights_reshaped, past_key_value = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                attn_output = attn_output.reshape(batch_size, tgt_len, dim)
                result = (attn_output, attn_weights_reshaped, past_key_value)
            else:
                if not self.csv_unlimiformer or self.is_first_test_decoding_step or self.is_input_encoding_pass:
                    result = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                else:
                    attention_mask = attention_mask.repeat(1,40,1,1)

                    # config_layer_head = {1: [35], 2: [], 3: [0], 4: [33, 2, 16, 29], 5: [0, 32, 4, 21, 26, 28], 6: [36, 7, 12, 31], 7: [36, 21, 24, 14], 8: [33, 36, 10, 14, 17, 24, 31], 9: [32, 34, 6, 14, 20, 29], 10: [3, 4, 6, 13, 18, 30, 31], 11: [33, 1, 4, 37, 7, 11], 12: [5, 19, 20, 21, 24, 28, 31], 13: [1, 2, 4, 5, 11, 12, 13, 17, 21, 23, 24, 26, 30, 33, 36, 39], 14: [4, 7, 10, 11, 12, 13, 15, 19, 20, 26, 27, 28, 31, 34, 36, 39], 15: [0, 2, 7, 16, 19, 23, 24, 25, 26, 34, 37, 39], 16: [1, 2, 3, 5, 6, 7, 9, 11, 12, 14, 20, 21, 25, 30, 31, 32, 33, 39], 17: [3, 5, 8, 11, 16, 19, 22, 25, 28, 30, 33, 36, 37], 18: [11, 16, 18, 19, 20, 27, 30, 31, 32, 34, 35], 19: [1, 3, 5, 7, 8, 10, 14, 16, 21, 25, 29, 33, 37], 20: [0, 1, 9, 11, 18, 19, 22, 23, 25, 26, 31, 34, 36], 21: [0, 1, 4, 5, 6, 7, 15, 19, 22, 25, 27, 34, 35, 37, 39], 22: [16, 33, 29, 3, 8, 13], 23: [0, 1, 2, 3, 33, 38, 9, 13, 23], 24: [16, 17, 2, 19, 24, 25], 25: [5, 4, 9, 18, 30], 26: [5, 30, 15], 27: [33, 1, 18, 36, 29], 28: [20, 8, 24], 29: [26, 31], 30: [5, 14], 31: [1, 21, 5], 32: [25, 19, 4], 33: [], 34: [32, 34, 2, 38, 7, 6], 35: [17, 29, 30], 36: [23, 8, 27], 37: [0, 4, 8, 14, 16, 24], 38: [37, 22, 12, 13], 39: [35, 36, 37, 39, 27]}
                    # config_layer_head = {1: [35], 2: [], 3: [0], 4: [33, 2, 16, 29], 5: [0, 32, 4, 21, 26, 28], 6: [36, 7, 12, 31], 7: [36, 21, 24, 14], 8: [33, 36, 10, 14, 17, 24, 31], 9: [32, 6, 14, 20, 29], 10: [3, 4, 6, 13, 18, 30, 31], 11: [33, 1, 4, 7, 11], 12: [5, 19, 20, 21, 24, 28, 31], 13: [1, 2, 4, 5, 12, 13, 17, 23, 26, 30, 33, 36, 39], 14: [4, 7, 10, 11, 12, 13, 19, 20, 26, 27, 28, 31, 34, 36, 39], 15: [0, 2, 7, 16, 19, 23, 24, 25, 26, 34, 37, 39], 16: [2, 3, 5, 6, 7, 9, 11, 12, 14, 21, 25, 30, 31, 32, 33, 39], 17: [3, 5, 8, 11, 16, 19, 22, 25, 28, 33, 36, 37], 18: [11, 16, 18, 19, 20, 27, 30, 31, 32, 34, 35], 19: [1, 3, 5, 7, 8, 10, 14, 16, 21, 25, 29, 37], 20: [0, 1, 11, 18, 19, 22, 23, 25, 26, 31, 34, 36], 21: [0, 4, 5, 6, 7, 15, 19, 22, 27, 34, 35, 37, 39], 22: [16, 33, 29, 3, 8, 13], 23: [1, 2, 3, 33, 38, 9, 13], 24: [16, 17, 2, 19, 24, 25], 25: [5, 4, 9, 18, 30], 26: [5, 30, 15], 27: [1, 18, 36, 29], 28: [20, 24], 29: [26, 31], 30: [5, 14], 31: [1, 21, 5], 32: [25, 19, 4], 33: [], 34: [32, 34, 2, 38, 7, 6], 35: [29, 30], 36: [23, 8, 27], 37: [0, 4, 8, 14, 16, 24], 38: [37, 22, 12, 13], 39: [35, 36, 37, 39, 27]}
                    # config_layer_head = {1: [35], 2: [], 3: [0], 4: [33, 2, 16, 29], 5: [0, 32, 4, 21, 26, 28], 6: [36, 7, 12, 31], 7: [36, 21, 24, 14], 8: [33, 4, 36, 10, 14, 17, 24, 31], 9: [32, 33, 34, 6, 14, 20, 29], 10: [3, 4, 6, 13, 18, 25, 30, 31], 11: [33, 1, 4, 37, 7, 11], 12: [34, 5, 16, 19, 20, 21, 24, 28, 31], 13: [1, 2, 4, 5, 9, 11, 12, 13, 14, 17, 21, 23, 24, 26, 30, 33, 36, 39], 14: [0, 2, 4, 7, 10, 11, 12, 13, 15, 19, 20, 21, 26, 27, 28, 31, 34, 36, 39], 15: [0, 2, 7, 16, 19, 23, 24, 25, 26, 34, 37, 39], 16: [1, 2, 3, 5, 6, 7, 9, 11, 12, 14, 20, 21, 25, 30, 31, 32, 33, 39], 17: [2, 3, 4, 5, 8, 11, 16, 19, 22, 25, 28, 30, 33, 36, 37], 18: [11, 14, 15, 16, 18, 19, 20, 27, 30, 31, 32, 34, 35], 19: [0, 1, 3, 4, 5, 7, 8, 10, 14, 16, 21, 25, 27, 29, 33, 37], 20: [0, 1, 9, 11, 18, 19, 22, 23, 25, 26, 31, 34, 36], 21: [0, 1, 4, 5, 6, 7, 14, 15, 19, 22, 25, 27, 34, 35, 37, 39], 22: [16, 33, 29, 3, 8, 13], 23: [0, 1, 2, 3, 33, 38, 9, 13, 15, 23], 24: [16, 17, 2, 19, 24, 25], 25: [33, 36, 5, 4, 9, 18, 30], 26: [29, 5, 13, 30, 15], 27: [33, 1, 18, 36, 29], 28: [20, 22, 8, 9, 24], 29: [26, 31], 30: [5, 24, 14], 31: [1, 21, 5], 32: [25, 19, 4], 33: [], 34: [32, 34, 2, 38, 7, 6], 35: [17, 29, 30], 36: [2, 23, 8, 27], 37: [0, 4, 8, 14, 15, 16, 21, 24], 38: [28, 37, 22, 12, 13], 39: [35, 36, 37, 39, 24, 27]}
                    # config_layer_head = {1: [35], 2: [], 3: [], 4: [16, 2, 29], 5: [4, 26, 28], 6: [36, 31], 7: [], 8: [33, 4], 9: [33, 34], 10: [18, 13], 11: [37, 7], 12: [], 13: [21, 23, 24, 26, 30], 14: [0, 2, 21, 26, 13], 15: [24, 2], 16: [1, 2, 33, 6, 39, 12, 14, 20, 21, 25, 30, 31], 17: [2, 3, 4, 5, 8, 11, 16, 30], 18: [16, 30], 19: [0, 33, 3, 4, 5, 10, 14, 16], 20: [18, 34, 25, 31], 21: [1, 4, 7, 14, 15, 19, 25], 22: [13], 23: [0, 1, 23, 13, 15], 24: [], 25: [36, 30], 26: [13, 29, 5], 27: [33, 36], 28: [24, 9], 29: [24], 30: [24], 31: [], 32: [], 33: [], 34: [2, 6, 7], 35: [], 36: [27, 3], 37: [15], 38: [13], 39: [24]}
                    # only_data = {1: [35], 2: [], 3: [], 4: [16, 2, 29], 5: [32, 4], 6: [36, 31], 7: [], 8: [33, 4], 9: [33, 34], 10: [18, 13], 11: [37, 7], 12: [21], 13: [14, 21, 23, 24, 26, 30], 14: [0, 2, 21, 26], 15: [2], 16: [33, 2, 12, 20, 21, 25, 30, 31], 17: [33, 2, 3, 5, 8, 11, 16, 28, 30], 18: [16, 35, 30, 14], 19: [0, 33, 1, 3, 4, 10, 14, 16, 27, 29], 20: [34, 36, 18, 22, 23, 25], 21: [4, 7, 14, 15, 19, 25], 22: [33], 23: [1, 3, 13, 38], 24: [], 25: [30], 26: [30], 27: [33, 18], 28: [24], 29: [], 30: [24], 31: [], 32: [], 33: [11], 34: [], 35: [], 36: [2], 37: [4], 38: [], 39: []}
                    
                    # config_layer_head = {1: [34, 35], 2: [25, 35], 3: [33, 2, 35], 4: [33, 2, 12, 16, 29, 31], 5: [32, 35, 4, 21, 26, 27, 28], 6: [36, 8, 12, 31], 7: [4, 21, 24], 8: [17, 33, 4, 14], 9: [33, 34, 6], 10: [18, 6, 25, 13], 11: [20, 37, 7], 12: [28, 5, 31], 13: [33, 2, 21, 23, 24, 26, 30], 14: [0, 2, 39, 11, 13, 15, 20, 21, 22, 26, 30, 31], 15: [0, 33, 2, 37, 7, 8, 39, 11, 23, 24, 26], 16: [1, 2, 5, 6, 7, 12, 14, 18, 20, 21, 25, 29, 30, 31, 32, 33, 39], 17: [33, 2, 3, 4, 5, 36, 8, 11, 16, 19, 25, 28, 30], 18: [16, 18, 30, 11, 14], 19: [0, 33, 1, 3, 4, 5, 6, 8, 10, 14, 16, 27], 20: [34, 11, 18, 19, 24, 25, 26, 31], 21: [1, 4, 5, 7, 14, 15, 19, 25, 33, 34, 37, 39], 22: [16, 6, 8, 13], 23: [0, 1, 23, 13, 14, 15], 24: [19, 6], 25: [33, 36, 5, 30], 26: [13, 29, 5], 27: [33, 36], 28: [22, 24, 9], 29: [24, 27], 30: [24, 14, 30], 31: [21], 32: [19, 23], 33: [28], 34: [2, 34, 6, 7, 8], 35: [10], 36: [32, 3, 23, 7, 27], 37: [5, 6, 30, 15], 38: [12, 13], 39: [35, 5, 24, 15]}
                    # config_layer_head = {1: [35], 2: [], 3: [], 4: [16, 2, 29], 5: [4, 26, 28], 6: [36, 31], 7: [], 8: [33, 4], 9: [33, 34], 10: [18, 13], 11: [37, 7], 12: [], 13: [21, 23, 24, 26, 30], 14: [0, 2, 21, 26, 13], 15: [24, 2], 16: [1, 2, 33, 6, 39, 12, 14, 20, 21, 25, 30, 31], 17: [2, 3, 4, 5, 8, 11, 16, 30], 18: [16, 30], 19: [0, 33, 3, 4, 5, 10, 14, 16], 20: [18, 34, 25, 31], 21: [1, 4, 7, 14, 15, 19, 25], 22: [13], 23: [0, 1, 23, 13, 15], 24: [], 25: [36, 30], 26: [13, 29, 5], 27: [33, 36], 28: [24, 9], 29: [24], 30: [24], 31: [], 32: [], 33: [], 34: [2, 6, 7], 35: [], 36: [27, 3], 37: [15], 38: [13], 39: [24]}
                    config_layer_head = {
                        0: [],
                        1: [],
                        2: [],
                        3: [],
                        4: [16, 2],
                        5: [26, 28],
                        6: [36, 31],
                        7: [24],
                        8: [17, 4, 33],
                        9: [32, 33, 34, 20],
                        10: [18, 25, 13, 30],
                        11: [33, 37, 7, 11],
                        12: [16, 36, 5, 20, 28],
                        13: [9, 11, 12, 14, 17, 21, 23, 24, 26, 30],
                        14: [0, 34, 38, 7, 11, 12, 13, 20, 21, 25],
                        15: [2, 36, 7, 39, 17, 24],
                        16: [1, 2, 8, 12, 21, 23, 25, 30, 31, 33],
                        17: [0, 2, 3, 4, 5, 6, 8, 11, 16, 19, 28, 30, 33],
                        18: [14, 16, 18, 19, 27, 30],
                        19: [0, 1, 3, 4, 8, 9, 10, 16, 27, 33],
                        20: [0, 34, 36, 18, 22, 23, 25],
                        21: [0, 1, 4, 6, 7, 14, 15, 19, 22, 25, 27, 34],
                        22: [33, 34, 3, 1, 5, 13, 23, 29],
                        23: [0, 33, 2, 1, 13, 14, 15, 23],
                        24: [2, 16, 19, 21, 24, 25, 26, 27],
                        25: [33, 4, 36, 9, 18, 23, 30],
                        26: [33, 5, 39, 13, 15, 29, 30],
                        27: [33, 1, 36, 18, 29],
                        28: [5, 8, 9, 10, 22, 24],
                        29: [22, 39, 31],
                        30: [5, 39, 24],
                        31: [0, 1, 21, 5],
                        32: [32, 35, 19, 12],
                        33: [25, 18, 33],
                        34: [38, 7],
                        35: [17],
                        36: [32, 2, 3, 38, 16, 20, 23, 27],
                        37: [0, 4, 5, 6, 8, 14, 15, 16, 21, 25, 38],
                        38: [35, 7, 12, 28, 30, 31],
                        39: [1, 5, 7, 9, 10, 21, 24, 25, 29, 30, 32, 33, 36, 37, 39]
                    }
                    for head in range(40):
                        if cur_layer_num in config_layer_head.keys():
                            if head in config_layer_head[cur_layer_num]:
                                attention_mask[:, head, :, :self.cover_start + self.num_extract + self.cover_end] = 0
                                attention_mask[:, head, :, self.cover_start + self.num_extract + self.cover_end:] = 0.1
                            else:
                                attention_mask[:, head, :, :self.cover_start + self.num_extract + self.cover_end] = 0
                                attention_mask[:, head, :, self.cover_start + self.num_extract + self.cover_end:] = 0.1
                        else:
                            attention_mask[:, head, :, self.cover_start + self.num_extract:] = 1
                        # else:
                            # attention_mask[:, head, :, self.cover_start + self.num_extract:] = 5
                            # attention_mask[:, head, :, :self.cover_start + self.num_extract] = 0
                            # attention_mask[:, head] = torch.ones_like(attention_mask[:, head])

                    # attention_mask = torch.ones_like(attention_mask)
                    kwargs['output_attentions'] = True
                    result = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                    attn_wts = result[1].squeeze(0).squeeze(1)
                    
                    for head in range(len(attn_wts)):
                        sort_indices = torch.topk(attn_wts[head], 10).indices
                        # sort_indices = torch.topk(attn_wts[head][: self.cover_start + self.num_extract + self.cover_end], 5).indices
                        cutoff_point = self.cover_start + self.num_extract
                        sort_wts = attn_wts[head][sort_indices]
                        flag = False
                        for ind in range(len(sort_wts)):
                            if sort_wts[ind] >= 0.1:
                                if sort_indices[ind] < cutoff_point:
                                    flag = True
                                    break
                            else:
                                break
                        if flag == True:
                            if head not in self.selected_heads[cur_layer_num]:
                                self.selected_heads[cur_layer_num].append(head)
                        # if attn_wts[head][: self.cover_start + self.num_extract + self.cover_end][sort_indices][0] >= 0.01:
                            logger.info(f"layer {cur_layer_num}")
                            logger.info(f"head {head}")
                            logger.info(f"{sort_wts}")
                            logger.info(f"{sort_indices}")
                            # logger.info(f"{attn_wts[head][:self.cover_start + self.num_extract + self.cover_end][sort_indices]}")
                            # logger.info(f"{torch.sum(attn_wts[head][:self.cover_start + self.num_extract + self.cover_end])}")
                            # logger.info(f"{self.tokenizer.decode(self.inputs_head[head][sort_indices.cpu()].tolist())}")
                            # logger.info(f"{self.tokens_head[head][sort_indices.cpu()]}")
                            # logger.info(f"")
                # Uri: this part adds the generated tokens to the prompt. 
                # However it was commented out because currently we always keep the generated tokens in the attention window
                # if not self.is_encoder_decoder and not self.is_input_encoding_pass and \
                #         past_key_value[0].shape[2] > self.prompt_keys[self.cur_decoder_layer_index].shape[2]:
                #     self.prompt_keys[self.cur_decoder_layer_index] = torch.cat([self.prompt_keys[self.cur_decoder_layer_index], past_key_value[0][:,:,-1:]], dim=-2)
                #     self.prompt_values[self.cur_decoder_layer_index] = torch.cat([self.prompt_values[self.cur_decoder_layer_index], past_key_value[1][:,:,-1:]], dim=-2)
                #     if self.cur_decoder_layer_index == self.model.config.num_hidden_layers - 1:
                #         self.prompt_attention_mask = torch.cat([
                #             self.prompt_attention_mask, 
                #             torch.ones([self.prompt_attention_mask.shape[0], 1], dtype=self.prompt_attention_mask.dtype).to(self.device)], dim=-1)
        
            return result
        return attention_pre_forward_hook


    def attention_forward_hook(self, module, input, output):
        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass or self.is_first_test_decoding_step:
            return
        with torch.no_grad():
            prompt_size = self.prompt_input_ids.shape[1]
            generated_size = self.input_ids_size - prompt_size
            window_size = self.cur_layer_key_value_placeholder[0].shape[-2]
            # topk = min(self.actual_model_window_size, attn_weights.shape[-1])
            topk = min(prompt_size, window_size)
            if not self.is_encoder_decoder:
                topk = min(topk, window_size - generated_size + 1)
            if self.gpu_index:
                if self.csv_unlimiformer:
                    topk = self.num_extract
                    # topk = min(2048, self.num_retrieved)
                else:
                    topk = min(topk, 2048)

            query = self.process_query(output)[:,-1] # (batch * beam, head, dim)
            query = query[:, self.head_nums] # (batch * beam, head, dim)
            if self.use_datastore:
                # query: (batch, beam, head, dim)
                # need to multiply by key vector
                # query.view(query.shape[0], query.shape[1] * query.shape[2])
                # k_proj in attention? 
                datastore_index = 0 if self.is_encoder_decoder else self.cur_decoder_layer_index
                attention_layer_list = self.get_kv_projections(self.layer_begin, self.layer_end)
                k_proj_layer = [layers[0] for layers in attention_layer_list][self.cur_decoder_layer_index]
                v_proj_layer = [layers[1] for layers in attention_layer_list][self.cur_decoder_layer_index]
                
                # modify query by k_projs 
                k_proj = k_proj_layer.weight
                datastore_query = self.preprocess_query(query, k_proj) # (batch * beam, num_heads, embed_dim)
                batch_size = self.datastore[datastore_index].batch_size
                datastore_query = datastore_query.view((batch_size, -1, datastore_query.shape[2])) # (batch, beam * num_heads, embed_dim)
                # then search
                if self.reconstruct_embeddings:
                    # embeddings: (batch, beam * head, actual_model_window_size, dim)
                    _, top_search_key_indices, embeddings = self.datastore[datastore_index].search_and_reconstruct(datastore_query, k=topk) 
                else:
                    def do_scoring(vecs, query, topk, new_ind):
                        s = []
                        t = []
                        for q in range(len(query[0])):
                            similarities = torch.mv(vecs, query[0][q]) / (torch.norm(vecs, dim=1) * torch.norm(query[0][q]))
                            _, ind = torch.topk(similarities, k=topk)
                            t.append(self.chunk_position_ids[0][ind].unsqueeze(0))
                            s.append(new_ind[ind].unsqueeze(0))
                        ind = torch.cat(s, dim=0).unsqueeze(0)
                        ch = torch.cat(t, dim=0).unsqueeze(0)
                        return ind, ch
                    # top_search_key_scores, top_search_key_indices = self.datastore[datastore_index].search(datastore_query, k=topk)
                    # top_search_key_indices = torch.arange(0, self.num_retrieved).repeat(40, 1).unsqueeze(0)
                    new_hidden_states = self.hidden_states[datastore_index][0][self.cover_start:-self.cover_end]
                    new_hidden_states = torch.cat(torch.split(new_hidden_states, self.fact_lengths[7:-7])[::1])
                    new_indices = torch.cat(torch.split(torch.arange(sum(self.fact_lengths[1:7]), sum(self.fact_lengths[1:-7])), self.fact_lengths[7:-7])[::1]).to(self.device)
                    top_search_key_indices, send_indices = do_scoring(new_hidden_states, datastore_query, topk, new_indices)
                    top_search_key_indices = torch.cat((torch.arange(0, self.cover_start).unsqueeze(0).unsqueeze(0).repeat(1, 40, 1).to(self.device), torch.sort(top_search_key_indices).values, torch.arange(self.update_end - self.update_begin - self.cover_end, self.update_end - self.update_begin).unsqueeze(0).unsqueeze(0).repeat(1, 40, 1).to(self.device)), dim=2)
                    # send_indices = torch.cat((torch.sort(send_indices).values, self.chunk_position_ids[0][torch.arange(self.update_end - self.cover_end, self.update_end)].unsqueeze(0).unsqueeze(0).repeat(1, 40, 1).to(self.device)), dim=2)     
                    for head in range(40):
                        self.inputs_head[head] = torch.tensor(self.input_ids)[top_search_key_indices[0,head,:self.cover_start + self.num_extract + self.cover_end].cpu()]
                        self.tokens_head[head] = top_search_key_indices[0,head,:self.cover_start+ self.num_extract + self.cover_end].cpu()
                    # self.embeddings: (batch,              src_len, dim)
                    # indices:         (batch, beam * head, actual_model_window_size)
                    # embeddings: (batch, beam * head, actual_model_window_size, dim)
                    embeddings = torch.take_along_dim(input=self.hidden_states[datastore_index].unsqueeze(1), 
                        indices=top_search_key_indices.unsqueeze(-1).to(self.hidden_states[datastore_index].device), dim=-2)
                    embeddings = embeddings.to(self.device)
                # (batch, beam, head, actual_model_window_size)
                # top_search_key_scores = top_search_key_scores.reshape(batch_size, -1, *top_search_key_scores.shape[1:])
                top_search_key_indices = top_search_key_indices.reshape(batch_size, -1, *top_search_key_indices.shape[1:])
                send_indices = send_indices.reshape(batch_size, -1, *send_indices.shape[1:])
                # embeddings: (batch, beam, head, actual_model_window_size, dim)
                embeddings = embeddings.reshape(batch_size, -1, self.num_heads, *embeddings.shape[2:])

            # raw_values are actually token indices; need to look them up
            if (not self.use_datastore) or self.test_datastore:
                this_layer_prompt_keys = self.prompt_keys[self.cur_decoder_layer_index]
                this_layer_prompt_values = self.prompt_values[self.cur_decoder_layer_index]
                # query: (batch * beam, head, dim)
                batch_size = self.prompt_input_ids.shape[0]
                beam_size = query.shape[0] // batch_size
                # query: (batch, beam, head, dim)
                query = query.reshape(batch_size, beam_size, *query.shape[1:])
                # this_layer_prompt_keys: (batch, head, source_len, dim)
                # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
                # query.unsqueeze(-1):             (batch, beam, head, dim, 1)
                # attn_weights:  (batch, beam, head, source_len)
                attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums], query.unsqueeze(-1)).squeeze(-1) 
                # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
                prompt_attention_mask_to_add = (1 - self.prompt_attention_mask) * -1e9 # (batch, source_len)
                prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1)
                attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)
                if self.exclude_attention and attn_weights.shape[-1] > self.actual_model_window_size:
                    attn_weights[..., :self.actual_model_window_size] -= 1e9

                # target_keys, target_values, topk = self.get_target_slices(output)
                top_key_scores, top_key_indices = torch.topk(attn_weights, k=topk, dim=-1, sorted=True) # (batch, beam, head, trunc_source)
                if self.save_heatmap:
                    # heatrow: (beam, heads, source_len)
                    heatrow = torch.zeros([top_key_indices.shape[1], top_key_indices.shape[2], this_layer_prompt_keys.shape[-2]], dtype=torch.float)
                    heatrow = heatrow.scatter(index=top_key_indices[0], src=torch.ones_like(top_key_scores[0]), dim=-1)
                    # heatrow = torch.nn.functional.softmax(heatrow, dim=-1)
                    # self.heatmap: (beam, heads, targets, source_len)
                    self.heatmap = torch.cat([self.heatmap, heatrow.unsqueeze(-2)], axis=-2)

            if self.test_datastore:
                assert top_key_indices.shape == top_search_key_indices.shape
                assert torch.mean((top_key_indices == top_search_key_indices).float()) > 0.99

            if self.verbose:
                if self.is_encoder_decoder:
                    for i, beam in enumerate(self.generated_input_ids):
                        print(f'({i}) Generated: {self.tokenizer.decode(beam)}')
                # else:
                #     print(f'Generated: {self.tokenizer.decode(self.input_ids)}')
                print()

        if self.use_datastore:
            # k_proj_layer.weight, v_proj_layer.weight: (embed_dim, embed_dim)
            # embeddings: (batch, beam, head, encoder_len, embed_dim)
            retrieved_keys, retrieved_values = self.post_process_retrieved(embeddings, k_proj_layer, v_proj_layer, top_search_key_indices)
        else:
            # this_layer_prompt_keys:   (batch,       head, source_len, dim)
            # top_key_indices:          (batch, beam, head, trunc_source)
            retrieved_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            retrieved_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)

        if self.test_datastore:
            correct_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            correct_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
                dim=-2) # (batch, head, trunc_source, attn_dim)
            assert correct_keys.shape == retrieved_keys.shape
            assert correct_values.shape == retrieved_values.shape
            assert torch.mean(torch.isclose(correct_keys, retrieved_keys, rtol=1e-3, atol=1e-3).float()) > 0.99
            assert torch.mean(torch.isclose(correct_values, retrieved_values, rtol=1e-3, atol=1e-3).float()) > 0.99

        # retrieved_keys, retrieved_values: (batch * beam, head, encoder_len, attn_dim)
        retrieved_keys = retrieved_keys.flatten(0, 1)[:,:,:topk + self.cover_end + self.cover_start]
        retrieved_values = retrieved_values.flatten(0, 1)[:,:,:topk + self.cover_end + self.cover_start]
        self.cur_layer_key_value_placeholder[0] = torch.cat([retrieved_keys, self.cur_layer_key_value_placeholder[0][:,:,topk + self.cover_end + self.cover_start:]], dim=-2)
        self.cur_layer_key_value_placeholder[1] = torch.cat([retrieved_values, self.cur_layer_key_value_placeholder[1][:,:,topk + self.cover_end + self.cover_start:]], dim=-2)
        return

    def train_attention_forward_hook(self, module, input, output):
        # output: (batch, time, 3 * heads * attention_dim)
        if self.is_input_encoding_pass or self.is_first_test_decoding_step:
            return
        this_layer_prompt_keys = self.cur_layer_key_value_placeholder[0]
        this_layer_prompt_values = self.cur_layer_key_value_placeholder[1]
        with torch.no_grad():
            query = self.process_query(output) # (batch * beam, tgt_len, head, dim)
            # query = query[:, :, self.head_nums] # (batch * beam, head, dim)
            
            # query: (batch * beam, tgt_len, head, dim)
            batch_size = this_layer_prompt_keys.shape[0]
            tgt_len = query.shape[0] // batch_size
            # query: (batch, tgt, head, dim)
            query = query.reshape(batch_size, tgt_len, *query.shape[2:])
            # this_layer_prompt_keys: (batch, head, source_len, dim)
            # this_layer_prompt_keys.unsqueeze(1):  (batch, 1, head, source_len, dim)
            # attn_weights:  (batch, tgt_len, head, 1, source_len)
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1).permute(0,1,2,4,3))
            attn_weights = torch.matmul(this_layer_prompt_keys.unsqueeze(1), query.unsqueeze(-1)) \
                .reshape(batch_size, tgt_len, query.shape[-2], 1, this_layer_prompt_keys.shape[-2])
            # attn_weights = torch.matmul(query.unsqueeze(-2), this_layer_prompt_keys.unsqueeze(1)[:, :, self.head_nums]).squeeze(-2) 
            prompt_attention_mask_to_add = (1 - self.long_inputs_mask) * -1e9 # (batch, source_len)
            prompt_attention_mask_to_add = prompt_attention_mask_to_add.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            attn_weights += prompt_attention_mask_to_add # (batch, beam, head, source_len)

            # target_keys, target_values, topk = self.get_target_slices(output)
            topk = min(self.actual_model_window_size, attn_weights.shape[-1])
            top_key_scores, top_key_indices = torch.topk(attn_weights, k=min(topk, attn_weights.shape[-1]), dim=-1, sorted=True) # (batch, beam, head, tgt, trunc_source)

                   
        # this_layer_prompt_keys:   (batch,          head,    source_len, dim)
        # top_key_indices:          (batch, tgt_len, head, 1, trunc_source)
        new_keys = torch.take_along_dim(this_layer_prompt_keys.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        new_values = torch.take_along_dim(this_layer_prompt_values.unsqueeze(2).unsqueeze(1), indices=top_key_indices.unsqueeze(-1), 
            dim=-2) # (batch, tgt_len, head, 1, trunc_source, attn_dim)
        
        # (batch * beam, head, tgt_len, trunc_source, attn_dim)
        self.cur_layer_key_value_placeholder[0] = new_keys.flatten(0, 1).squeeze(2)
        self.cur_layer_key_value_placeholder[1] = new_values.flatten(0, 1).squeeze(2)
        return

    
    def preprocess_query(self, query, k_proj_weight):
        k_proj = k_proj_weight.view(1, self.num_heads, query.shape[-1], k_proj_weight.shape[0]) # (1, num_heads, attn_dim, embed_dim)
        datastore_query = query.unsqueeze(-2) # (batch * beam, num_heads, 1, attn_dim)
        datastore_query = torch.matmul(datastore_query, k_proj) # (batch * beam, num_heads, 1, embed_dim)
        datastore_query = datastore_query.squeeze(-2)  # (batch * beam, num_heads, embed_dim)
        return datastore_query

    def post_process_retrieved(self, embeddings, k_proj_layer, v_proj_layer, top_search_key_indices):
        embed_dim = embeddings.shape[-1]
        k_weight = k_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1) # (1, 1, heads, embed_dim, attn_dim)
        k_bias = 0
        if k_proj_layer.bias is not None:
            k_bias = k_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2).unsqueeze(0)
        v_weight = v_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1)  # (1, heads, embed_dim, attn_dim)
        v_bias = 0
        if v_proj_layer.bias is not None:
            v_bias = v_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2).unsqueeze(0)
        # new_keys, new_values: (batch, beam, head, encoder_len, attn_dim)
        retrieved_keys = torch.matmul(embeddings, k_weight) + k_bias # (beam, head, encoder_len, embed_dim)
        retrieved_values = torch.matmul(embeddings, v_weight) + v_bias # (beam, head, encoder_len, embed_dim)
        return retrieved_keys, retrieved_values

    def set_gradient_checkpointing(self, value):
        self.model.base_model.decoder.gradient_checkpointing = value

    def reorder_cache_hook(self, past, beam_idx):
        self.last_beam_idx = beam_idx
        self.generated_input_ids = self.generated_input_ids[beam_idx]
        for i, layer_prev_tokens in enumerate(self.prev_tokens):
            if layer_prev_tokens is not None:
                self.prev_tokens[i] = layer_prev_tokens.flatten(0, 1)[beam_idx].reshape(layer_prev_tokens.shape)
        if self.save_heatmap and self.heatmap.numel() > 0:
            self.heatmap = self.heatmap[beam_idx]
        return self.original_reorder_cache_func(past, beam_idx)
    
    @classmethod
    def convert_model(cls, model, *args, **kwargs):
        # if type(model.config) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
        # elif type(model.config) in MODEL_WITH_LM_HEAD_MAPPING:
        # else:
        #     raise ValueError(f'Unsupported model type: {type(model.config)}')
        # if model.config.is_encoder_decoder:
        #     model_clone = AutoModelForSeq2SeqLM.from_config(model.config)
        # else:
        #     model_clone = AutoModelForCausalLM.from_config(model.config)
        # model_clone.load_state_dict(model.state_dict()).to(args.device)
        type_to_class = {
            BartModel: UnlimiformerBART,
            BartForConditionalGeneration: UnlimiformerBART,
            T5Model: UnlimiformerT5,
            T5ForConditionalGeneration: UnlimiformerT5,
            LEDModel: UnlimiformerLED,
            LEDForConditionalGeneration: UnlimiformerLED,
            LlamaModel: UnlimiformerLLaMa,
            LlamaForCausalLM: UnlimiformerLLaMa,
        }
        type_to_class[type(model)](model, *args, **kwargs)
        return model
        

    def plot_heatmap(self, data, xticklabels='auto', yticklabels='auto'):
        # data: (heads, targets, source_len)
        import seaborn as sb
        import matplotlib.pyplot as plt
        # print('gat = np.array([')
        # for row in data[0]:
        #     rowstr = ', '.join([f'{x:.2f}' for x in row])
        #     print(f'    [{rowstr}],')
        # print(']')

        # sb.set(font_scale=1.5, rc={'text.usetex': True})
        for i in range(data.shape[0]):
            fig, axes = plt.subplots(1, 1, figsize=(40, 100))
            cur_ax = axes
            axes.set_title(f'Head #{i}, length: {data.shape[2]}, target length: {data.shape[1]}')
            cur_ax = axes
            # annot = [[x for x in row] for row in data]
            ax = sb.heatmap(data[i], annot=False, fmt='.2f',
                            xticklabels=512, yticklabels=yticklabels, ax=cur_ax)
            ax.xaxis.tick_top()
            plt.savefig(f'knns_head{i}.pdf')
            # plt.savefig('gat_s10_contrast.pdf')
            plt.show()


class UnlimiformerBART(Unlimiformer[BartModel]):
    def __init__(self, model: BartModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def create_key_value(self, encoder_hidden_states, decoder_layer):
        # (batch, time, hidden_dim)
        attention = decoder_layer.encoder_attn
        # key, value: (batch, heads, time, attn_dim)
        key = attention.k_proj(encoder_hidden_states)
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = attention.v_proj(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        # key, value: (batch, heads, time, attn_dim)
        return key, value 

    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value

    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.layers[-1].encoder_attn
        # query: (batch, heads, time, attn_dim)
        # query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query

    def get_kv_projections(self, layer_begin, layer_end): 
        return [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]

    def activation_to_capture(self, layer_begin, layer_end): 
        if self.use_datastore:
            return [self.model.base_model.encoder.layers[-1]]
        else:
            return self.get_kv_projections(layer_begin, layer_end)

    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.encoder_attn.q_proj
                for layer in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]

    def attention_layer_to_run(self, layer_begin, layer_end): 
        return self.model.base_model.decoder.layers[layer_begin:layer_end]

    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 

    def cross_attention(self, decoder_layer):
        return decoder_layer.encoder_attn
    
    def window_size(self):
        return self.model.config.max_position_embeddings

    def create_decoder_layer_args(self, hidden_states, attention_mask, encoder_hidden_states,
                encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask,
                past_key_value, output_attentions, position_bias,
                encoder_decoder_position_bias, use_cache, key, value):
        args = {'hidden_states': hidden_states, 
                'attention_mask': attention_mask, 
                'encoder_hidden_states': encoder_hidden_states, 
                'encoder_attention_mask': encoder_attention_mask, 
                'layer_head_mask': layer_head_mask, 
                'cross_attn_layer_head_mask': cross_attn_layer_head_mask, 
                'past_key_value': (None, None, key, value), 
                'output_attentions': output_attentions, 
                'use_cache': use_cache,}
        if key is None and value is None:
            args['past_key_value'] = None
        return args

class UnlimiformerT5(Unlimiformer[T5Model]):
    def __init__(self, model: T5Model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def create_key_value(self, encoder_hidden_states, decoder_layer):
        # (batch, time, hidden_dim)
        attention = decoder_layer.layer[1].EncDecAttention
        # key, value: (batch, heads, time, attn_dim)
        key = attention.k(encoder_hidden_states)
        key = key.view(key.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        value = attention.v(encoder_hidden_states)
        value = value.view(value.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        
        return key, value 
    
    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        key, value = key_capturer.captured, value_capturer.captured
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.block[-1].layer[1].EncDecAttention

        # query, key, value: (batch, heads, time, attn_dim)
        # query = query.view(query.shape[0], query.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        key = key.view(key.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).transpose(1, 2).contiguous()
        
        return key, value

    def process_query(self, output):
        # (batch, time, heads, attn_dim)
        attention = self.model.base_model.decoder.block[-1].layer[1].EncDecAttention
        # query: (batch, heads, time, attn_dim)
        query = output.view(output.shape[0], -1, attention.n_heads, attention.key_value_proj_dim).contiguous()
        return query

    def get_kv_projections(self, layer_begin, layer_end):
        return [
            [layer.layer[1].EncDecAttention.k, layer.layer[1].EncDecAttention.v]
                for layer in self.model.base_model.decoder.block[layer_begin:layer_end]
        ]
    
    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.layer[1].EncDecAttention.q
                for layer in self.model.base_model.decoder.block[layer_begin:layer_end]
        ]
    
    def attention_layer_to_run(self, layer_begin, layer_end): 
        return self.model.base_model.decoder.block[layer_begin:layer_end]

    def self_attention(self, decoder_layer):
        return decoder_layer.layer[0]

    def cross_attention(self, decoder_layer):
        return decoder_layer.layer[1]

    def window_size(self):
        try:
            size = self.model.config.n_positions
        except AttributeError:
            size = 1024
        return size

    def create_decoder_layer_args(self, hidden_states, attention_mask, encoder_hidden_states,
            encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask,
            past_key_value, output_attentions, position_bias,
            encoder_decoder_position_bias, use_cache, key, value):
        args = {'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_bias': position_bias,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask,
            'encoder_decoder_position_bias': encoder_decoder_position_bias,
            'layer_head_mask': layer_head_mask,
            'cross_attn_layer_head_mask': cross_attn_layer_head_mask,
            'past_key_value': (None, None, key, value),
            'use_cache': use_cache,
            'output_attentions': output_attentions}
        if key is None and value is None:
            args['past_key_value'] = None
        return args

class UnlimiformerLED(UnlimiformerBART):
    def __init__(self, model: LEDModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def window_size(self):
        return self.model.config.max_encoder_position_embeddings

class UnlimiformerLLaMa(Unlimiformer[LlamaModel]):
    def __init__(self, model: LlamaModel, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
    
    def get_kv_projections(self, layer_begin, layer_end): 
        return [
            [layer.self_attn.k_proj, layer.self_attn.v_proj]
            for layer in self.model.base_model.layers[layer_begin:layer_end]
        ]

    def activation_to_capture(self, layer_begin, layer_end): 
        if self.use_datastore:
            return [
                layer.input_layernorm
                for layer in self.model.base_model.layers[layer_begin:layer_end]
            ]
        else:
            return self.get_kv_projections(layer_begin, layer_end)

    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.self_attn.q_proj
                for layer in self.model.base_model.layers[layer_begin:layer_end]
        ]

    def attention_layer_to_run(self, layer_begin, layer_end): 
        return self.model.base_model.layers[layer_begin:layer_end]

    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn 

    def cross_attention(self, decoder_layer):
        return decoder_layer.self_attn
    
    def window_size(self):
        return self.model.config.max_position_embeddings
    
    def set_gradient_checkpointing(self, value):
        self.model.base_model.gradient_checkpointing = value

    def process_key_value(self, capturers):
        key_capturer, value_capturer = capturers
        # (batch, time, heads * attn_dim)
        key, value = key_capturer.captured, value_capturer.captured
        attention = self.model.base_model.layers[-1].self_attn

        # (batch, heads, time, attn_dim)
        key = key.view(key.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        value = value.view(value.shape[0], -1, attention.num_heads, attention.head_dim).transpose(1, 2).contiguous()
        
        return key, value
    
    def process_query(self, output):
        # output: (batch, time, heads * attn_dim)
        attention = self.model.base_model.layers[-1].self_attn
        
        # query: (batch, time, heads, attn_dim)
        query = output.view(output.shape[0], output.shape[1], attention.num_heads, attention.head_dim).contiguous()
        return query

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def preprocess_query(self, query, k_proj_weight):
        # query: (batch * time, head, dim)
        attention = self.model.base_model.layers[-1].self_attn
        # num_generated = min(self.input_ids_size - self.prompt_input_ids.shape[1], self.actual_model_window_size)
        cos, sin = attention.rotary_emb(query, seq_len= self.update_begin + self.num_generated)
        cos1, sin1 = attention.rotary_emb(query, seq_len= sum(self.fact_lengths[1:-7]) + self.num_generated)
        cos = cos[:,:,-1]  # [1, 1, dim]
        cos1 = cos1[:,:,-1]  # [1, 1, dim]
        sin = sin[:,:,-1]  # [1, 1, dim]
        sin1 = sin1[:,:,-1]  # [1, 1, dim]
        # cos = cos[-1].unsqueeze(0).unsqueeze(0)  # [bs, 1, seq_len, dim]
        # sin = sin[-1].unsqueeze(0)  # [bs, 1, seq_len, dim]
        query_dat = (query * cos) + (self.rotate_half(query) * sin)
        # query_dat += (query * cos1) + (self.rotate_half(query) * sin1)
        query = query_dat

        k_proj = k_proj_weight.view(1, self.num_heads, query.shape[-1], k_proj_weight.shape[0]) # (1, num_heads, attn_dim, embed_dim)
        k_proj_l = k_proj[..., :k_proj.shape[-2] // 2, :]
        k_proj_r = k_proj[..., k_proj.shape[-2] // 2:, :]
        k_proj_rotated = torch.cat([-k_proj_l, k_proj_r], dim=-2)

        datastore_query = query.unsqueeze(-2) # (batch * beam, num_heads, 1, attn_dim)
        # datastore_query = torch.matmul(datastore_query, k_proj) # (batch * beam, num_heads, 1, embed_dim)
        datastore_query = torch.matmul(datastore_query, k_proj + k_proj_rotated) # (batch * beam, num_heads, 1, embed_dim)
        datastore_query = datastore_query.squeeze(-2)  # (batch * beam, num_heads, embed_dim)
        return datastore_query

    def post_process_retrieved(self, embeddings, k_proj_layer, v_proj_layer, top_search_key_indices):
        embed_dim = embeddings.shape[-1]
        k_weight = k_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1) # (1, 1, heads, embed_dim, attn_dim)
        k_bias = 0
        if k_proj_layer.bias is not None:
            k_bias = k_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2).unsqueeze(0)
        v_weight = v_proj_layer.weight.view(1, 1, self.num_heads, embed_dim // self.num_heads, embed_dim).transpose(-2,-1)  # (1, heads, embed_dim, attn_dim)
        v_bias = 0
        if v_proj_layer.bias is not None:
            v_bias = v_proj_layer.bias.view(1, self.num_heads, embed_dim // self.num_heads).unsqueeze(-2).unsqueeze(0)
        # new_keys, new_values: (batch, beam, head, encoder_len, attn_dim)
        retrieved_keys = torch.matmul(embeddings, k_weight) + k_bias # (beam, head, encoder_len, embed_dim)
        retrieved_values = torch.matmul(embeddings, v_weight) + v_bias # (beam, head, encoder_len, embed_dim)
        attention = self.model.base_model.layers[-1].self_attn
        cos, sin = attention.rotary_emb(retrieved_values, seq_len=4000)
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        if False:#self.prompt_input_ids.shape[1] > self.actual_model_window_size:
            # scale the top key indices to the actual model window size, such that the model will not see
            # positional embeddings that did not appear at training time
            scaled_key_indices = ((top_search_key_indices / self.prompt_input_ids.shape[1]) * self.actual_model_window_size).int()
        else:
            # scaled_key_indices = torch.zeros_like(top_search_key_indices).to(cos.device)
            # scaled_key_indices = torch.tensor(len(self.input_ids[0])).to(cos.device) + torch.tensor(self.question_len).to(cos.device) - top_search_key_indices.to(cos.device)
            # top_search_key_indices[:, :, :, :self.num_extract] = 0
            # top_search_key_indices[:, :, :, :self.num_extract] = sum(self.fact_lengths[1:-7]) - top_search_key_indices[:, :, :, :self.num_extract]
            top_search_key_indices = torch.arange(top_search_key_indices[0][0][0][-1] - len(top_search_key_indices[0][0][0]), top_search_key_indices[0][0][0][-1]).repeat(40,1).unsqueeze(0).unsqueeze(0)
            scaled_key_indices = torch.tensor(self.question_len).to(cos.device) + top_search_key_indices.to(cos.device)
        # top_search_key_indices = top_search_key_indices.to(cos.device)
        scaled_key_indices = scaled_key_indices.to(cos.device)
        cos = cos[scaled_key_indices]  # [bs, 1, seq_len, dim]
        sin = sin[scaled_key_indices]  # [bs, 1, seq_len, dim]
        retrieved_keys = (retrieved_keys * cos) + (self.rotate_half(retrieved_keys) * sin)
        return retrieved_keys, retrieved_values


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def unwrap_tuple(self, t):
        if isinstance(t, tuple) and len(t) == 1:
            t = t[0]
        return t
    
    def forward(self, module, layer_input, layer_output):
        if self.capture_input:
            self.captured = self.unwrap_tuple(layer_input)
        else:
            self.captured = self.unwrap_tuple(layer_output)
    