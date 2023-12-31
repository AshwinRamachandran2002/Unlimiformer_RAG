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
            index_devices=(0,), datastore_device=0, tokens_ind=[]
            ):
        super().__init__()
        self.csv_unlimiformer = True
        self.not_first_encoding_pass = False
        self.input_ids_full = []
        self.tokens_ind = tokens_ind
        self.input_ids_full_extra = []
        self.attention_weights = []
        self.my_method = False
        self.apply_boundary = False
        self.one_by_one = True
        self.num_retrieved = 0
        self.gap = 5
        self.kt = 3
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
        self.is_second_test_decoding_step = False
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
                self.hidden_layer_our = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
                self.hidden_layer_our_anchor = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
                self.hidden_layer_our_bits = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
                self.comma_hidden_state = [[] for _ in range(self.model.config.num_hidden_layers)[self.layer_begin:self.layer_end]]
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
        
        if self.csv_unlimiformer:
            # prefix_input_id = self.tokenizer.encode(self.prefix(2), add_special_tokens=False, return_tensors="pt")
            # prefix_len = len(prefix_input_id[0])
            window_indices = self.window_indices(input_ids.shape[-1], 0)
        else:
            window_indices = self.window_indices(input_ids.shape[-1])

        # Experiment result: Putting just a single JSON data through and collecting the exact amount worked
        # Experiment result: Putting two JSON data? it gave a value but not the correct one
        # Experiment result: Putting three JSON data? it gave a value that is a mix and match of several different values
        # Looking at this from a different angle, it might be due to the fact that hidden states saved in datastore havent learned to 
        # distinguish themselves or 
        # we have established that the model can attend to the padded spaces, so that isnt a problem, problem is the retrieval step
        # if we get all the json data out?
        if self.csv_unlimiformer:
            # with open("data_final_data1/config_data_2.json", "r") as f:
            #     text = f.read()
            #     import json
            #     parsed_data = json.loads(text)
            #     update_lengths = parsed_data["update_length"]
            with open("data_final_data1/config_data.json", "r") as f:
                text = f.read()
                import json
                parsed_data = json.loads(text)
                segment_lengths = parsed_data["segment_length"]
        ind_up = 0
        pre_len = 0
        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in window_indices:
            if self.csv_unlimiformer:
                # update_start_ind += update_lengths[ind_up]
                ind_up += 1
            # if self.csv_unlimiformer:
            #     update_start_ind = 0
            #     update_end_ind = None
            logger.info(f'Encoding {context_start_ind} to {context_end_ind} out of {input_ids.shape[-1]}')
            if self.csv_unlimiformer:
                # FAILIURE: An experiment to see if hidden states can link better if they are spaced more widely
                if self.apply_boundary:
                    self.boundary = 40
                    chunk = torch.cat((torch.ones(1, self.boundary).to(torch.int64), prefix_input_id, input_ids[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    chunk_attention_mask = torch.cat((torch.zeros(1, self.boundary), torch.ones_like(prefix_input_id), attention_mask[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    chunk_position_ids = torch.cat((torch.zeros(1, self.boundary), torch.linspace(1, self.boundary, len(chunk[0]) - self.boundary).to(torch.int64).unsqueeze(0)), dim=1).to(self.device)
                    update_start_ind += self.boundary
                    update_end_ind += self.boundary
                else:
                    # Encode each segment of the data seperately: position ids start from 0 to len(segment) (Assumption doesnt matter)
                    # chunk = torch.cat((torch.ones(1, 2048).to(torch.int64), prefix_input_id, input_ids[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    self.ind_up = ind_up
                    if ind_up >= 2:
                        prefix_input_id = self.tokenizer.encode(self.prefix(ind_up), add_special_tokens=False, return_tensors="pt")
                        # chunk = torch.cat((torch.ones(1, 2 * (ind_up - 1)).to(torch.int64), input_ids[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                        chunk = torch.cat((prefix_input_id, input_ids[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                        # chunk_attention_mask = torch.cat((torch.ones(1, 2 * (ind_up - 1)), attention_mask[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                        chunk_attention_mask = torch.cat((torch.ones_like(prefix_input_id), attention_mask[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                        if False:#ind_up >=3:
                            pos_context = torch.tensor([self.tokens_ind]).repeat(self.ind_up-2, 1)
                            for i in range(pos_context.shape[0]):
                                pos_context[i] += 12 * i
                            pos_context = pos_context.reshape((-1))

                            if ind_up == 20:
                                chunk_position_ids = torch.cat((torch.zeros(2), pos_context, torch.arange(1 + (ind_up-2) * 11, (ind_up)*11))).unsqueeze(0).to(self.device)
                            else:
                                chunk_position_ids = torch.cat((torch.zeros(2), pos_context, torch.arange(1 + (ind_up-2) * 11, 1+(ind_up)*11))).unsqueeze(0).to(self.device)
                        else:
                            chunk_position_ids = None
                    else:
                        chunk = input_ids[:, context_start_ind:context_end_ind].to(self.device)
                        chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind].to(self.device)
                        chunk_position_ids = None #torch.arange(0 + pre_len, pre_len + len(chunk[0])).unsqueeze(0).to(self.device)
                    # chunk = torch.cat((prefix_input_id, input_ids[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    # chunk_attention_mask = torch.cat((torch.zeros(1, 2048), torch.ones_like(prefix_input_id), attention_mask[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    # chunk_attention_mask = torch.cat((torch.ones_like(prefix_input_id), attention_mask[:, context_start_ind:context_end_ind]), dim = 1).to(self.device)
                    # if ind_up > 1:
                    #     chunk_position_ids = torch.cat((torch.zeros(2048 + len(prefix_input_id[0])), torch.arange(0, len(chunk[0]) + update_start_ind - 2048 - 2), torch.arange(pre_len, pre_len - update_start_ind + 1))).unsqueeze(0).to(self.device)
                    # else:
                    # pre_len += -update_start_ind +  
                    # self.num_retrieved += -update_start_ind
                    # self.num_retrieved += len(chunk[0])
                    self.input_ids_full.append(chunk[0])
                logger.info(f'{(self.tokenizer.decode(chunk[0]))}')
                # logger.info(f'{self.tokenizer.decode(self.input_ids_full)}, {len(self.input_ids_full + self.input_ids_full_extra)}')
            else:
                chunk = input_ids[:, context_start_ind:context_end_ind].to(self.device)
                chunk_attention_mask = attention_mask[:, context_start_ind:context_end_ind].to(self.device)
                chunk_position_ids = None
            with torch.inference_mode():
                _ = self.model(chunk, attention_mask=chunk_attention_mask, labels=dummy_labels, position_ids=chunk_position_ids) # , return_dict=True, output_hidden_states=True)
            if self.use_datastore:
                # TODO: verify with BART as well
                # hidden_states_to_index = [hidden_states.encoder_last_hidden_state] # list of length 1 of (batch, chunked_source_len, dim)
                hidden_states_to_index = [
                    layer_capturer.captured for layer_capturer in self.activation_capturer
                ] 
                # hidden_states_to_index = list(hidden_states.hidden_states)[:-1][self.layer_begin:self.layer_end]
                to_add = [state[:, update_start_ind:update_end_ind].detach() for state in hidden_states_to_index]
                logger.info(f'{self.tokenizer.decode(chunk[0][update_start_ind:update_end_ind])}')
                to_apply_mask = chunk_attention_mask[:, update_start_ind:update_end_ind]
                # to_apply_mask = to_apply_mask.log().to(to_add[0].dtype)
                to_apply_mask = to_apply_mask.to(to_add[0].dtype)
                if True:#self.ind_up != 18 and self.ind_up != 15:
                    self.num_retrieved += len(to_apply_mask[0])
                if not self.reconstruct_embeddings:
                    to_add_embeddings = to_add
                    if not self.gpu_datastore:
                        to_add_embeddings = [states.cpu() for states in to_add_embeddings]
                        to_apply_mask = to_apply_mask.cpu()
                    for i, layer_states in enumerate(to_add_embeddings):
                        layer_states = layer_states * to_apply_mask.unsqueeze(-1) # [1, seq_len, dim]
                        if True:#self.ind_up != 18 and self.ind_up != 15:
                            self.hidden_states[i].append(layer_states.to(self.datastore_device))
                        # if self.csv_unlimiformer and self.ind_up <= 50:# not self.not_first_encoding_pass:
                            # self.hidden_layer_our[i] = layer_states.to(self.datastore_device)
                            # if self.ind_up == 2:
                            #     self.hidden_layer_our_bits[i] = self.hidden_layer_our[i][:, self.tokens_ind].to(self.datastore_device)
                            # elif self.ind_up >2:
                            #     self.hidden_layer_our_bits[i] =  torch.cat((self.hidden_layer_our_bits[i], self.hidden_layer_our[i][:, self.tokens_ind]), dim = -2).to(self.datastore_device)
                            # self.hidden_layer_our[i] = layer_states.to(self.datastore_device)
                        if self.csv_unlimiformer and not self.not_first_encoding_pass:
                            self.hidden_layer_our_anchor[i] = layer_states[:, 0:2].reshape((1, 2, -1)).to(self.datastore_device)
                        # if self.csv_unlimiformer and self.not_first_encoding_pass:
                        #     self.hidden_layer_our_anchor[i]  = torch.cat((self.hidden_layer_our_anchor[i] , layer_states[:, 0:1].reshape((1, 1, -1)).to(self.datastore_device)), dim = -2)
                logger.info(f'using only the first hidden states, stablising first also, so discard baking, trying to make Lee the stabliser, third in book, also adding diff position ids, making Zelensky stabiliser')
                if self.csv_unlimiformer:
                    self.not_first_encoding_pass = True
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

    def window_indices(self, total_seq_len, prefix_len=None):
        # Copied from SLED (Ivgy et al., 2022)
        # https://github.com/Mivg/SLED/blob/main/sled/modeling_sled.py#L467
        if not self.csv_unlimiformer and total_seq_len <= self.model_encoder_max_len:
            return [(0, total_seq_len, 0, total_seq_len)]
        else:
            results = []
            # if self.chunk_overlap == 0:
            #     stride = self.model_encoder_max_len
            if self.csv_unlimiformer:
                with open("data_final_data1/config_data.json", "r") as f:
                    text = f.read()
                    import json
                    parsed_data = json.loads(text)
                    segment_lengths = parsed_data["segment_length"]

                results = []
                context_start = 0
                # context_end = segment_lengths[0] - 2
                context_end = segment_lengths[0]
                # results.append((0, None, 0, None))
                # results.append((context_start, context_end + 1, -segment_lengths[0] + 1, None))  
                # results.append((0, None, 0, None))
                # return results
                results.append((context_start, context_end - 1, -segment_lengths[0], None))  
                # results.append((context_start, context_end - 1, prefix_len, prefix_len + segment_lengths[0] - 1))  

                for i in range(1, len(segment_lengths)):
                    # context_start = context_start + (segment_lengths[i - 2] if i>1 else 0)
                    context_start = context_start + segment_lengths[i - 1] - 1
                    context_end = context_end + segment_lengths[i] - 1
                    # context_end = context_end + segment_lengths[i] - 1
                    
                    update_start_ind = -segment_lengths[i] + 1
                    update_end_ind = None
                    # update_end_ind = segment_lengths[i] - 1
                    
                    cs, ce, us, ue = context_start, context_end - 1, update_start_ind, update_end_ind
                    # cs, ce, us, ue = context_start - (1 if i > 1 else 0), context_end + 1, update_start_ind - 1, update_end_ind
                    # cs, ce, us, ue = context_start - 1, context_end - 1, update_start_ind + prefix_len, update_end_ind + prefix_len - 1
                    # cs, ce, us, ue = context_start, context_end - 1, update_start_ind + prefix_len, update_end_ind + prefix_len
                    results.append((cs, ce, us, ue))
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
    def suffix(self, val=""):
        # return '[INST] Extract the value for the key from the JSON data given above. Print only the value. Nothing else.\nKey: "' + val + '"\nCorresponding value: [\INST]'
        # return '<<SYS>>\n You are a helpful assistant. Answer with detailed responses according to the entire instruction or question. \n<</SYS>>\n\n[INST] Extract the value for the key from the JSON data given above. Print only the value. Nothing else.\nKey: "' + val + '"\nCorresponding value: [\INST]'
        # return '<<SYS>>\n You are a helpful assistant. Answer with detailed responses according to the entire instruction or question. \n<</SYS>>\n\n[INST] Extract the value corresponding to the specified key in the JSON object below.\nKey: "a"\nCorresponding value: [\INST]'
        # return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with concise and very very short responses according to the instruction. \n<</SYS>>\n\nFrom the JSON data above, key "' + val + '", value: \n[/INST]'
        return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with short responses according to the question. \n<</SYS>>\n\n'
        # return '-- Using valid SQLite, answer the following questions for the tables provided above.\n\n'
        # return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with detailed responses according to the entire instruction or question.\n<</SYS>>\n\nExtract the value corresponding to the key "' + val + '" in the JSON object below.\n [/INST' # intentionally deleted ']'
        # return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with detailed responses according to the entire instruction or question.\n<</SYS>>\n\nExtract the value corresponding to the key "' + val + '" in the JSON object below.\n [/INST]' # intentionally deleted ']'
        # return '<s>[INST] <<SYS>>\nYou are a helpful assistant. Answer with detailed responses according to the entire instruction or question.\n<</SYS>>\n\nExtract the value corresponding to the specified key in the JSON object below.\nKey: "' + val + '"\nCorresponding value: [/INST]'
        # return '<<SYS>>\n You are a helpful assistant. Answer with detailed responses according to the entire instruction or question. \n<</SYS>>\n\n[INST] How do I read a map? [\INST]'
    
    # format copied from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py
    def suffix2(self, i=0):
        # return 'From the JSON data above, key "' + val + '", value: \n[/INST]'
        # return 'From the JSON data above, key "' + val + '", value: \n[/INST] Sure! Here is the value associated with the key "' + val + '"'
        # return 'Read the above JSON object and tell the value corresponding to the specified key in the given JSON object above\nKey: "' + val + '"\nCorresponding value: [\INST]'
        # return '-- how many stadiums in total?\n\nSELECT'
        with open("data_final_data1/original_data.txt", 'r') as f:
            cont = f.read()
        cont = cont.split(',')
        if i==0 or i==1:
            if i%2:
                return 'Based on the above numbered list of facts, can you tell me what ' + cont[i//2].split(' ')[3] + ' is doing?[/INST]'
            else:
                return 'Based on the above numbered list of facts, can you tell me who is ' + cont[i//2].split(' ')[5] + '?[/INST]'
        else:
            if i%2:
                return 'Based on the above numbered list of facts, can you tell me what ' + cont[i//2].split(' ')[4] + ' is doing?[/INST]'
            else:
                return 'Based on the above numbered list of facts, can you tell me who is ' + cont[i//2].split(' ')[6] + '?[/INST]'
        
        if i == 0:
            # return 'Based on the above numbered list of facts, can you tell me the value associated with key "ed1ef023-f1bb-4bf9-8b54-910bbd1c2750"?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "aj12"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Oppenheimer is doing?[/INST]'
        elif i==1:
            # return 'Based on the above numbered list of facts, can you tell me the value associated with key "62eff267-d0e6-4c65-81cb-6b6b7db9c63b"?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "we34"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Rithik is doing?[/INST]'
        elif i==2:
            # return 'Based on the above numbered list of facts, can you tell me the key associated with value "f3142b5e-ccc7-49c2-ab5f-fbf402b2becd"?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "dh83"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is baking?[/INST]'
        elif i==3:
            # return 'Based on the above numbered list of facts, can you tell me the key associated with value "a4eacd0b-5962-46d3-9877-0f0e9c5b892f"?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "es52"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is cycling?[/INST]'
        elif i==4:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "sw45"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is sleeping?[/INST]'
        elif i==5:
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "hs57"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Leechenbaum is doing?[/INST]'
        elif i==6:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "hs57"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Zelensky is doing?[/INST]'
        elif i==7:
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "ex88"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is relaxing?[/INST]'
        elif i==8:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "fe80"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Murugan is doing?[/INST]'
        elif i==9:
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "rs52"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is eating?[/INST]'
        elif i==10:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "hr79"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Vaibhav is doing?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me who is in Siberia?[/INST]'
        elif i==11:
            # return 'Based on the above numbered list of facts, can you tell me who is in India?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "er34"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is drinking?[/INST]'
        elif i==12:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "op18"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Rohit is doing?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me who is in Mexico?[/INST]'
        elif i==13:
            # return 'Based on the above numbered list of facts, can you tell me who is in Lithuania?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "dw43"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is cooking?[/INST]'
        elif i==14:
            # return 'Based on the above numbered list of facts, can you tell me what is key corresponding to "dr45"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me what Rakesh is doing?[/INST]'
        elif i==15:
            # return 'Based on the above numbered list of facts, can you tell me who is in Lithuania?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me what is value corresponding to "ir83"?[/INST]'
            return 'Based on the above numbered list of facts, can you tell me who is bowling?[/INST]'
            # return 'Based on the above numbered list of facts, can you tell me who is in America?[/INST]'
        # return 'Based on the above numbered list of facts, can you tell me what the value is for "' + val + '"?[/INST]'
        # return 'Can you tell me, from the JSON key-value pairs given above, the value for the key "'+ val + '"?[\INST]'
        
    # format copied from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py
    def prefix(self, leni):
        if True:
            return ' '.join(['Fact'] * 2)
        elif self.ind_up == 2:
            return ' '.join(['Fact'] * (2 + 11))
        else:
            return ' '.join(['Fact'] * (2 + 11 + len(self.tokens_ind) * (self.ind_up - 2)))
        if leni == 1:
            # return '{"A" : "B"},'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
        if leni == 2:
        #     # return ', , , , , , Williamson is baking,'
        #     # return '{"A" : "B"},'
            # return 'Fact number 2:'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact number 1: Williamson is baking,'
            # return '{"c7ec84a3-7d9c-4446-b8d5-1175e77e894e" : "cd870d30-cfa5-42f4-8573-f6508a9f581a"},'
        #     # return '{"A" : "B"},'
        elif leni == 3:
        #     # return ', , , , , , , , , , , Oppenheimer is cycling,'
            # return 'William is baking, Oppenheimer is cycling,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'

            # return 'Fact, Fact number 2: Oppenheimer is cycling,'
            # return 'Fact number 3:'
            # return 'William, William is baking,'
            # return 'Williamson is baking, Oppenheimer is cycling,'
            # return '{"c7ec84a3-7d9c-4446-b8d5-1175e77e894e" : "cd870d30-cfa5-42f4-8573-f6508a9f581a"}, {"293daef2-7a2f-4b0f-913f-a2df84e91ada" : "31d23a75-28e2-4c0f-a75f-0dc88e0f93e4"},'
        #     # return ', A is B,'
        # #     return ',{"A" : "B"},'
        elif leni == 4:
        #     return 'Ashwin, Ashwin, Williamson is baking,'
        #     # return ', , , , , , , , , , , , , , , , , , Leechenbaum is painting,'
            # return '{"c7ec84a3-7d9c-4446-b8d5-1175e77e894e" : "cd870d30-cfa5-42f4-8573-f6508a9f581a"}, {"293daef2-7a2f-4b0f-913f-a2df84e91ada" : "31d23a75-28e2-4c0f-a75f-0dc88e0f93e4"}, {"8e8e0ad4-9053-4456-a351-e1642f304fb6" : "b588c603-d428-48f0-ab08-b452355c348f"},'
            # return ', , Leechenbaum is painting,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 3: Leechenbaum is painting,'
            # return 'Williamson is baking, Oppen Leechenbaum is painting,'
            # return ','
            # return 'Fact number 4:'
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting,'
        #     # return ',, A is B,'
        # #     return ',,{"A" : "B"},'
        elif leni == 5:
            # return '{"c7ec84a3-7d9c-4446-b8d5-1175e77e894e" : "cd870d30-cfa5-42f4-8573-f6508a9f581a"}, {"293daef2-7a2f-4b0f-913f-a2df84e91ada" : "31d23a75-28e2-4c0f-a75f-0dc88e0f93e4"}, {"8e8e0ad4-9053-4456-a351-e1642f304fb6" : "b588c603-d428-48f0-ab08-b452355c348f"}, {"5abe49de-9c8a-4a58-b7d1-83b76447e01d" : "f2028aea-b913-42dc-8e32-ac3f75a89415"},'
            # return ', , , Zelensky is relaxing,'
            # return ','
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 4: Zelensky is relaxing,'
            # return 'Zelensky is relaxing,'
            # return 'Fact number 5:'
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing,'
        elif leni == 6:
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing, Murugan is eating,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 5: Murugan is eating,'
        elif leni == 7:
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing, Murugan is eating, Mohan is drinking,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 6: Mohan is drinking,'
        elif leni == 8:
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing, Murugan is eating, Mohan is drinking, Shaheen is talking,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 7: Shaheen is talking,'
        elif leni == 9:
            # return 'William is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing, Murugan is eating, Mohan is drinking, Shaheen is talking,'
            return 'Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact Fact'
            # return 'Fact, Fact number 8: Rakesh is bowling,'

        # elif leni == 6:
            # return '{"c7ec84a3-7d9c-4446-b8d5-1175e77e894e" : "cd870d30-cfa5-42f4-8573-f6508a9f581a"}, {"293daef2-7a2f-4b0f-913f-a2df84e91ada" : "31d23a75-28e2-4c0f-a75f-0dc88e0f93e4"}, {"8e8e0ad4-9053-4456-a351-e1642f304fb6" : "b588c603-d428-48f0-ab08-b452355c348f"}, {"5abe49de-9c8a-4a58-b7d1-83b76447e01d" : "f2028aea-b913-42dc-8e32-ac3f75a89415"}, {"40fc0325-7953-4007-9c94-75026035fd6b" : "718275b1-a3fe-4110-9343-7a457caa9104"},'
            # return ', , , , Murugan is eating,'
            # return 'Williamson is baking, Oppenheimer is cycling, Leechenbaum is painting, Zelensky is relaxing, Murugan is eating,'
            # return 'Ashwin, Ashwin, Ashwin, Williamson is baking,'
        # return ' '
        # return 'Williamson is baking in America,'
        # return ''.join([','] * (leni-2)) + '{"A" : "B"},'
        # return "JSON data: {"
        # return "<s>[INST] <<SYS>>\n\n<</SYS>>\n\nJSON data: {"

    def pre_generate_hook(self, input_ids, **kwargs):
        if 'attention_mask' not in kwargs:
            kwargs['attention_mask'] = torch.ones_like(input_ids)
        self.reset_memory(input_ids, kwargs['attention_mask'])
        if self.csv_unlimiformer:
            torch.save(self.attention_weights, './attn_wts_119.pt')
            torch.save(self.input_ids_full, './inputs_119.pt')      
            # exit(0)    
        new_kwargs = kwargs
        if 'attention_mask' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
            new_kwargs['attention_mask'] = kwargs['attention_mask'][:, :self.actual_model_window_size].to(self.device)
        new_kwargs['use_cache'] = True
        if self.is_encoder_decoder:
            input_ids_prefix = input_ids[:, :self.actual_model_window_size]
        else:
            if self.csv_unlimiformer:
                with open("data9/sampled_keys.txt", "r") as f:
                    key_vals = f.read().split("\n")
                vals_pred = []
                for i in range(40):
                    # encode the suffix prompt
                    self.curr_key = i
                    input_ids_prefix = self.tokenizer.encode(self.suffix(" "), add_special_tokens=False, return_tensors="pt")
                    # prepare the attention mask for it; we additionally also prepare space for retrieved tokens since the suffix is short
                    # if we do not consider initial padding, then we would not have enough space for retrieval space
                    # this space allocated for retrieved keys is given by self.num_retrieved
                    self.input_ids_full += input_ids_prefix[0]
                    new_kwargs["attention_mask"] = torch.cat((torch.zeros(1, self.num_retrieved), torch.ones(1, len(input_ids_prefix[0]))), dim = 1).to(self.device)
                    # pad the input (doesnt matter with which since attention mask is 0)
                    input_ids_prefix = torch.cat((torch.ones(1, self.num_retrieved).to(torch.int64), input_ids_prefix), dim = 1)
                    input_ids_prefix = input_ids_prefix.to(self.device)
                    vals_pred.append(self.original_generate_func(input_ids_prefix, **new_kwargs))
                    # attn_wts = torch.stack(self.attention_weights, dim = 0)
                    # inputs = torch.stack(self.input_ids_full, dim = 0)
                    # torch.save(self.attention_weights, './attn_wts_106_'+str(i)+'.pt')
                    # torch.save(self.input_ids_full + self.input_ids_full_extra, './inputs_106_'+str(i)+'.pt')
                    self.attention_weights = []
                    self.input_ids_full_extra = []
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
                        # in the initial forward pass, also set the position ids to indicate padding and then input
                        question_len = (attention_mask[0] == 1).sum(dim=0)
                        self.is_second_test_decoding_step = False
                        self.num_generated = 0
                        # The initial position ids will determine
                        kwargs["position_ids"] = torch.cat((torch.zeros(self.num_retrieved), torch.arange(1, question_len + 1))).unsqueeze(0).to(self.device)
                        self.rotation_retrieved = question_len + self.gap
                        self.what_num = question_len + 1
                # else:
                #     if self.csv_unlimiformer:
                #         self.input_ids_full_extra += input_ids[0]
                # else:
                    # From the next generation take in all the values, they are now not padded but retrieved hidden states
                    # Issue: Need this to be done only after a particular number of layers when we fetch the retrieved keys
                    # if self.csv_unlimiformer:
                    #     attention_mask = torch.ones_like(attention_mask)
                
                # When this suffix2 goes in, it will start to retrieve keys from the datastore
                # this indicates to the model the start of answering, so now model accumulates all information
                # from prefix to the queries, we now use this accumulated-info query to get relevant keys and store in
                # We assume previously it was just building up context
                # can experiment with putting everything here only, since we are leaving space for retrieved keys
                # if self.csv_unlimiformer and self.is_second_test_decoding_step:
                #     input_ids = self.tokenizer.encode(self.suffix2(self.curr_key), add_special_tokens=False, return_tensors="pt").to(self.device)
                #     attention_mask = torch.cat((attention_mask[:, :-1], torch.ones_like(input_ids).to(self.device)), dim=1)
                #     kwargs["position_ids"] = torch.arange(20 + int(kwargs["position_ids"][0]) + 1, 20 + int(kwargs["position_ids"][0]) + len(input_ids[0]) + 1).unsqueeze(0).to(self.device)
                # if self.csv_unlimiformer and not self.is_second_test_decoding_step and not self.is_first_test_decoding_step:
                #     input_ids_temp = self.tokenizer.encode(self.suffix2(self.curr_key), add_special_tokens=False, return_tensors="pt").to(self.device)
                #     attention_mask = torch.cat((attention_mask[:, :-1], torch.ones_like(input_ids_temp).to(self.device)), dim=1)
                #     kwargs["position_ids"] = torch.arange(20 + int(kwargs["position_ids"][0]) + len(input_ids_temp[0]), 20 + int(kwargs["position_ids"][0]) + len(input_ids_temp[0]) + 1).unsqueeze(0).to(self.device)

                # Here we send all the suffix that we need to push and expect attention to retrieved keys
                if self.csv_unlimiformer and self.is_second_test_decoding_step:
                    input_ids_suffix = self.tokenizer.encode(self.suffix2(self.curr_key), add_special_tokens=False, return_tensors="pt")
                    self.curr_suffix_len = len(input_ids_suffix[0])
                    if self.one_by_one:
                        if self.num_generated == (len(input_ids_suffix[0]) + 1):
                            self.is_second_test_decoding_step = False
                        else:
                            input_ids = input_ids_suffix[:, self.num_generated - 1].unsqueeze(0).to(self.device)
                            kwargs["position_ids"] = torch.arange(self.num_retrieved + int(kwargs["position_ids"][0]), self.num_retrieved + int(kwargs["position_ids"][0]) + 1).unsqueeze(0).to(self.device)
                            # kwargs["position_ids"] = torch.arange(self.gap * 2 + int(kwargs["position_ids"][0]), self.gap * 2 + int(kwargs["position_ids"][0]) + 1).unsqueeze(0).to(self.device)
                    else:
                        input_ids = input_ids_suffix.to(self.device)
                        kwargs["position_ids"] = torch.arange(int(kwargs["position_ids"][0]), int(kwargs["position_ids"][0]) + len(input_ids[0])).unsqueeze(0).to(self.device)
                        attention_mask = torch.cat((attention_mask, torch.ones(1, len(input_ids[0]) - 1).to(self.device)), dim = 1)

                if self.csv_unlimiformer and not self.is_second_test_decoding_step and not self.is_first_test_decoding_step:
                    if self.one_by_one:
                        kwargs["position_ids"] = torch.arange(self.num_retrieved + int(kwargs["position_ids"][0]), self.num_retrieved + int(kwargs["position_ids"][0]) + 1).unsqueeze(0).to(self.device)
                        # kwargs["position_ids"] = torch.arange(self.gap * 2 + int(kwargs["position_ids"][0]), self.gap * 2 + int(kwargs["position_ids"][0]) + 1).unsqueeze(0).to(self.device)
                    else:
                        attention_mask = torch.cat((attention_mask, torch.ones(1, self.curr_suffix_len - 1).to(self.device)), dim = 1)
                        kwargs["position_ids"] = torch.arange(int(kwargs["position_ids"][0]) + self.curr_suffix_len, int(kwargs["position_ids"][0]) + self.curr_suffix_len + 1).unsqueeze(0).to(self.device)

                # print(attention_mask, kwargs["position_ids"])
                if not kwargs.get('past_key_values') is None and self.csv_unlimiformer:
                    self.input_ids_full_extra += input_ids[0]
                # el
                if input_ids is not None:
                    self.input_ids_size += 1
                    # to keep track of the number of generations as done in earlier code
                    if self.csv_unlimiformer:
                        self.num_generated += 1
                    # logger.info(f'{self.tokenizer.decode(self.input_ids_full + self.input_ids_full_extra)}, {len(self.input_ids_full + self.input_ids_full_extra)}')
                if kwargs.get('decoder_input_ids') is not None:
                    self.generated_input_ids = torch.cat([self.generated_input_ids, kwargs['decoder_input_ids']], axis=-1)
            logger.info(f'"Pre Forward Hook", {self.tokenizer.decode(input_ids[0])}, {len(input_ids[0])}')
        # the position id is used to calculate by how much to rotate the key and query newly added
        # So according to our thing, we can start the position ids to start from any number
        result = self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
        if not self.is_input_encoding_pass and not self.is_first_test_decoding_step and self.csv_unlimiformer:
            logits = result.logits
            top_k_values, top_k_indices = torch.topk(logits, k=10, dim=-1)
            top_k_token_ids = top_k_indices.tolist()[0][0]
            top_k_tokens = [self.tokenizer.convert_ids_to_tokens(token_id) for token_id in top_k_token_ids]
            logger.info(f'{top_k_tokens}')
            logger.info(f'{top_k_values}')

        if self.csv_unlimiformer:
            # self.is_second_test_decoding_step = False
            if not self.one_by_one:
                if self.is_second_test_decoding_step:
                    self.is_second_test_decoding_step = False
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
                # The below logs cur_layer_num only range is [0, 17] where layer_begin is 22
                # print(self.is_first_test_decoding_step, cur_layer_num, self.is_input_encoding_pass, self.layer_begin)
                if not self.csv_unlimiformer or self.is_first_test_decoding_step or self.is_input_encoding_pass:
                    # if it's the prefix that is being input then apply the attention mask input to the function
                    # logger.info(f'{cur_layer_num}, {hidden_states.shape}, {self.hidden_layer_our[cur_layer_num].shape}')
                    # logger.info(f'{hidden_states.shape}')
                    # if self.is_input_encoding_pass and not self.not_first_encoding_pass:
                    #     self.comma_hidden_state[cur_layer_num] = hidden_states[:, -1].reshape((1, 1, -1))
                    #     # print(hidden_states.shape)
                    #     # print(self.comma_hidden_state.shape)
                    #     # torch.save(hidden_states[-1], './hidden_110.pt')
                    # if self.is_input_encoding_pass and self.not_first_encoding_pass:
                    #     hidden_states[:, 0:2*(self.ind_up - 1)] = self.comma_hidden_state[cur_layer_num].repeat(1, 2 * (self.ind_up - 1), 1)
                    if self.not_first_encoding_pass and self.is_input_encoding_pass:
                        # topk = self.hidden_layer_our[cur_layer_num].shape[-2]
                        topk = self.hidden_layer_our_anchor[cur_layer_num].shape[-2]
                        if False:#self.ind_up >= 3:
                            topk += self.hidden_layer_our_bits[cur_layer_num].shape[-2]
                            hidden_states = torch.cat([self.hidden_layer_our_anchor[cur_layer_num], self.hidden_layer_our_bits[cur_layer_num], self.hidden_layer_our[cur_layer_num], hidden_states[:,topk:]], dim=-2)
                        else:
                            hidden_states = torch.cat([self.hidden_layer_our_anchor[cur_layer_num], hidden_states[:,topk:]], dim=-2)
                            # hidden_states = torch.cat([self.hidden_layer_our_anchor[cur_layer_num], self.hidden_layer_our[cur_layer_num], hidden_states[:,topk:]], dim=-2)
                    kwargs['output_attentions'] = True
                    result = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                    if self.is_input_encoding_pass:
                        attn_wts = result[1].squeeze(0).squeeze(1)
                        self.attention_weights.append(attn_wts)
                    # result = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                else:
                    attention_mask = torch.ones_like(attention_mask)
                    # kwargs['output_attentions'] = True
                    result = original_cross_attn_forward_func(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
                    # attn_wts = result[1].squeeze(0).squeeze(1)
                    # self.attention_weights.append(attn_wts)
                    # indexes = torch.topk(attn_wts, attn_wts.shape[1]).indices
                    # for index in range(len(indexes)):
                    #     for j in range(attn_wts.shape[1]):
                    #         logger.info(f'({self.tokenizer.decode(self.input_ids_full[indexes[index][j]])})')
                    #     logger.info(f'\n')
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
            # if self.is_input_encoding_pass and self.csv_unlimiformer and self.not_first_encoding_pass:
                # hidden_states = self.comma_hidden_state.repeat(1, self.ind_up - 1, 1).to(self.device)
                # print(self.comma_hidden_state.shape)
                # print(hidden_states.shape)
                # indices = torch.arange(0, hidden_states.shape[1]).repeat(40, 1).unsqueeze(0)
                # embeddings = torch.take_along_dim(input=hidden_states.unsqueeze(1), 
                #         indices=indices.unsqueeze(-1).to(hidden_states.device), dim=-2)
                # embeddings = embeddings.reshape(1, -1, self.num_heads, *embeddings.shape[2:])

                # attention_layer_list = self.get_kv_projections(self.layer_begin, self.layer_end)
                # k_proj_layer = [layers[0] for layers in attention_layer_list][self.cur_decoder_layer_index]
                # v_proj_layer = [layers[1] for layers in attention_layer_list][self.cur_decoder_layer_index]

                # retrieved_keys, retrieved_values = self.post_process_retrieved(embeddings, k_proj_layer, v_proj_layer, indices)
                # retrieved_keys = retrieved_keys.flatten(0, 1)
                # retrieved_values = retrieved_values.flatten(0, 1)
                # topk = retrieved_keys.shape[2]
                # self.cur_layer_key_value_placeholder[0] = torch.cat([retrieved_keys, self.cur_layer_key_value_placeholder[0][:,:,topk:]], dim=-2)
                # self.cur_layer_key_value_placeholder[1] = torch.cat([retrieved_values, self.cur_layer_key_value_placeholder[1][:,:,topk:]], dim=-2)
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
                    topk = self.num_retrieved
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
                    # _, top_search_key_indices = self.datastore[datastore_index].search(datastore_query, k=topk)
                    # top_search_key_indices = torch.sort(top_search_key_indices).values
                    top_search_key_indices = torch.arange(0, self.num_retrieved).repeat(40, 1).unsqueeze(0)
                    # self.embeddings: (batch,              src_len, dim)
                    # indices:         (batch, beam * head, actual_model_window_size)
                    # embeddings: (batch, beam * head, actual_model_window_size, dim)
                    embeddings = torch.take_along_dim(input=self.hidden_states[datastore_index].unsqueeze(1), 
                        indices=top_search_key_indices.unsqueeze(-1).to(self.hidden_states[datastore_index].device), dim=-2)
                    embeddings = embeddings.to(self.device)
                # (batch, beam, head, actual_model_window_size)
                # top_search_key_scores = top_search_key_scores.reshape(batch_size, -1, *top_search_key_scores.shape[1:])
                top_search_key_indices = top_search_key_indices.reshape(batch_size, -1, *top_search_key_indices.shape[1:])
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
        retrieved_keys = retrieved_keys.flatten(0, 1)[:,:,:topk]
        retrieved_values = retrieved_values.flatten(0, 1)[:,:,:topk]
        self.cur_layer_key_value_placeholder[0] = torch.cat([retrieved_keys, self.cur_layer_key_value_placeholder[0][:,:,topk:]], dim=-2)
        self.cur_layer_key_value_placeholder[1] = torch.cat([retrieved_values, self.cur_layer_key_value_placeholder[1][:,:,topk:]], dim=-2)
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

                   
        # `this_layer_prompt_keys`:   (batch,          head,    source_len, dim)
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
        ####### The rotation given to q should not matter
        attention = self.model.base_model.layers[-1].self_attn
        num_generated = min(self.input_ids_size - self.prompt_input_ids.shape[1], self.actual_model_window_size)
        # cos, sin = attention.rotary_emb(query, seq_len=self.num_generated)
        cos, sin = attention.rotary_emb(query, seq_len=num_generated)
        # # Experiment 1
        # cos, sin = attention.rotary_emb(query, seq_len = 1)
        cos = cos[:,:,-1]  # [1, 1, dim]
        sin = sin[:,:,-1]  # [1, 1, dim]
        # # cos = cos[-1].unsqueeze(0).unsqueeze(0)  # [bs, 1, seq_len, dim]
        # # sin = sin[-1].unsqueeze(0)  # [bs, 1, seq_len, dim]
        query = (query * cos) + (self.rotate_half(query) * sin)

        k_proj = k_proj_weight.view(1, self.num_heads, query.shape[-1], k_proj_weight.shape[0]) # (1, num_heads, attn_dim, embed_dim)
        k_proj_l = k_proj[..., :k_proj.shape[-2] // 2, :]
        k_proj_r = k_proj[..., k_proj.shape[-2] // 2:, :]
        k_proj_rotated = torch.cat([-k_proj_l, k_proj_r], dim=-2)

        datastore_query = query.unsqueeze(-2) # (batch * beam, num_heads, 1, attn_dim)
        datastore_query = torch.matmul(datastore_query, k_proj + k_proj_rotated) # (batch * beam, num_heads, 1, embed_dim)
        # Experiment 1
        # datastore_query = torch.matmul(datastore_query, k_proj) # (batch * beam, num_heads, 1, embed_dim)
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
        if self.is_input_encoding_pass:
            # attention = self.model.base_model.layers[-1].self_attn
            # cos, sin = attention.rotary_emb(retrieved_values, seq_len=top_search_key_indices.shape[-1])
            # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            # retrieved_keys = (retrieved_keys * cos) + (self.rotate_half(retrieved_keys) * sin)
            return retrieved_keys, retrieved_values
        # attention = self.model.base_model.layers[-1].self_attn
        # cos, sin = attention.rotary_emb(retrieved_values, seq_len=self.hidden_states[0].shape[1])
        # cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        # sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        # if self.prompt_input_ids.shape[1] > self.actual_model_window_size:
        #     # scale the top key indices to the actual model window size, such that the model will not see
        #     # positional embeddings that did not appear at training time
        #     scaled_key_indices = ((top_search_key_indices / self.prompt_input_ids.shape[1]) * self.actual_model_window_size).int()
        # else:
        #     scaled_key_indices = top_search_key_indices
        # # top_search_key_indices = top_search_key_indices.to(cos.device)
        # scaled_key_indices = scaled_key_indices.to(cos.device)
        # cos = cos[scaled_key_indices]  # [bs, 1, seq_len, dim]
        # sin = sin[scaled_key_indices]  # [bs, 1, seq_len, dim]
        # retrieved_keys = (retrieved_keys * cos) + (self.rotate_half(retrieved_keys) * sin)
        # retrieved_keys = retrieved_keys
        
        # Experiment: We provide all the retrieved keys a constant rotation between prefix and suffix
        attention = self.model.base_model.layers[-1].self_attn
        # with open("data_final_data1/config_data_2.json", "r") as f:
        #     text = f.read()
        #     import json
        #     parsed_data = json.loads(text)
        #     update_lengths = parsed_data["update_length"]
        with open("data_final_data1/config_data.json", "r") as f:
            text = f.read()
            import json
            parsed_data = json.loads(text)
            segment_lengths = parsed_data["segment_length"]
        # cossin = [attention.rotary_emb(retrieved_values, seq_len=(update_lengths[i])) for i in range(len(segment_lengths))]
        # cossin = [attention.rotary_emb(retrieved_values, seq_len=(segment_lengths[i] - update_lengths[i])) for i in range(len(segment_lengths))]
        # cos = torch.cat([cossin[i][0].squeeze(1).squeeze(0) for i in range(len(cossin))], dim = 0)
        # sin = torch.cat([cossin[i][1].squeeze(1).squeeze(0) for i in range(len(cossin))], dim = 0)
        # sum_num = 0
        # for seg in segment_lengths:
        #     sum_num += seg
        # cos, sin = attention.rotary_emb(retrieved_values, seq_len=self.what_num + 1)
        cos, sin = attention.rotary_emb(retrieved_values, seq_len=self.what_num + self.num_retrieved + 1)
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[self.what_num + 1:]
        sin = sin[self.what_num + 1:]
        # cos = cos[-1]  # [1, 1, dim]
        # sin = sin[-1]  # [1, 1, dim]
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
    