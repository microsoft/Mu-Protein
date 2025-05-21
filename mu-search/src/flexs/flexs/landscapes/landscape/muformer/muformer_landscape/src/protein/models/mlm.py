import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils, checkpoint_utils

from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)

from fairseq.models.masked_lm import MaskedLMEncoder

from fairseq.modules import LayerNorm
from .attMapModel.transformer_sentence_encoder import TransformerSentenceEncoder,init_bert_params

logger = logging.getLogger(__name__)

@register_model('prot_mlm')
class ProtBaseModel(BaseFairseqModel):
    """Base class for encoder-only models for protein.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, args, encoder, classification_heads=nn.ModuleDict()):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.classification_heads = classification_heads

        assert isinstance(self.encoder, FairseqEncoder)
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)  # TODO whether triage bug

    def forward(self, src_tokens, segment_labels=None, extra_only=False, masked_only=False,
                classification_head_name=None, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        x, extra = self.encoder(src_tokens, segment_labels=segment_labels,
                                extra_only=False, masked_only=False, **kwargs)
        
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance.
        This model is 1) pre-trained for a transformer-encoder
        2) fine-tuning trained for various evaluation tasks include TAPE's 5 tasks
        3) capable of embedding & validating & testing & evaluation
        """

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        if args.pretrained_file:
            """
            Arguments:
                state_dict (dict): a dict containing parameters and
                    persistent buffers.
                strict (bool, optional): whether to strictly enforce that the keys
                    in :attr:`state_dict` match the keys returned by this module's
                    :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            make sure all arguments are present in older models
            """
            models, pharsed_args, _ = checkpoint_utils.load_model_ensemble_and_task(
                args.pretrained_file.split(os.pathsep),
                arg_overrides={
                    'data': getattr(args, "data", None), 
                    'eval_task': None,  # legacy
                    'max_positions': args.max_positions, 
                    'tokens_per_sample': args.tokens_per_sample,
                    'task': args.task,
                    'arch': args.arch,
                    'criterion': args.criterion,
                    '_name': None, # For compatibility
                    'eval_task': args.eval_task
                },
                suffix=getattr(args, "checkpoint_suffix", ""),
                task=task,
                strict=False,  # TODO
            )
            model = models[0]
            model.encoder.remove_head()
            print('Loaded pre-trained model encoder from ', args.pretrained_file.split(os.pathsep))
            logger.info(args)
            return cls(args, model.encoder)

        base_architecture(args)
        logger.info(args)
        encoder = ProtMaskedLMEncoder(args, task.dictionary)
        return cls(args, encoder)

    def prob(self, x):
        return self.encoder.prob(x)

    @property
    def vocab_size(self):
        return self.encoder.vocab_size

class ProtMaskedLMEncoder(MaskedLMEncoder):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = utils.get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, self.vocab_size, bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim, self.sentence_out_dim, bias=False
                )

    def forward(self, src_tokens, attn_bias, biased_layers, segment_labels=None, masked_tokens=None,
                extra_only=False, masked_only=False, **unused):

        x, attns, sentence_rep = self.sentence_encoder(
                    src_tokens,
                    attn_bias,
                    segment_labels=segment_labels,
                    biased_layers = biased_layers,
                ) # inner_states[-1]: (T, B, C)
        
        return x, { 'sentence_rep': sentence_rep, 'attns': attns }

    def prob(self, x):
        h_lm = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        if self.share_input_output_embed and hasattr(self.sentence_encoder.embed_tokens, "weight"):
            prob_lm = F.linear(h_lm, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            prob_lm = self.embed_out(h_lm)
        if self.lm_output_learned_bias is not None:
            prob_lm = prob_lm + self.lm_output_learned_bias
        return prob_lm 


@register_model_architecture('prot_mlm', 'prot_mlm_base')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)

@register_model_architecture("prot_mlm", "prot_mlm_1b")
def prot_1b_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_layers = getattr(args, "encoder_layers", 34)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 5120)
    base_architecture(args)
