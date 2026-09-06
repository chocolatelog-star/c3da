from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


_GRAPH_ADAPTER_PREFIX = "syntactic_graph_adapter."
_COMPOSITIONAL_RELATION_PREFIX = f"{_GRAPH_ADAPTER_PREFIX}compositional_"


def graph_checkpoint_uses_compositional_relations(parameter_names) -> bool:
    """Return the relation encoder mode encoded by a graph checkpoint."""
    return any(str(name).startswith(_COMPOSITIONAL_RELATION_PREFIX) for name in parameter_names)


def _checkpoint_graph_relation_mode(model_path: str | Path) -> bool | None:
    checkpoint = Path(model_path)
    safetensors_path = checkpoint / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    else:
        pytorch_path = checkpoint / "pytorch_model.bin"
        if not pytorch_path.is_file():
            return None
        state_dict = torch.load(str(pytorch_path), map_location="meta", weights_only=True)
        keys = list(state_dict)
    if not any(str(name).startswith(_GRAPH_ADAPTER_PREFIX) for name in keys):
        return None
    return graph_checkpoint_uses_compositional_relations(keys)


def _record_trace(trace, stage: str, value: torch.Tensor, axes=()):
    if trace is not None:
        trace.record(stage, value, axes=axes)


@dataclass
class GraphAdapterOutput:
    fused_hidden: torch.Tensor
    graph_hidden: torch.Tensor
    word_hidden: torch.Tensor
    residual: torch.Tensor
    gate: torch.Tensor


class SyntacticGraphAdapter(nn.Module):
    """A fixed-size relation-aware graph attention adapter for T5 encoder states."""

    def __init__(
        self,
        hidden_size: int,
        graph_hidden_size: int = 256,
        attention_heads: int = 4,
        head_size: int = 64,
        num_relations: int = 1,
        dropout: float = 0.1,
        num_dependency_relations: int | None = None,
        num_pos_pair_relations: int | None = None,
        compositional_dependency_vocab_size: int = 40,
        compositional_direction_vocab_size: int = 3,
        compositional_pos_vocab_size: int = 18,
        focus_enabled: bool = False,
        compositional_relation: bool = True,
    ):
        super().__init__()
        if graph_hidden_size != attention_heads * head_size:
            raise ValueError("graph_hidden_size must equal attention_heads * head_size")
        self.hidden_size = int(hidden_size)
        self.graph_hidden_size = int(graph_hidden_size)
        self.attention_heads = int(attention_heads)
        self.head_size = int(head_size)
        self.num_relations = max(1, int(num_relations))
        self.focus_enabled = bool(focus_enabled)
        self.compositional_relation = bool(compositional_relation)
        self.node_projection = nn.Linear(hidden_size, graph_hidden_size)
        self.query_projection = nn.Linear(graph_hidden_size, graph_hidden_size)
        self.key_projection = nn.Linear(graph_hidden_size, graph_hidden_size)
        self.value_projection = nn.Linear(graph_hidden_size, graph_hidden_size)
        self.relation_embedding = nn.Embedding(self.num_relations, graph_hidden_size)
        if self.compositional_relation:
            self.compositional_dependency_embedding = nn.Embedding(
                max(1, int(compositional_dependency_vocab_size)), graph_hidden_size
            )
            self.compositional_direction_embedding = nn.Embedding(
                max(1, int(compositional_direction_vocab_size)), graph_hidden_size
            )
            self.compositional_src_pos_embedding = nn.Embedding(
                max(1, int(compositional_pos_vocab_size)), graph_hidden_size
            )
            self.compositional_dst_pos_embedding = nn.Embedding(
                max(1, int(compositional_pos_vocab_size)), graph_hidden_size
            )
        self.dependency_bias = nn.Embedding(
            max(1, int(num_dependency_relations or num_relations)), attention_heads
        )
        self.pos_pair_bias = nn.Embedding(
            max(1, int(num_pos_pair_relations or num_relations)), attention_heads
        )
        self.graph_dropout = nn.Dropout(float(dropout))
        self.output_projection = nn.Linear(graph_hidden_size, hidden_size, bias=False)
        self.gate_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the standard child-module initialization contract."""
        self.node_projection.reset_parameters()
        self.query_projection.reset_parameters()
        self.key_projection.reset_parameters()
        self.value_projection.reset_parameters()
        self.relation_embedding.reset_parameters()
        if self.compositional_relation:
            self.compositional_dependency_embedding.reset_parameters()
            self.compositional_direction_embedding.reset_parameters()
            self.compositional_src_pos_embedding.reset_parameters()
            self.compositional_dst_pos_embedding.reset_parameters()
        self.dependency_bias.reset_parameters()
        self.pos_pair_bias.reset_parameters()
        self.output_projection.reset_parameters()
        self.gate_projection.reset_parameters()
        nn.init.zeros_(self.output_projection.weight)

    def _pool_word_hidden(
        self,
        hidden: torch.Tensor,
        word_to_subword: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, hidden_size = hidden.shape
        if word_to_subword.ndim != 3:
            raise ValueError("word_to_subword must have shape [batch, words, subwords]")
        if word_to_subword.size(0) != batch_size:
            raise ValueError("word_to_subword batch size mismatch")
        safe_indices = word_to_subword.clamp(min=0, max=max(0, token_count - 1))
        expanded_hidden = hidden.unsqueeze(1).expand(
            batch_size,
            word_to_subword.size(1),
            token_count,
            hidden_size,
        )
        gather_indices = safe_indices.unsqueeze(-1).expand(
            batch_size,
            word_to_subword.size(1),
            word_to_subword.size(2),
            hidden_size,
        )
        gathered = expanded_hidden.gather(2, gather_indices)
        valid = word_to_subword.ge(0) & word_to_subword.lt(token_count)
        valid = valid & word_mask.unsqueeze(-1).bool()
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        return (gathered * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    @staticmethod
    def _gather_nodes(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, _node_count, width = nodes.shape
        return nodes.gather(
            1,
            indices.unsqueeze(-1).expand(batch_size, indices.size(1), width),
        )

    def _graph_attention(
        self,
        projected_nodes: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        relation_id: torch.Tensor,
        dependency_relation_id: torch.Tensor,
        pos_pair_id: torch.Tensor,
        compositional_dependency_id: torch.Tensor | None,
        compositional_direction_id: torch.Tensor | None,
        compositional_src_pos_id: torch.Tensor | None,
        compositional_dst_pos_id: torch.Tensor | None,
        edge_mask: torch.Tensor,
        word_mask: torch.Tensor,
        trace=None,
    ) -> torch.Tensor:
        batch_size, node_count, _ = projected_nodes.shape
        edge_count = edge_src.size(1)
        valid_edges = edge_mask.bool()
        if (
            torch.any(valid_edges & edge_src.lt(0))
            or torch.any(valid_edges & edge_dst.lt(0))
            or torch.any(valid_edges & edge_src.ge(node_count))
            or torch.any(valid_edges & edge_dst.ge(node_count))
        ):
            raise ValueError("graph edge index is out of range")
        safe_src = edge_src.clamp(min=0, max=max(0, node_count - 1))
        safe_dst = edge_dst.clamp(min=0, max=max(0, node_count - 1))
        queries = self.query_projection(projected_nodes).view(
            batch_size, node_count, self.attention_heads, self.head_size
        )
        keys = self.key_projection(projected_nodes).view(
            batch_size, node_count, self.attention_heads, self.head_size
        )
        values = self.value_projection(projected_nodes).view(
            batch_size, node_count, self.attention_heads, self.head_size
        )
        _record_trace(trace, "query_projection", queries, ("batch", "node", "head", "feature"))
        _record_trace(trace, "key_projection", keys, ("batch", "node", "head", "feature"))
        _record_trace(trace, "value_projection", values, ("batch", "node", "head", "feature"))
        edge_queries = self._gather_nodes(queries.reshape(batch_size, node_count, -1), safe_dst).view(
            batch_size, edge_count, self.attention_heads, self.head_size
        )
        edge_keys = self._gather_nodes(keys.reshape(batch_size, node_count, -1), safe_src).view(
            batch_size, edge_count, self.attention_heads, self.head_size
        )
        edge_values = self._gather_nodes(values.reshape(batch_size, node_count, -1), safe_src).view(
            batch_size, edge_count, self.attention_heads, self.head_size
        )
        _record_trace(trace, "edge_query", edge_queries, ("batch", "edge", "head", "feature"))
        _record_trace(trace, "edge_key", edge_keys, ("batch", "edge", "head", "feature"))
        _record_trace(trace, "edge_value", edge_values, ("batch", "edge", "head", "feature"))
        relation_ids = relation_id.clamp(min=0, max=self.relation_embedding.num_embeddings - 1)
        dependency_ids = dependency_relation_id.clamp(min=0, max=self.dependency_bias.num_embeddings - 1)
        pos_ids = pos_pair_id.clamp(min=0, max=self.pos_pair_bias.num_embeddings - 1)
        if self.compositional_relation and compositional_dependency_id is not None:
            relation = (
                self.compositional_dependency_embedding(compositional_dependency_id.clamp(0, self.compositional_dependency_embedding.num_embeddings - 1))
                + self.compositional_direction_embedding(compositional_direction_id.clamp(0, self.compositional_direction_embedding.num_embeddings - 1))
                + self.compositional_src_pos_embedding(compositional_src_pos_id.clamp(0, self.compositional_src_pos_embedding.num_embeddings - 1))
                + self.compositional_dst_pos_embedding(compositional_dst_pos_id.clamp(0, self.compositional_dst_pos_embedding.num_embeddings - 1))
            ).view(batch_size, edge_count, self.attention_heads, self.head_size)
        else:
            relation = self.relation_embedding(relation_ids).view(batch_size, edge_count, self.attention_heads, self.head_size)
        if trace is None:
            logits = (edge_queries * edge_keys).sum(dim=-1) / math.sqrt(self.head_size)
            logits = logits + self.dependency_bias(dependency_ids) + self.pos_pair_bias(pos_ids)
        else:
            _record_trace(trace, "relation_embeddings", relation, ("batch", "edge", "head", "feature"))
            query_key_product = edge_queries * edge_keys
            _record_trace(trace, "query_key_product", query_key_product, ("batch", "edge", "head", "feature"))
            logits_before_scaling = query_key_product.sum(dim=-1)
            _record_trace(trace, "attention_logits_before_scaling", logits_before_scaling, ("batch", "edge", "head"))
            logits_scaled = logits_before_scaling / math.sqrt(self.head_size)
            _record_trace(trace, "attention_logits_scaled", logits_scaled, ("batch", "edge", "head"))
            dependency_bias = self.dependency_bias(dependency_ids)
            pos_pair_bias = self.pos_pair_bias(pos_ids)
            _record_trace(trace, "dependency_bias", dependency_bias, ("batch", "edge", "head"))
            _record_trace(trace, "pos_pair_bias", pos_pair_bias, ("batch", "edge", "head"))
            logits = logits_scaled + dependency_bias + pos_pair_bias
            _record_trace(trace, "final_attention_logits", logits, ("batch", "edge", "head"))
            softmax_input_float32 = logits.float()
            _record_trace(trace, "softmax_input_float32_logits", softmax_input_float32, ("batch", "edge", "head"))
            attention_probabilities = torch.zeros_like(softmax_input_float32)
            edge_messages = edge_values + relation
            _record_trace(trace, "edge_messages", edge_messages, ("batch", "edge", "head", "feature"))
        messages = torch.zeros(
            batch_size,
            node_count,
            self.attention_heads,
            self.head_size,
            dtype=projected_nodes.dtype,
            device=projected_nodes.device,
        )
        for batch_index in range(batch_size):
            for node_index in range(node_count):
                active = valid_edges[batch_index] & safe_dst[batch_index].eq(node_index)
                if not bool(word_mask[batch_index, node_index]) or not bool(active.any()):
                    continue
                attention = torch.softmax(logits[batch_index, active].float(), dim=0).to(projected_nodes.dtype)
                if trace is not None:
                    attention_probabilities[batch_index, active] = attention.float()
                message = edge_values[batch_index, active] + relation[batch_index, active]
                messages[batch_index, node_index] = (attention.unsqueeze(-1) * message).sum(dim=0)
        if trace is not None:
            _record_trace(trace, "attention_probabilities", attention_probabilities, ("batch", "edge", "head"))
            _record_trace(trace, "aggregated_messages", messages, ("batch", "node", "head", "feature"))
        return messages.reshape(batch_size, node_count, self.graph_hidden_size)

    def _broadcast_to_subwords(
        self,
        hidden: torch.Tensor,
        word_delta: torch.Tensor,
        word_to_subword: torch.Tensor,
        word_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        fused = hidden.clone()
        token_count = hidden.size(1)
        for batch_index in range(hidden.size(0)):
            for word_index in range(word_to_subword.size(1)):
                if not bool(word_mask[batch_index, word_index]):
                    continue
                for token_index in word_to_subword[batch_index, word_index].tolist():
                    if token_index < 0 or token_index >= token_count:
                        continue
                    if attention_mask is not None and not bool(attention_mask[batch_index, token_index]):
                        continue
                    fused[batch_index, token_index] = fused[batch_index, token_index] + word_delta[
                        batch_index, word_index
                    ]
        return fused

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        word_to_subword: torch.Tensor,
        word_mask: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        relation_id: torch.Tensor,
        dependency_relation_id: torch.Tensor,
        pos_pair_id: torch.Tensor,
        compositional_dependency_id: torch.Tensor | None = None,
        compositional_direction_id: torch.Tensor | None = None,
        compositional_src_pos_id: torch.Tensor | None = None,
        compositional_dst_pos_id: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        trace=None,
    ) -> GraphAdapterOutput:
        word_hidden = self._pool_word_hidden(hidden, word_to_subword, word_mask)
        _record_trace(trace, "pooled_word_hidden", word_hidden, ("batch", "node", "feature"))
        projected = self.node_projection(word_hidden)
        _record_trace(trace, "node_projection", projected, ("batch", "node", "feature"))
        graph_hidden = self._graph_attention(
            projected,
            edge_src,
            edge_dst,
            relation_id,
            dependency_relation_id,
            pos_pair_id,
            compositional_dependency_id,
            compositional_direction_id,
            compositional_src_pos_id,
            compositional_dst_pos_id,
            edge_mask,
            word_mask,
            trace=trace,
        )
        _record_trace(trace, "graph_hidden", graph_hidden, ("batch", "node", "feature"))
        graph_hidden = self.graph_dropout(graph_hidden)
        _record_trace(trace, "dropout_graph_hidden", graph_hidden, ("batch", "node", "feature"))
        _record_trace(trace, "output_projection_input", graph_hidden, ("batch", "node", "feature"))
        residual = self.output_projection(graph_hidden)
        _record_trace(trace, "output_projection_output", residual, ("batch", "node", "feature"))
        _record_trace(trace, "residual", residual, ("batch", "node", "feature"))
        gate = torch.sigmoid(self.gate_projection(torch.cat([word_hidden, residual], dim=-1)))
        if self.focus_enabled:
            # Focus is deliberately residual-only: it scales graph write-back
            # and never participates in RGAT attention logits.
            salience = torch.sigmoid(graph_hidden.float().norm(dim=-1, keepdim=True)).to(residual.dtype)
            _record_trace(trace, "element_salience", salience, ("batch", "node", "feature"))
        else:
            salience = torch.ones_like(gate[..., :1])
        _record_trace(trace, "gate", gate, ("batch", "node", "feature"))
        word_delta = gate * salience * residual
        _record_trace(trace, "word_delta", word_delta, ("batch", "node", "feature"))
        fused_hidden = self._broadcast_to_subwords(
            hidden,
            word_delta,
            word_to_subword,
            word_mask,
            attention_mask,
        )
        _record_trace(trace, "fused_hidden", fused_hidden, ("batch", "token", "feature"))
        return GraphAdapterOutput(
            fused_hidden=fused_hidden,
            graph_hidden=graph_hidden * word_mask.unsqueeze(-1).to(graph_hidden.dtype),
            word_hidden=word_hidden,
            residual=residual,
            gate=gate,
        )


def graph_model_config(config, relation_vocab_size: int) -> None:
    config.use_syntactic_graph_adapter = True
    config.graph_layers = 1
    config.graph_hidden_size = 256
    config.graph_attention_heads = 4
    config.graph_head_size = 64
    config.graph_relation_vocab_size = int(max(1, relation_vocab_size))
    config.graph_compositional_relation = True
    config.graph_compositional_dependency_vocab_size = 40
    config.graph_compositional_direction_vocab_size = 3
    config.graph_compositional_pos_vocab_size = 18
    config.graph_focus_enabled = bool(getattr(config, "graph_focus_enabled", False))
    config.graph_use_dependency = True
    config.graph_use_reverse_dependency = True
    config.graph_use_pos_neighbor = True
    config.graph_use_self_loop = True
    config.graph_external_word_embeddings = False
    config.graph_sentiment_embedding = False


def _graph_adapter_from_config(config) -> SyntacticGraphAdapter:
    return SyntacticGraphAdapter(
        hidden_size=int(config.d_model),
        graph_hidden_size=int(getattr(config, "graph_hidden_size", 256)),
        attention_heads=int(getattr(config, "graph_attention_heads", 4)),
        head_size=int(getattr(config, "graph_head_size", 64)),
        num_relations=int(getattr(config, "graph_relation_vocab_size", 1)),
        compositional_dependency_vocab_size=int(getattr(config, "graph_compositional_dependency_vocab_size", 40)),
        compositional_direction_vocab_size=int(getattr(config, "graph_compositional_direction_vocab_size", 3)),
        compositional_pos_vocab_size=int(getattr(config, "graph_compositional_pos_vocab_size", 18)),
        focus_enabled=bool(getattr(config, "graph_focus_enabled", False)),
        compositional_relation=bool(getattr(config, "graph_compositional_relation", True)),
        dropout=float(getattr(config, "dropout_rate", 0.1)),
    )


try:
    from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:  # pragma: no cover - only permits graph utility tests without transformers
    AutoModelForSeq2SeqLM = None
    T5ForConditionalGeneration = object
    BaseModelOutput = None


if AutoModelForSeq2SeqLM is not None:

    class SyntacticGraphT5ForConditionalGeneration(T5ForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            self.syntactic_graph_adapter = _graph_adapter_from_config(config)
            self.graph_parameter_initialization = {
                "initialization_mode": "constructor_default",
                "initialized_from_base_checkpoint": False,
                "graph_checkpoint_detected": False,
            }

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            """Load T5 weights while initializing only missing graph parameters."""
            caller_requested_loading_info = bool(kwargs.get("output_loading_info", False))
            kwargs["output_loading_info"] = True
            model, loading_info = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            )
            graph_parameter_names = {
                f"syntactic_graph_adapter.{name}"
                for name, _ in model.syntactic_graph_adapter.named_parameters()
            }
            missing_graph_parameters = graph_parameter_names.intersection(
                set(loading_info.get("missing_keys", []))
            )
            loaded_graph_parameters = graph_parameter_names - missing_graph_parameters
            if missing_graph_parameters and loaded_graph_parameters:
                missing = ", ".join(sorted(missing_graph_parameters))
                raise RuntimeError(
                    "checkpoint contains only a partial syntactic graph adapter; "
                    f"refusing to reset or overwrite graph parameters (missing: {missing})"
                )
            if missing_graph_parameters:
                model.syntactic_graph_adapter.reset_parameters()
                model.graph_parameter_initialization = {
                    "initialization_mode": "base_checkpoint_missing_graph_parameters",
                    "initialized_from_base_checkpoint": True,
                    "graph_checkpoint_detected": False,
                }
            else:
                model.graph_parameter_initialization = {
                    "initialization_mode": "graph_checkpoint_loaded",
                    "initialized_from_base_checkpoint": False,
                    "graph_checkpoint_detected": True,
                }
            if caller_requested_loading_info:
                return model, loading_info
            return model

        @property
        def use_syntactic_graph_adapter(self) -> bool:
            return True

        def _encode_with_graph(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            word_to_subword=None,
            word_mask=None,
            edge_src=None,
            edge_dst=None,
            relation_id=None,
            dependency_relation_id=None,
            pos_pair_id=None,
            compositional_dependency_id=None,
            compositional_direction_id=None,
            compositional_src_pos_id=None,
            compositional_dst_pos_id=None,
            edge_mask=None,
            trace=None,
        ):
            graph_fields = {
                "word_to_subword": word_to_subword,
                "word_mask": word_mask,
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "relation_id": relation_id,
                "dependency_relation_id": dependency_relation_id,
                "pos_pair_id": pos_pair_id,
                "compositional_dependency_id": compositional_dependency_id,
                "compositional_direction_id": compositional_direction_id,
                "compositional_src_pos_id": compositional_src_pos_id,
                "compositional_dst_pos_id": compositional_dst_pos_id,
                "edge_mask": edge_mask,
            }
            missing = [name for name, value in graph_fields.items() if value is None and not name.startswith("compositional_")]
            if self.syntactic_graph_adapter.compositional_relation:
                missing.extend(
                    name
                    for name in (
                        "compositional_dependency_id",
                        "compositional_direction_id",
                        "compositional_src_pos_id",
                        "compositional_dst_pos_id",
                    )
                    if graph_fields[name] is None
                )
            if missing:
                raise ValueError(f"syntactic graph inputs missing: {', '.join(missing)}")
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict=True,
            )
            _record_trace(
                trace,
                "t5_encoder_last_hidden_state",
                encoder_outputs.last_hidden_state,
                ("batch", "token", "feature"),
            )
            adapter_output = self.syntactic_graph_adapter(
                encoder_outputs.last_hidden_state,
                attention_mask=attention_mask,
                **graph_fields,
                trace=trace,
            )
            return BaseModelOutput(
                last_hidden_state=adapter_output.fused_hidden,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            graph_word_to_subword=None,
            graph_word_mask=None,
            graph_edge_src=None,
            graph_edge_dst=None,
            graph_relation_id=None,
            graph_dependency_relation_id=None,
            graph_pos_pair_id=None,
            graph_compositional_dependency_id=None,
            graph_compositional_direction_id=None,
            graph_compositional_src_pos_id=None,
            graph_compositional_dst_pos_id=None,
            graph_edge_mask=None,
            graph_trace=None,
            **kwargs,
        ):
            if encoder_outputs is None:
                encoder_outputs = self._encode_with_graph(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    word_to_subword=graph_word_to_subword,
                    word_mask=graph_word_mask,
                    edge_src=graph_edge_src,
                    edge_dst=graph_edge_dst,
                    relation_id=graph_relation_id,
                    dependency_relation_id=graph_dependency_relation_id,
                    pos_pair_id=graph_pos_pair_id,
                    compositional_dependency_id=graph_compositional_dependency_id,
                    compositional_direction_id=graph_compositional_direction_id,
                    compositional_src_pos_id=graph_compositional_src_pos_id,
                    compositional_dst_pos_id=graph_compositional_dst_pos_id,
                    edge_mask=graph_edge_mask,
                    trace=graph_trace,
                )
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if graph_trace is not None:
                if getattr(outputs, "logits", None) is not None:
                    _record_trace(
                        graph_trace,
                        "decoder_logits",
                        outputs.logits,
                        ("batch", "token", "feature"),
                    )
                if getattr(outputs, "loss", None) is not None:
                    _record_trace(graph_trace, "final_loss", outputs.loss, ())
            return outputs

        def generate(self, inputs=None, **kwargs):
            graph_trace = kwargs.pop("graph_trace", None)
            graph_names = (
                "graph_word_to_subword",
                "graph_word_mask",
                "graph_edge_src",
                "graph_edge_dst",
                "graph_relation_id",
                "graph_dependency_relation_id",
                "graph_pos_pair_id",
                "graph_compositional_dependency_id",
                "graph_compositional_direction_id",
                "graph_compositional_src_pos_id",
                "graph_compositional_dst_pos_id",
                "graph_edge_mask",
            )
            graph_fields = {name: kwargs.pop(name, None) for name in graph_names}
            if any(value is not None for value in graph_fields.values()):
                input_ids = kwargs.get("input_ids", inputs)
                attention_mask = kwargs.get("attention_mask")
                encoder_outputs = self._encode_with_graph(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **{
                        name.removeprefix("graph_"): value
                        for name, value in graph_fields.items()
                    },
                    trace=graph_trace,
                )
                kwargs["encoder_outputs"] = encoder_outputs
            return super().generate(inputs=inputs, **kwargs)


def load_seq2seq_model(
    model_path: str,
    use_syntactic_graph_adapter: bool = False,
    relation_vocab_size: int = 1,
    focus_enabled: bool | None = None,
    compositional_relation: bool = True,
):
    if not use_syntactic_graph_adapter:
        return AutoModelForSeq2SeqLM.from_pretrained(model_path)
    from transformers import AutoConfig

    checkpoint_relation_mode = _checkpoint_graph_relation_mode(model_path)
    if checkpoint_relation_mode is not None:
        compositional_relation = checkpoint_relation_mode
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    graph_model_config(config, relation_vocab_size)
    if focus_enabled is not None:
        config.graph_focus_enabled = bool(focus_enabled)
    config.graph_compositional_relation = bool(compositional_relation)
    return SyntacticGraphT5ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
    )
