from syntactic_graph import UNK_RELATION_KEY, _assign_edge_ids, apply_relation_vocab
from syntactic_graph_adapter import graph_checkpoint_uses_compositional_relations


def test_relation_vocab_reserves_explicit_unknown_relation():
    records = [{"edges": [{"relation_key": "known", "dependency_key": "dep", "pos_pair_key": "NOUN|NOUN"}]}]
    vocab = apply_relation_vocab(records)
    assert vocab[-1] == UNK_RELATION_KEY
    edges = _assign_edge_ids(
        [{"relation_key": "unseen", "dependency_key": "dep", "pos_pair_key": "NOUN|NOUN"}],
        relation_vocab=vocab,
    )
    assert edges[0]["relation_id"] == vocab.index(UNK_RELATION_KEY)
    assert edges[0]["relation_id"] != 0


def test_legacy_graph_checkpoint_uses_legacy_relation_encoder():
    keys = {
        "syntactic_graph_adapter.relation_embedding.weight",
        "syntactic_graph_adapter.node_projection.weight",
    }

    assert graph_checkpoint_uses_compositional_relations(keys) is False


def test_new_graph_checkpoint_uses_compositional_relation_encoder():
    keys = {
        "syntactic_graph_adapter.relation_embedding.weight",
        "syntactic_graph_adapter.compositional_dependency_embedding.weight",
    }

    assert graph_checkpoint_uses_compositional_relations(keys) is True
