def custom_collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    input_ids = [item['text']['input_ids'].squeeze(0) for item in batch]  # Remueve la dimensión de batch innecesaria.
    attention_masks = [item['text']['attention_mask'].squeeze(0) for item in batch]  # Lo mismo para las máscaras de atención.

    # Aplica padding al nivel de batch. Esto garantiza que todos los tensores tengan la misma longitud.
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    batched_graphs = GeoBatch.from_data_list(graphs)

    # Puedes retornar también las máscaras de atención si tu modelo las necesita.
    return batched_graphs, {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks}

