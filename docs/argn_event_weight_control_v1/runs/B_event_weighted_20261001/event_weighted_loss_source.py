def _calculate_sample_losses(
    model: FlatModel | SequentialModel | GradSampleModule, data: dict[str, torch.Tensor]
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
        output, _ = model(data, mode="trn")
    criterion = nn.CrossEntropyLoss(reduction="none")

    tgt_cols = (
        list(model.tgt_cardinalities.keys())
        if not isinstance(model, GradSampleModule)
        else model._module.tgt_cardinalities.keys()
    )
    if isinstance(model, SequentialModel) or (
        isinstance(model, GradSampleModule) and isinstance(model._module, SequentialModel)
    ):
        slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]

        # generate masks for SLEN and time step
        slen_mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.int64)
        for slen_col in slen_cols:
            slen_mask |= data[slen_col] != 0  # mask loss for padded rows, which have SLEN=0
        slen_mask = slen_mask.squeeze(-1)
        time_step_mask = torch.zeros_like(slen_mask, dtype=torch.int64)
        time_step_mask[:, 0] = 10  # mask loss for all time steps except the first one, and emphasize that one by 10x

        # calculate per column losses
        sidx_cols = {k for k in data if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        sdec_cols = {k for k in data if k.startswith(SDEC_SUB_COLUMN_PREFIX)}
        losses_by_column = []
        for col in tgt_cols:
            if col in slen_cols:
                # mask out SLEN for steps > 1
                mask = time_step_mask
            elif col in sidx_cols or col in sdec_cols:
                # SIDX and SDEC columns need to be present in the computation graph for DP to work
                # so we're only masking them instead of skipping them completely
                mask = torch.zeros_like(slen_mask, dtype=torch.int64)
            else:
                # mask out paddings
                mask = slen_mask

            column_loss = criterion(output[col].transpose(1, 2), data[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
            if col not in slen_cols and col not in sidx_cols and col not in sdec_cols:
                masked_loss = masked_loss * slen_mask.sum(dim=1) / 1318.7124394184168
            losses_by_column.append(masked_loss)
    else:
        losses_by_column = [criterion(output[col], data[col].squeeze(1)) for col in tgt_cols]
    # sum up column level losses to get overall losses at sample level
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)
    return losses
