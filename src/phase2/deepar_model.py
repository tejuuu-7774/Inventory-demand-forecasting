"""A small, readable DeepAR-style model implemented directly in PyTorch."""

from copy import deepcopy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .evaluation import PREDICTION_COLUMNS, validate_prediction_schema
from .kalman_features import projected_kalman_features


def set_random_seeds(seed):
    """Reset all supported random generators for one independent experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # CPU LSTM kernels routed through oneDNN can vary across independent
    # processes even with a single thread. The native PyTorch path is slower
    # but reproducible for this small teaching experiment.
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    # A single CPU thread avoids small run-to-run differences in recurrent
    # kernels on machines with different thread scheduling.
    torch.set_num_threads(1)


def choose_device(preference="mps"):
    """Use Apple GPU when requested and available, otherwise use the CPU."""

    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_scale_map(data, training_end, series_ids=None):
    """Calculate one demand scale per series using training observations only."""

    history = data[data["date"] <= pd.Timestamp(training_end)]
    if series_ids is not None:
        history = history[history["id"].isin(series_ids)]
    observed = history[history["observation_mask"].eq(1)]
    means = observed.groupby("id")["sales"].mean()
    return {series_id: max(1.0, float(value)) for series_id, value in means.items()}


def make_fallback_scales(data, training_end, training_series_ids):
    """Create category and global scales for series absent from training."""

    history = data[
        (data["date"] <= pd.Timestamp(training_end))
        & data["id"].isin(training_series_ids)
        & data["observation_mask"].eq(1)
    ]
    category_scale = history.groupby("cat_id")["sales"].mean().clip(lower=1.0).to_dict()
    global_scale = max(1.0, float(history["sales"].mean()))
    return category_scale, global_scale


def _one_sequence(series, end_position, context_length, horizon, dynamic_features, static_columns, scale):
    """Turn one time-series slice into one teacher-forced training example."""

    sequence_length = context_length + horizon
    start_position = end_position - sequence_length
    segment = series.iloc[start_position:end_position]

    previous_sales = series["sales"].iloc[start_position - 1 : end_position - 1].to_numpy(dtype="float32")
    dynamic = segment[dynamic_features].to_numpy(dtype="float32")
    target = segment["sales"].to_numpy(dtype="float32")
    loss_mask = np.zeros(sequence_length, dtype="float32")
    loss_mask[-horizon:] = segment["observation_mask"].to_numpy(dtype="float32")[-horizon:]
    static = segment.iloc[-1][[f"{column}_index" for column in static_columns]].to_numpy(dtype="int64")

    return previous_sales, dynamic, static, target, loss_mask, np.float32(scale)


def make_training_arrays(
    data,
    training_end,
    context_length,
    horizon,
    stride,
    dynamic_features,
    static_columns,
    scale_map,
    series_ids=None,
):
    """Create rolling sequence examples whose targets end before training_end."""

    selected = data[data["date"] <= pd.Timestamp(training_end)]
    if series_ids is not None:
        selected = selected[selected["id"].isin(series_ids)]

    examples = []
    minimum_end = context_length + horizon + 1
    for series_id, series in selected.groupby("id", sort=False):
        series = series.sort_values("date").reset_index(drop=True)
        endpoints = list(range(minimum_end, len(series) + 1, stride))
        if endpoints and endpoints[-1] != len(series):
            endpoints.append(len(series))
        for end_position in endpoints:
            examples.append(
                _one_sequence(
                    series,
                    end_position,
                    context_length,
                    horizon,
                    dynamic_features,
                    static_columns,
                    scale_map.get(series_id, 1.0),
                )
            )
    return _stack_examples(examples)


def make_origin_arrays(
    data,
    origins,
    context_length,
    horizon,
    dynamic_features,
    static_columns,
    scale_map,
    series_ids=None,
):
    """Create examples at explicitly supplied forecast origins."""

    selected = data
    if series_ids is not None:
        selected = data[data["id"].isin(series_ids)]
    examples = []

    for series_id, series in selected.groupby("id", sort=False):
        series = series.sort_values("date").reset_index(drop=True)
        for origin in origins:
            forecast_end = pd.Timestamp(origin) + pd.Timedelta(days=horizon)
            matching = series.index[series["date"].eq(forecast_end)]
            if len(matching) != 1:
                continue
            end_position = int(matching[0]) + 1
            if end_position < context_length + horizon + 1:
                continue
            examples.append(
                _one_sequence(
                    series,
                    end_position,
                    context_length,
                    horizon,
                    dynamic_features,
                    static_columns,
                    scale_map.get(series_id, 1.0),
                )
            )
    return _stack_examples(examples)


def _stack_examples(examples):
    if not examples:
        raise ValueError("No sequence examples were created.")
    names = ["previous_sales", "dynamic", "static", "target", "loss_mask", "scale"]
    return {name: np.stack([example[index] for example in examples]) for index, name in enumerate(names)}


class SequenceDataset(Dataset):
    """A thin wrapper that gives PyTorch one sequence at a time."""

    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays["target"])

    def __getitem__(self, index):
        return {
            "previous_sales": torch.tensor(self.arrays["previous_sales"][index]),
            "dynamic": torch.tensor(self.arrays["dynamic"][index]),
            "static": torch.tensor(self.arrays["static"][index]),
            "target": torch.tensor(self.arrays["target"][index]),
            "loss_mask": torch.tensor(self.arrays["loss_mask"][index]),
            "scale": torch.tensor(self.arrays["scale"][index]),
        }


class SimpleDeepAR(nn.Module):
    """Global autoregressive LSTM with a Negative Binomial output."""

    def __init__(
        self,
        number_of_dynamic_features,
        static_cardinalities,
        hidden_size=24,
        number_of_layers=1,
        embedding_size=4,
    ):
        super().__init__()
        self.static_names = list(static_cardinalities)
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, embedding_size)
                for name, cardinality in static_cardinalities.items()
            }
        )
        input_size = 1 + number_of_dynamic_features + embedding_size * len(self.static_names)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_size, 1)
        self.dispersion_head = nn.Linear(hidden_size, 1)

    def forward(self, previous_sales, dynamic, static, scale, hidden=None):
        """Return Negative Binomial mean and dispersion for every time step."""

        scaled_previous = torch.log1p(previous_sales / scale.unsqueeze(1)).unsqueeze(-1)
        pieces = [scaled_previous, dynamic]

        for column_index, name in enumerate(self.static_names):
            embedded = self.embeddings[name](static[:, column_index])
            repeated = embedded.unsqueeze(1).expand(-1, dynamic.shape[1], -1)
            pieces.append(repeated)

        model_input = torch.cat(pieces, dim=-1)
        output, hidden = self.lstm(model_input, hidden)
        mean = F.softplus(self.mean_head(output).squeeze(-1)) * scale.unsqueeze(1) + 1e-4
        dispersion = F.softplus(self.dispersion_head(output).squeeze(-1)) + 0.05
        return mean, dispersion, hidden


def negative_binomial_distribution(mean, dispersion):
    """Convert mean and dispersion to PyTorch's Negative Binomial parameters."""

    total_count = 1.0 / dispersion
    probabilities = mean / (mean + total_count)
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return torch.distributions.NegativeBinomial(total_count=total_count, probs=probabilities)


def masked_negative_binomial_loss(mean, dispersion, target, loss_mask):
    distribution = negative_binomial_distribution(mean, dispersion)
    negative_log_likelihood = -distribution.log_prob(target)
    denominator = loss_mask.sum().clamp_min(1.0)
    return (negative_log_likelihood * loss_mask).sum() / denominator


def _move_batch(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def train_deepar(
    model,
    training_arrays,
    validation_arrays,
    config,
    device,
):
    """Train with teacher forcing and keep the best validation checkpoint."""

    training_loader = DataLoader(
        SequenceDataset(training_arrays),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_arrays),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    model = model.to(device)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    epochs_without_improvement = 0
    history_rows = []
    training_start = time.perf_counter()

    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        training_losses = []
        for batch in training_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            mean, dispersion, _ = model(
                batch["previous_sales"],
                batch["dynamic"],
                batch["static"],
                batch["scale"],
            )
            loss = masked_negative_binomial_loss(
                mean,
                dispersion,
                batch["target"],
                batch["loss_mask"],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move_batch(batch, device)
                mean, dispersion, _ = model(
                    batch["previous_sales"],
                    batch["dynamic"],
                    batch["static"],
                    batch["scale"],
                )
                loss = masked_negative_binomial_loss(
                    mean,
                    dispersion,
                    batch["target"],
                    batch["loss_mask"],
                )
                validation_losses.append(float(loss.detach().cpu()))

        training_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        history_rows.append(
            {"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss}
        )

        if validation_loss < best_validation - 1e-4:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - training_start
    return model, pd.DataFrame(history_rows), elapsed


def save_checkpoint(model, path, model_metadata):
    """Save weights and a separate readable metadata JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(model_metadata, indent=2))
    return path, metadata_path


def _context_inputs(series, origin, context_length, dynamic_features, history_limit=None):
    history = series[series["date"] <= pd.Timestamp(origin)].sort_values("date")
    if history_limit is not None:
        history = history.tail(history_limit)
    context = history.tail(context_length)

    dynamic = context[dynamic_features].to_numpy(dtype="float32")
    sales = context["sales"].to_numpy(dtype="float32")
    if len(sales):
        earlier = series[series["date"] < context["date"].min()].sort_values("date").tail(1)
        first_previous = float(earlier["sales"].iloc[0]) if len(earlier) and history_limit is None else 0.0
        previous = np.concatenate([[first_previous], sales[:-1]]).astype("float32")
    else:
        previous = np.zeros(0, dtype="float32")

    padding = context_length - len(context)
    if padding > 0:
        dynamic = np.pad(dynamic, ((padding, 0), (0, 0)))
        previous = np.pad(previous, (padding, 0))
    return history, dynamic, previous


def forecast_one_series(
    model,
    series,
    origin,
    horizon,
    context_length,
    dynamic_features,
    static_columns,
    scale,
    number_of_samples,
    device,
    history_limit=None,
    use_history_scale=False,
):
    """Generate recursive sample paths and return q10, q50 and q90."""

    history, context_dynamic, context_previous = _context_inputs(
        series,
        origin,
        context_length,
        dynamic_features,
        history_limit=history_limit,
    )
    if use_history_scale:
        observed = history.loc[history["observation_mask"].eq(1), "sales"]
        if len(observed):
            scale = max(1.0, float(observed.mean()))
    future = series[
        (series["date"] > pd.Timestamp(origin))
        & (series["date"] <= pd.Timestamp(origin) + pd.Timedelta(days=horizon))
    ].sort_values("date")
    if len(future) != horizon:
        raise ValueError("A complete future horizon was not found.")

    future_dynamic = future[dynamic_features].copy().reset_index(drop=True)
    kalman_names = {
        "kalman_level_scaled",
        "kalman_variance_scaled",
        "kalman_innovation_scaled",
    }
    if kalman_names & set(dynamic_features):
        projected = projected_kalman_features(history, horizon)
        for column in kalman_names & set(dynamic_features):
            future_dynamic[column] = projected[column]

    static_values = future.iloc[0][[f"{column}_index" for column in static_columns]].to_numpy(dtype="int64")
    static_tensor = torch.tensor(static_values).unsqueeze(0).to(device)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device=device)
    context_previous_tensor = torch.tensor(context_previous).unsqueeze(0).to(device)
    context_dynamic_tensor = torch.tensor(context_dynamic).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        _, _, hidden = model(
            context_previous_tensor,
            context_dynamic_tensor,
            static_tensor,
            scale_tensor,
        )
        hidden = tuple(state.repeat(1, number_of_samples, 1) for state in hidden)
        static_samples = static_tensor.repeat(number_of_samples, 1)
        scale_samples = scale_tensor.repeat(number_of_samples)
        previous_value = float(history["sales"].iloc[-1]) if len(history) else 0.0
        previous_samples = torch.full(
            (number_of_samples, 1),
            previous_value,
            dtype=torch.float32,
            device=device,
        )
        sample_paths = []

        for step in range(horizon):
            dynamic_row = torch.tensor(
                future_dynamic.iloc[step].to_numpy(dtype="float32"),
                device=device,
            ).view(1, 1, -1)
            dynamic_samples = dynamic_row.repeat(number_of_samples, 1, 1)
            mean, dispersion, hidden = model(
                previous_samples,
                dynamic_samples,
                static_samples,
                scale_samples,
                hidden,
            )
            distribution = negative_binomial_distribution(mean[:, -1], dispersion[:, -1])
            sampled_sales = distribution.sample().clamp_min(0.0)
            sample_paths.append(sampled_sales.cpu().numpy())
            previous_samples = sampled_sales.unsqueeze(1)

    sample_paths = np.stack(sample_paths, axis=1)
    quantiles = np.quantile(sample_paths, [0.1, 0.5, 0.9], axis=0)
    return future, quantiles


def forecast_windows(
    model,
    data,
    windows,
    experiment_id,
    model_name,
    config,
    dynamic_features,
    static_columns,
    scale_map,
    fallback_category_scales=None,
    fallback_global_scale=1.0,
    series_ids=None,
    history_limit=None,
    use_history_scale=False,
    sample_seed=None,
    device=None,
):
    """Forecast every selected series at every rolling origin."""

    if device is None:
        device = torch.device("cpu")
    set_random_seeds(config["random_seed"] if sample_seed is None else sample_seed)
    selected = data
    if series_ids is not None:
        selected = data[data["id"].isin(series_ids)]
    rows = []

    for series_id, series in selected.groupby("id", sort=False):
        series = series.sort_values("date")
        info = series.iloc[0]
        scale = scale_map.get(series_id)
        if scale is None:
            scale = (fallback_category_scales or {}).get(info["cat_id"], fallback_global_scale)

        for window in windows.itertuples(index=False):
            future, quantiles = forecast_one_series(
                model=model,
                series=series,
                origin=window.origin,
                horizon=config["forecast_horizon"],
                context_length=config["context_length"],
                dynamic_features=dynamic_features,
                static_columns=static_columns,
                scale=scale,
                number_of_samples=config["number_of_forecast_samples"],
                device=device,
                history_limit=history_limit,
                use_history_scale=use_history_scale,
            )
            for step, (_, actual_row) in enumerate(future.iterrows(), start=1):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "Model": model_name,
                        "id": series_id,
                        "item_id": actual_row["item_id"],
                        "store_id": actual_row["store_id"],
                        "cat_id": actual_row["cat_id"],
                        "forecast_origin": pd.Timestamp(window.origin),
                        "forecast_date": actual_row["date"],
                        "horizon_step": step,
                        "actual": float(actual_row["sales"]),
                        "q10": float(quantiles[0, step - 1]),
                        "q50": float(quantiles[1, step - 1]),
                        "q90": float(quantiles[2, step - 1]),
                    }
                )

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    validate_prediction_schema(predictions)
    return predictions
