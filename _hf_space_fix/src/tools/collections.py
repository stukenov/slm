import pandas as pd
from huggingface_hub import add_collection_item, delete_collection_item, get_collection, update_collection_item
from huggingface_hub.utils._errors import HfHubHTTPError
from pandas import DataFrame

from src.display.utils import AutoEvalColumn, ModelType
from src.envs import H4_TOKEN, PATH_TO_COLLECTION

# Specific intervals for the collections
intervals = {
    "1B": pd.Interval(0, 1.5, closed="right"),
    "3B": pd.Interval(2.5, 3.5, closed="neither"),
    "7B": pd.Interval(6, 8, closed="neither"),
    "13B": pd.Interval(10, 14, closed="neither"),
    "30B": pd.Interval(25, 35, closed="neither"),
    "65B": pd.Interval(60, 70, closed="neither"),
}


def _filter_by_type_and_size(df, model_type, size_interval):
    """Filter DataFrame by model type and parameter size interval."""
    type_emoji = model_type.value.symbol[0]
    filtered_df = df[df[AutoEvalColumn.model_type_symbol.name] == type_emoji]
    params_column = pd.to_numeric(df[AutoEvalColumn.params.name], errors="coerce")
    mask = params_column.apply(lambda x: x in size_interval)
    return filtered_df.loc[mask]


def _add_models_to_collection(collection, models, model_type, size):
    """Add best models to the collection and update positions."""
    cur_len_collection = len(collection.items)
    for ix, model in enumerate(models, start=1):
        try:
            collection = add_collection_item(
                PATH_TO_COLLECTION,
                item_id=model,
                item_type="model",
                exists_ok=True,
                note=f"Best {model_type.to_str(' ')} model of around {size} on the leaderboard today!",
                token=H4_TOKEN,
            )
            # Ensure position is correct if item was added
            if len(collection.items) > cur_len_collection:
                item_object_id = collection.items[-1].item_object_id
                update_collection_item(collection_slug=PATH_TO_COLLECTION, item_object_id=item_object_id, position=ix)
                cur_len_collection = len(collection.items)
            break  # assuming we only add the top model
        except HfHubHTTPError:
            continue


def update_collections(df: DataFrame):
    """Update collections by filtering and adding the best models."""
    collection = get_collection(collection_slug=PATH_TO_COLLECTION, token=H4_TOKEN)
    cur_best_models = []

    for model_type in ModelType:
        if not model_type.value.name:
            continue
        for size, interval in intervals.items():
            filtered_df = _filter_by_type_and_size(df, model_type, interval)
            best_models = list(
                filtered_df.sort_values(AutoEvalColumn.average.name, ascending=False)[AutoEvalColumn.fullname.name][:10]
            )
            print(model_type.value.symbol, size, best_models)
            _add_models_to_collection(collection, best_models, model_type, size)
            cur_best_models.extend(best_models)

    # Cleanup
    existing_models = {item.item_id for item in collection.items}
    to_remove = existing_models - set(cur_best_models)
    for item_id in to_remove:
        try:
            delete_collection_item(collection_slug=PATH_TO_COLLECTION, item_object_id=item_id, token=H4_TOKEN)
        except HfHubHTTPError:
            continue
