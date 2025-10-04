import pandas as pd
from collections import Counter
from typing import List, Union


def split_base_new_classes(index_csv_path, out_base_path, out_new_path, base_ratio=0.7):
    """
    split the dataset into base and new classes based on item frequency.
    """
    df = pd.read_csv(index_csv_path, header=None, names=["id", "item_name", "scene_description"])

    item_counts = Counter(df["item_name"])

    most_common = item_counts.most_common()
    candidate_items = [item for item, count in most_common if count > 40]
    least_items = [item for item, count in most_common if count <= 40]

    split_idx = int(len(candidate_items) * base_ratio)
    base_items = candidate_items[:split_idx] + least_items
    new_items = candidate_items[split_idx:]

    # base_ids = set(df[df['item_name'].isin(base_items)]['id'])
    # new_ids = set(df[df['item_name'].isin(new_items)]['id'])
    base_df = df[df["item_name"].isin(base_items)]
    new_df = df[df["item_name"].isin(new_items)]

    base_df.to_csv(out_base_path, index=False, header=False)
    new_df.to_csv(out_new_path, index=False, header=False)

    # return (base_ids, new_ids), (base_items, new_items)
    return


def check_ids_in_base_or_new(test_ids: List[Union[int, str]], base_ids: set, new_ids: set) -> dict:
    """
    For debug use. Checks the proportion of given IDs that belong to the base or new categories.
    """
    test_ids = set(test_ids)
    total = len(test_ids)

    base_count = len(test_ids & base_ids)
    new_count = len(test_ids & new_ids)

    base_ratio = base_count / total if total > 0 else 0
    new_ratio = new_count / total if total > 0 else 0

    if base_count == total:
        status = "base"
    elif new_count == total:
        status = "new"
    else:
        status = "mixed"

    return {
        "base_count": base_count,
        "new_count": new_count,
        "base_ratio": base_ratio,
        "new_ratio": new_ratio,
        "total": total,
        "status": status,
    }


def random_split(input_file, output_selected, output_remaining, select_num, random_state=None):
    df = pd.read_csv(input_file)

    if len(df) < select_num:
        raise ValueError(f"not enough rows to select {select_num} from {len(df)}")

    selected = df.sample(n=select_num, random_state=random_state)
    remaining = df.drop(selected.index)

    selected.to_csv(output_selected, index=False, header=False)
    remaining.to_csv(output_remaining, index=False, header=False)


if __name__ == "__main__":
    # 1. split seen and unseen based on item frequency
    # 2. split seen part into train and test_seen

    index_csv = "split/grasp_anything_match/all_filter_match.csv"
    out_seen_path = "split/grasp_anything_match/seen.csv"
    out_test_unseen_path = "split/grasp_anything_match/test_unseen.csv"
    out_train_path = "split/grasp_anything_match/train.csv"
    out_test_seen_path = "split/grasp_anything_match/test_seen.csv"

    split_base_new_classes(index_csv, out_seen_path, out_test_unseen_path)
    random_split(out_seen_path, out_test_seen_path, out_train_path, 3000)
