# ============================================================
# prepare_hybrid_training_data.py
# ============================================================
#
# Purpose:
#   Prepare leakage-free next-item training examples for the
#   Hybrid LightGCN + SASRec recommendation model.
#
# Method:
#   - Uses training-specific user/item IDs.
#   - Preserves chronological order.
#   - Uses item_index as deterministic same-day tie-breaker.
#   - Creates one next-item example for every possible
#     position in each user's training history.
#   - Uses only the history BEFORE the target.
#   - Truncates prefixes to the latest 50 interactions.
#   - Uses training items only for negative sampling.
#   - Excludes every item in the user's COMPLETE training
#     history from negative sampling.
#   - Uses SASRec padding ID = 0.
#
# IMPORTANT:
#   The global filtering rule was user >=5 interactions.
#   After temporal splitting, some users can legitimately have
#   fewer than 5 interactions inside the training period.
#   Therefore this script does NOT require training histories
#   to have >=5 interactions.
#
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_FILE = (
    DATA_DIR /
    "transactions_train_model_ids.parquet"
)

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

USERS_OUTPUT = (
    OUTPUT_DIR /
    "hybrid_train_users.npy"
)

POSITIVE_OUTPUT = (
    OUTPUT_DIR /
    "hybrid_train_positive_items.npy"
)

NEGATIVE_OUTPUT = (
    OUTPUT_DIR /
    "hybrid_train_negative_items.npy"
)

SEQUENCE_OUTPUT = (
    OUTPUT_DIR /
    "hybrid_train_sequences.npy"
)

METADATA_OUTPUT = (
    OUTPUT_DIR /
    "hybrid_training_data_metadata.json"
)


# ------------------------------------------------------------
# Training vocabulary
# ------------------------------------------------------------

NUM_TRAIN_USERS = 778_421
NUM_TRAIN_ITEMS = 74_541


# ------------------------------------------------------------
# Sequence configuration
# ------------------------------------------------------------

MAX_SEQUENCE_LENGTH = 50

# SASRec reserves ID 0 for padding.
#
# train_item_id:
#     0 ... 74540
#
# SASRec input:
#     1 ... 74541
#
PADDING_ID = 0
ITEM_ID_OFFSET = 1


# ------------------------------------------------------------
# Sampling configuration
# ------------------------------------------------------------
#
# First run:
#
#     5,000
#
# After sanity test passes:
#
#     2,000,000
#
NUM_EXAMPLES = 2_000_000

SEED = 42


# ============================================================
# Utility
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 2. Load training data
# ============================================================

print_header("[1] Loading training data")

validate_file_exists(TRAIN_FILE)

required_columns = [
    "user_index",
    "item_index",
    "train_user_id",
    "train_item_id",
    "t_dat",
]

df = pd.read_parquet(
    TRAIN_FILE,
    columns=required_columns,
)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Columns: {list(df.columns)}"
)

assert len(df) > 0, (
    "Training dataframe is empty."
)


# ============================================================
# 3. Validate training-specific IDs
# ============================================================

print_header(
    "[2] Validating training-specific IDs"
)

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

for column in required_columns:
    assert column in df.columns, (
        f"Missing required column: {column}"
    )


# ------------------------------------------------------------
# Missing-value checks
# ------------------------------------------------------------

assert df["train_user_id"].notna().all(), (
    "Missing train_user_id values found."
)

assert df["train_item_id"].notna().all(), (
    "Missing train_item_id values found."
)

assert df["item_index"].notna().all(), (
    "Missing item_index values found."
)

assert df["t_dat"].notna().all(), (
    "Missing t_dat values found."
)


# ------------------------------------------------------------
# Training-specific user IDs
# ------------------------------------------------------------

assert df["train_user_id"].min() >= 0

assert df["train_user_id"].max() < (
    NUM_TRAIN_USERS
)


# ------------------------------------------------------------
# Training-specific item IDs
# ------------------------------------------------------------

assert df["train_item_id"].min() >= 0

assert df["train_item_id"].max() < (
    NUM_TRAIN_ITEMS
)


# ------------------------------------------------------------
# Vocabulary counts
# ------------------------------------------------------------

num_users_in_data = (
    df["train_user_id"].nunique()
)

num_items_in_data = (
    df["train_item_id"].nunique()
)

print(
    f"Training users: "
    f"{num_users_in_data:,}"
)

print(
    f"Training items: "
    f"{num_items_in_data:,}"
)

print(
    f"Train user ID range: "
    f"{df['train_user_id'].min()} -> "
    f"{df['train_user_id'].max()}"
)

print(
    f"Train item ID range: "
    f"{df['train_item_id'].min()} -> "
    f"{df['train_item_id'].max()}"
)


assert num_users_in_data == (
    NUM_TRAIN_USERS
), (
    "Training user vocabulary mismatch."
)

assert num_items_in_data == (
    NUM_TRAIN_ITEMS
), (
    "Training item vocabulary mismatch."
)

print(
    "Training-specific ID validation: PASS"
)


# ============================================================
# 4. Chronological ordering
# ============================================================

print_header(
    "[3] Sorting chronological training histories"
)

df["t_dat"] = pd.to_datetime(
    df["t_dat"]
)

#
# Frozen dataset methodology:
#
#     user_index
#     t_dat
#     item_index
#
# are used for deterministic ordering.
#
# Here we use train_user_id for model-specific grouping,
# while preserving item_index as the same-day tie-breaker.
#

df = df.sort_values(
    by=[
        "train_user_id",
        "t_dat",
        "item_index",
    ],
    kind="mergesort",
).reset_index(
    drop=True
)

print(
    "Chronological sorting completed."
)


# ------------------------------------------------------------
# Validate chronological ordering
# ------------------------------------------------------------

#
# We check that dates never move backwards inside a user's
# history.
#

user_values = (
    df["train_user_id"].to_numpy()
)

date_values = (
    df["t_dat"].to_numpy()
)

chronological_pass = True

for i in range(1, len(df)):

    if user_values[i] == user_values[i - 1]:

        if date_values[i] < date_values[i - 1]:

            chronological_pass = False
            break


assert chronological_pass, (
    "Chronological ordering validation failed."
)

print(
    "Chronological ordering: PASS"
)


# ============================================================
# 5. Build complete training histories
# ============================================================

print_header(
    "[4] Building complete training histories"
)

#
# Each history contains the COMPLETE chronological training
# history for one training user.
#

user_histories = (
    df.groupby(
        "train_user_id",
        sort=True
    )["train_item_id"]
    .agg(list)
)

assert len(user_histories) == (
    NUM_TRAIN_USERS
), (
    "Not all training users have histories."
)


print(
    f"Users with histories: "
    f"{len(user_histories):,}"
)


# ------------------------------------------------------------
# History length statistics
# ------------------------------------------------------------

history_lengths = (
    user_histories.map(len)
)

print(
    f"Minimum history length: "
    f"{history_lengths.min():,}"
)

print(
    f"Maximum history length: "
    f"{history_lengths.max():,}"
)

print(
    f"Mean history length: "
    f"{history_lengths.mean():.2f}"
)


# ------------------------------------------------------------
# IMPORTANT:
# Do NOT assert minimum history >= 5 here.
# ------------------------------------------------------------
#
# The >=5 filtering threshold was applied to the FULL dataset
# before temporal train/validation/test splitting.
#
# Example:
#
# Full history:
#
#     A B C D E
#
# Training:
#
#     A B
#
# Validation:
#
#     C D
#
# Test:
#
#     E
#
# This user legitimately has only 2 training interactions.
#
# For next-item prediction:
#
#     N interactions -> N - 1 examples
#
# Therefore:
#
#     N = 1 -> 0 examples
#     N = 2 -> 1 example
#     N = 3 -> 2 examples
#
# ------------------------------------------------------------

users_with_examples = int(
    (history_lengths >= 2).sum()
)

users_without_examples = int(
    (history_lengths < 2).sum()
)

print(
    f"Users with >=2 training interactions: "
    f"{users_with_examples:,}"
)

print(
    f"Users with <2 training interactions: "
    f"{users_without_examples:,}"
)

assert (
    users_with_examples +
    users_without_examples
) == NUM_TRAIN_USERS

print(
    "Training-history eligibility: PASS"
)


# ============================================================
# 6. Calculate possible next-item examples
# ============================================================

print_header(
    "[5] Calculating possible next-item examples"
)

num_training_interactions = len(df)

num_training_users = len(
    user_histories
)

#
# For every user with N interactions:
#
#     number of next-item examples = N - 1
#
# Therefore:
#
#     sum(N - 1)
#       = sum(N) - number_of_users
#       = interactions - users
#
possible_examples = (
    num_training_interactions -
    num_training_users
)

expected_examples = 19_058_317

print(
    f"Training interactions: "
    f"{num_training_interactions:,}"
)

print(
    f"Training users: "
    f"{num_training_users:,}"
)

print(
    f"Possible next-item examples: "
    f"{possible_examples:,}"
)

assert possible_examples == (
    expected_examples
), (
    f"Unexpected example count. "
    f"Expected {expected_examples:,}, "
    f"got {possible_examples:,}"
)

if NUM_EXAMPLES > possible_examples:

    raise ValueError(
        f"NUM_EXAMPLES={NUM_EXAMPLES:,} "
        f"exceeds possible examples="
        f"{possible_examples:,}."
    )

print(
    "Possible-example count: PASS"
)


# ============================================================
# 7. Build cumulative example index
# ============================================================

print_header(
    "[6] Building example index"
)

#
# Example:
#
# User history:
#
#     [A, B, C, D]
#
# Examples:
#
#     [A]       -> B
#     [A, B]    -> C
#     [A, B, C] -> D
#
# Thus:
#
#     N history items -> N-1 examples
#
#

user_ids = (
    user_histories.index.to_numpy(
        dtype=np.int32
    )
)

history_lengths_array = (
    history_lengths.to_numpy(
        dtype=np.int64
    )
)

examples_per_user = (
    history_lengths_array - 1
)

#
# Users with one interaction have:
#
#     1 - 1 = 0
#
# examples.
#

assert np.all(
    examples_per_user >= 0
)

cumulative_examples = np.cumsum(
    examples_per_user,
    dtype=np.int64
)

assert (
    cumulative_examples[-1]
    == possible_examples
)

print(
    f"Example index built for "
    f"{len(user_ids):,} users."
)

print(
    "Example index: PASS"
)


# ============================================================
# 8. Sample examples
# ============================================================

print_header(
    "[7] Sampling training examples"
)

rng = np.random.default_rng(
    SEED
)

#
# Each integer in sample_offsets identifies one possible
# next-item training example across the entire training set.
#

sample_offsets = rng.choice(
    possible_examples,
    size=NUM_EXAMPLES,
    replace=False,
)

#
# Sorting makes generation deterministic and easier to inspect.
#

sample_offsets.sort()

print(
    f"Requested examples: "
    f"{NUM_EXAMPLES:,}"
)

print(
    f"Random seed: {SEED}"
)


# ============================================================
# 9. Allocate arrays
# ============================================================

print_header(
    "[8] Allocating output arrays"
)

sampled_users = np.empty(
    NUM_EXAMPLES,
    dtype=np.int32,
)

sampled_positive_items = np.empty(
    NUM_EXAMPLES,
    dtype=np.int32,
)

sampled_negative_items = np.empty(
    NUM_EXAMPLES,
    dtype=np.int32,
)

#
# Sequence:
#
#     shape = [NUM_EXAMPLES, 50]
#
#     0       = padding
#     1..74541 = real SASRec item IDs
#
sampled_sequences = np.zeros(
    (
        NUM_EXAMPLES,
        MAX_SEQUENCE_LENGTH,
    ),
    dtype=np.int32,
)

print(
    f"Sequence array shape: "
    f"{sampled_sequences.shape}"
)


# ============================================================
# 10. Generate leakage-free examples
# ============================================================

print_header(
    "[9] Generating leakage-free next-item examples"
)

#
# For a target position t:
#
#     complete history:
#
#     [i0, i1, i2, ..., i(t-1), i(t), ...]
#
#     prefix:
#
#     [i0, i1, ..., i(t-1)]
#
#     target:
#
#     i(t)
#
# The SASRec sequence contains ONLY the prefix.
#
# If prefix length > 50:
#
#     prefix[-50:]
#
# is retained.
#
# This prevents future/target leakage.
#

for sample_idx, global_offset in enumerate(
    sample_offsets
):

    # --------------------------------------------------------
    # Identify user
    # --------------------------------------------------------

    user_position = np.searchsorted(
        cumulative_examples,
        global_offset,
        side="right",
    )

    #
    # Safety check.
    #

    if user_position >= len(user_ids):

        user_position = (
            len(user_ids) - 1
        )

    # --------------------------------------------------------
    # Determine local target position
    # --------------------------------------------------------

    if user_position == 0:

        previous_cumulative = 0

    else:

        previous_cumulative = (
            cumulative_examples[
                user_position - 1
            ]
        )

    #
    # Example:
    #
    # global example offset
    #     = 10
    #
    # previous users contributed
    #     = 8 examples
    #
    # local example offset
    #     = 2
    #
    # target position
    #     = 3
    #
    local_example_offset = (
        global_offset -
        previous_cumulative
    )

    local_target_position = (
        local_example_offset + 1
    )

    user_id = int(
        user_ids[user_position]
    )

    history = user_histories.iloc[
        user_position
    ]

    history_length = len(history)

    assert (
        1 <= local_target_position
        < history_length
    )


    # --------------------------------------------------------
    # Prefix
    # --------------------------------------------------------

    prefix = history[
        :local_target_position
    ]

    # --------------------------------------------------------
    # Positive target
    # --------------------------------------------------------

    positive_item = int(
        history[
            local_target_position
        ]
    )


    # --------------------------------------------------------
    # Truncate prefix
    # --------------------------------------------------------
    #
    # Keep most recent 50 interactions.
    #

    prefix = prefix[
        -MAX_SEQUENCE_LENGTH:
    ]

    sequence_length = len(prefix)

    assert (
        1 <= sequence_length
        <= MAX_SEQUENCE_LENGTH
    )


    # --------------------------------------------------------
    # Convert to SASRec item IDs
    # --------------------------------------------------------
    #
    # train_item_id 0
    #     ->
    # SASRec ID 1
    #
    # train_item_id 74540
    #     ->
    # SASRec ID 74541
    #

    sequence_values = (
        np.asarray(
            prefix,
            dtype=np.int32
        )
        + ITEM_ID_OFFSET
    )

    #
    # Left padding:
    #
    # Example sequence length = 3:
    #
    #     [0, 0, ..., 0, A+1, B+1, C+1]
    #
    sampled_sequences[
        sample_idx,
        -sequence_length:
    ] = sequence_values


    # --------------------------------------------------------
    # Negative sampling
    # --------------------------------------------------------
    #
    # Negative candidate must:
    #
    #   1. be a valid training item
    #   2. not equal positive
    #   3. not occur anywhere in the user's complete
    #      training history
    #
    # IMPORTANT:
    #
    # We exclude the COMPLETE history, not just the prefix.
    #
    # This matches the corrected SASRec negative-sampling
    # methodology.
    #

    user_seen_items = set(
        history
    )

    while True:

        negative_item = int(
            rng.integers(
                0,
                NUM_TRAIN_ITEMS,
            )
        )

        if (
            negative_item ==
            positive_item
        ):
            continue

        if (
            negative_item in
            user_seen_items
        ):
            continue

        break


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    sampled_users[
        sample_idx
    ] = user_id

    sampled_positive_items[
        sample_idx
    ] = positive_item

    sampled_negative_items[
        sample_idx
    ] = negative_item


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (sample_idx + 1) % 1000 == 0
        or
        sample_idx == NUM_EXAMPLES - 1
    ):

        print(
            f"Generated "
            f"{sample_idx + 1:,} / "
            f"{NUM_EXAMPLES:,}"
        )


# ============================================================
# 11. Validate array shapes
# ============================================================

print_header(
    "[10] Validating generated training data"
)

assert sampled_users.shape == (
    NUM_EXAMPLES,
)

assert sampled_positive_items.shape == (
    NUM_EXAMPLES,
)

assert sampled_negative_items.shape == (
    NUM_EXAMPLES,
)

assert sampled_sequences.shape == (
    NUM_EXAMPLES,
    MAX_SEQUENCE_LENGTH,
)

print(
    "Array shapes: PASS"
)


# ============================================================
# 12. Validate user IDs
# ============================================================

assert sampled_users.min() >= 0

assert sampled_users.max() < (
    NUM_TRAIN_USERS
)

print(
    "User ID range: PASS"
)


# ============================================================
# 13. Validate positive items
# ============================================================

assert (
    sampled_positive_items.min()
    >= 0
)

assert (
    sampled_positive_items.max()
    < NUM_TRAIN_ITEMS
)

print(
    "Positive item range: PASS"
)


# ============================================================
# 14. Validate negative items
# ============================================================

assert (
    sampled_negative_items.min()
    >= 0
)

assert (
    sampled_negative_items.max()
    < NUM_TRAIN_ITEMS
)

print(
    "Negative item range: PASS"
)


# ============================================================
# 15. Validate positive != negative
# ============================================================

assert np.all(
    sampled_positive_items
    != sampled_negative_items
)

print(
    "Positive != negative: PASS"
)


# ============================================================
# 16. Validate SASRec sequence IDs
# ============================================================

assert (
    sampled_sequences.min()
    >= PADDING_ID
)

assert (
    sampled_sequences.max()
    <= NUM_TRAIN_ITEMS
)

print(
    "SASRec sequence ID range: PASS"
)


# ============================================================
# 17. Validate exact prefix-target relationship
# ============================================================

print_header(
    "[11] Validating prefix-target semantics"
)

#
# IMPORTANT:
#
# Do NOT check:
#
#     positive_item NOT IN prefix
#
# because repeated purchases are valid.
#
# Example:
#
#     history = [10, 20, 10]
#
# target = 10
#
# The target appears earlier in the prefix, and this is valid.
#
# Instead we verify:
#
#     stored prefix
#
# corresponds exactly to:
#
#     history[:target_position]
#
# truncated to the most recent 50 items,
#
# and that:
#
#     history[target_position]
#
# equals the stored positive item.
#
#
# We can reconstruct the original target position from the
# sampled example offset.
#

for sample_idx, global_offset in enumerate(
    sample_offsets
):

    # --------------------------------------------------------
    # Identify user
    # --------------------------------------------------------

    user_position = np.searchsorted(
        cumulative_examples,
        global_offset,
        side="right",
    )

    if user_position >= len(user_ids):

        user_position = (
            len(user_ids) - 1
        )

    # --------------------------------------------------------
    # Previous cumulative examples
    # --------------------------------------------------------

    if user_position == 0:

        previous_cumulative = 0

    else:

        previous_cumulative = (
            cumulative_examples[
                user_position - 1
            ]
        )

    # --------------------------------------------------------
    # Target position
    # --------------------------------------------------------

    local_example_offset = (
        global_offset -
        previous_cumulative
    )

    target_position = (
        local_example_offset + 1
    )

    user_id = int(
        user_ids[user_position]
    )

    history = user_histories.iloc[
        user_position
    ]

    # --------------------------------------------------------
    # Expected prefix
    # --------------------------------------------------------

    expected_prefix = history[
        :target_position
    ]

    expected_prefix = expected_prefix[
        -MAX_SEQUENCE_LENGTH:
    ]

    expected_prefix = np.asarray(
        expected_prefix,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Decode stored SASRec sequence
    # --------------------------------------------------------

    stored_sequence = (
        sampled_sequences[
            sample_idx
        ]
    )

    non_padding = (
        stored_sequence[
            stored_sequence != PADDING_ID
        ]
    )

    decoded_prefix = (
        non_padding -
        ITEM_ID_OFFSET
    )

    decoded_prefix = (
        decoded_prefix.astype(
            np.int32
        )
    )

    # --------------------------------------------------------
    # Compare prefix
    # --------------------------------------------------------

    assert np.array_equal(
        decoded_prefix,
        expected_prefix,
    ), (
        "Stored prefix does not match "
        "the expected chronological prefix."
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    expected_positive = int(
        history[target_position]
    )

    actual_positive = int(
        sampled_positive_items[
            sample_idx
        ]
    )

    assert (
        actual_positive ==
        expected_positive
    ), (
        "Positive item does not match "
        "the next chronological interaction."
    )


print(
    "Prefix -> positive target semantics: PASS"
)


# ============================================================
# 18. Validate negative sampling
# ============================================================

print_header(
    "[12] Validating negative sampling"
)

#
# Negative item must not appear anywhere in the user's complete
# training history.
#

for sample_idx in range(
    NUM_EXAMPLES
):

    user_id = int(
        sampled_users[
            sample_idx
        ]
    )

    negative_item = int(
        sampled_negative_items[
            sample_idx
        ]
    )

    history = user_histories.loc[
        user_id
    ]

    assert (
        negative_item not in
        set(history)
    ), (
        "Invalid negative item found "
        "inside complete user history."
    )


print(
    "Negative items absent from "
    "complete user histories: PASS"
)


# ============================================================
# 19. Validate padding convention
# ============================================================

print_header(
    "[13] Validating padding convention"
)

for sample_idx in range(
    NUM_EXAMPLES
):

    sequence = sampled_sequences[
        sample_idx
    ]

    seen_real_item = False

    for value in sequence:

        if value == PADDING_ID:

            #
            # Padding must occur only before real items.
            #
            assert not seen_real_item, (
                "Padding found after a "
                "real item. Expected left padding."
            )

        else:

            seen_real_item = True

            assert (
                1 <= value
                <= NUM_TRAIN_ITEMS
            )


print(
    "Left-padding convention: PASS"
)


# ============================================================
# 20. Validate sequence lengths
# ============================================================

sequence_lengths = np.count_nonzero(
    sampled_sequences,
    axis=1,
)

assert (
    sequence_lengths.min()
    >= 1
)

assert (
    sequence_lengths.max()
    <= MAX_SEQUENCE_LENGTH
)

print(
    f"Sequence length min: "
    f"{sequence_lengths.min()}"
)

print(
    f"Sequence length max: "
    f"{sequence_lengths.max()}"
)

print(
    f"Sequence length mean: "
    f"{sequence_lengths.mean():.2f}"
)

print(
    "Sequence lengths: PASS"
)


# ============================================================
# 21. Save arrays
# ============================================================

print_header(
    "[14] Saving training data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    USERS_OUTPUT,
    sampled_users,
)

np.save(
    POSITIVE_OUTPUT,
    sampled_positive_items,
)

np.save(
    NEGATIVE_OUTPUT,
    sampled_negative_items,
)

np.save(
    SEQUENCE_OUTPUT,
    sampled_sequences,
)


# ============================================================
# 22. Save metadata
# ============================================================

metadata = {

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    "dataset": (
        "H&M Personalized Fashion "
        "Recommendations"
    ),

    "source_file": str(
        TRAIN_FILE
    ),

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    "num_training_interactions": int(
        num_training_interactions
    ),

    "num_training_users": int(
        NUM_TRAIN_USERS
    ),

    "num_training_items": int(
        NUM_TRAIN_ITEMS
    ),

    "possible_next_item_examples": int(
        possible_examples
    ),

    "num_sampled_examples": int(
        NUM_EXAMPLES
    ),

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    "user_id_column": (
        "train_user_id"
    ),

    "item_id_column": (
        "train_item_id"
    ),

    "same_day_tiebreaker": (
        "item_index"
    ),

    "chronological_sort": [
        "train_user_id",
        "t_dat",
        "item_index",
    ],

    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    "max_sequence_length": int(
        MAX_SEQUENCE_LENGTH
    ),

    "padding_id": int(
        PADDING_ID
    ),

    "item_id_offset": int(
        ITEM_ID_OFFSET
    ),

    "sequence_construction": {

        "method": (
            "next_item_prediction"
        ),

        "prefix_before_target": True,

        "truncate_to_recent_items": True,

        "max_length": int(
            MAX_SEQUENCE_LENGTH
        ),

        "left_padding": True,

        "preserve_repeated_purchases": True,

        #
        # Repeated purchases mean the target may have appeared
        # earlier in the prefix.
        #
        "target_absence_from_prefix_required": False,
    },

    # --------------------------------------------------------
    # Negative sampling
    # --------------------------------------------------------

    "negative_sampling": {

        "candidate_vocabulary": (
            "training_items_only"
        ),

        "num_items": int(
            NUM_TRAIN_ITEMS
        ),

        "exclude_positive": True,

        "exclude_complete_user_training_history": True,
    },

    # --------------------------------------------------------
    # Leakage prevention
    # --------------------------------------------------------

    "leakage_prevention": {

        "future_interactions_excluded_from_prefix": True,

        "target_not_used_in_prefix": True,

        "future_history_not_used_for_sequence": True,

        "prefix_is_strictly_before_target": True,
    },

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    "sampling": {

        "method": (
            "uniform_without_replacement"
        ),

        "seed": int(
            SEED
        ),
    },

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    "output_files": {

        "users": str(
            USERS_OUTPUT
        ),

        "positive_items": str(
            POSITIVE_OUTPUT
        ),

        "negative_items": str(
            NEGATIVE_OUTPUT
        ),

        "sequences": str(
            SEQUENCE_OUTPUT
        ),

        "metadata": str(
            METADATA_OUTPUT
        ),
    },

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    "validation": {

        "array_shapes": True,

        "user_id_range": True,

        "positive_item_range": True,

        "negative_item_range": True,

        "positive_not_equal_negative": True,

        "sequence_id_range": True,

        "prefix_target_semantics": True,

        "negative_sampling": True,

        "padding_convention": True,

        "sequence_lengths": True,
    },
}


with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# 23. Final summary
# ============================================================

print_header(
    "[15] HYBRID TRAINING DATA PREPARATION COMPLETE"
)

print(
    f"Training interactions: "
    f"{num_training_interactions:,}"
)

print(
    f"Training users: "
    f"{NUM_TRAIN_USERS:,}"
)

print(
    f"Training items: "
    f"{NUM_TRAIN_ITEMS:,}"
)

print(
    f"Possible examples: "
    f"{possible_examples:,}"
)

print(
    f"Examples generated: "
    f"{NUM_EXAMPLES:,}"
)

print(
    f"Maximum sequence length: "
    f"{MAX_SEQUENCE_LENGTH}"
)

print(
    f"Sequence shape: "
    f"{sampled_sequences.shape}"
)

print()
print(
    f"Saved: {USERS_OUTPUT}"
)

print(
    f"Saved: {POSITIVE_OUTPUT}"
)

print(
    f"Saved: {NEGATIVE_OUTPUT}"
)

print(
    f"Saved: {SEQUENCE_OUTPUT}"
)

print(
    f"Saved: {METADATA_OUTPUT}"
)

print()
print("=" * 70)
print(
    "HYBRID TRAINING DATA PREPARATION PASSED"
)
print("=" * 70)