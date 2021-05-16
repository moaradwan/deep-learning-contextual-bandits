from zipfile import ZipFile
import pandas as pd
import tensorflow as tf
from collections import defaultdict

features_columns = [
    "Unnamed: 0",
    #     "timestamp",
    "item_id",
    "position",
    "click",
    "propensity_score",
    "user_feature_0",
    "user_feature_1",
    "user_feature_2",
    "user_feature_3",
    "user-item_affinity_0",
    "user-item_affinity_1",
    "user-item_affinity_2",
    "user-item_affinity_3",
    "user-item_affinity_4",
    "user-item_affinity_5",
    "user-item_affinity_6",
    "user-item_affinity_7",
    "user-item_affinity_8",
    "user-item_affinity_9",
    "user-item_affinity_10",
    "user-item_affinity_11",
    "user-item_affinity_12",
    "user-item_affinity_13",
    "user-item_affinity_14",
    "user-item_affinity_15",
    "user-item_affinity_16",
    "user-item_affinity_17",
    "user-item_affinity_18",
    "user-item_affinity_19",
    "user-item_affinity_20",
    "user-item_affinity_21",
    "user-item_affinity_22",
    "user-item_affinity_23",
    "user-item_affinity_24",
    "user-item_affinity_25",
    "user-item_affinity_26",
    "user-item_affinity_27",
    "user-item_affinity_28",
    "user-item_affinity_29",
    "user-item_affinity_30",
    "user-item_affinity_31",
    "user-item_affinity_32",
    "user-item_affinity_33",
    "weekday",
    "hour",
]
user_features_ids = list(
    filter(lambda x: x[1].startswith("user_feature"), enumerate(features_columns))
) + [
    (features_columns.index("weekday"), "weekday"),
    (features_columns.index("hour"), "hour"),
]
outputs_type = (
    tf.int32,
    #                 tf.string,
    tf.int32,
    tf.float32,
    tf.int32,
    tf.float32,
    tf.string,
    tf.string,
    tf.string,
    tf.string,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.int64,
    tf.int64,
)


def create_tables(vocabularies):
    global dv
    tables = {}
    for k in vocabularies:
        vocab = vocabularies[k]
        key_type = tf.string
        if len(vocab) > 0:
            if type(vocab[0]) == int:
                key_type = tf.int64
        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(vocab, dtype=key_type),
            values=tf.constant(range(0, len(vocab)), dtype=tf.int64),
        )

        table = tf.lookup.StaticVocabularyTable(
            init, 1, lookup_key_dtype=key_type, name=None
        )
        tables[k] = table
    dv = tables
    return tables


def get_vocabs(ds):
    vocabs = defaultdict(set)

    for batch in ds:
        for i, col in user_features_ids:
            vocabs[i].update(set(batch[i].numpy().tolist()))
    vocabs = {k: sorted(list(v)) for k, v in vocabs.items()}
    return vocabs


def get_raw_dataset(data_path, policies_campaigns, load_batch_size):
    files = [
        f"open_bandit_dataset/{policy}/{campaign}/{campaign}.csv"
        for policy in policies_campaigns
        for campaign in policies_campaigns[policy]
    ]
    ds = tf.data.Dataset.from_generator(
        get_transform(ZipFile(data_path), load_batch_size, files),
        output_types=outputs_type,
    )

    return ds


def get_dataset(data_path, policies_campaigns, load_batch_size):
    ds = get_raw_dataset(data_path, policies_campaigns, load_batch_size)
    tables = create_tables(get_vocabs(ds))
    ds = ds.map(get_numeric_transform(tables)).unbatch()
    return ds


def get_transform(data_zip, batch_size, campaigns):
    def _transform():
        for campaign in campaigns:
            with data_zip.open(campaign) as campaign_file:
                for batch in pd.read_csv(campaign_file, chunksize=batch_size):
                    # labels = batch.pop("click")
                    dt = batch.pop("timestamp")  # .str.split(r"[\.+]", n= 1)
                    # dt = dt.map(lambda x: datetime.datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))
                    dt = pd.to_datetime(dt, infer_datetime_format=True)
                    batch["weekday"] = dt.map(lambda x: x.weekday)
                    batch["hour"] = dt.map(lambda x: x.hour)
                    # ds = tf.data.Dataset.from_tensor_slices((dict(batch), labels))
                    # tf.print(batch.iloc[:, 0].to_numpy())
                    yield tuple(
                        [
                            batch.iloc[:, i].to_numpy().tolist()
                            for i in range(len(features_columns))
                        ]
                    )

    return _transform


def get_numeric_transform(tables):
    @tf.function
    def _transform(*batch):
        def table_map(item):
            if item[0] in tables:
                return tf.cast(
                    tf.one_hot(
                        tables[item[0]][item[1]],
                        tf.cast(tables[item[0]].size(), tf.int32),
                    ),
                    tf.float32,
                )
            return item[1]

        return tuple(map(table_map, enumerate(batch)))

    return _transform


if __name__ == "__main__":
    obd_path = "../../data/open_bandit_dataset.zip"
    obd_ds = get_dataset(obd_path, {"random": ["men"]}, 50000).take(1)
    for d in obd_ds.take(1):
        print(d)
        print(len(d))
