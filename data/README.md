# Data

The dataset used in this project is the **Last.fm-1K Dataset** (Celma, 2010).

- Official source: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
- GitHub mirror (preprocessed parquet format): https://github.com/eifuentes/lastfm-dataset-1K

The raw data files are **not included in this repository**. To download them, run:

```
scripts/download_data.sh
```

## Dataset Description

Contains `<user_id, timestamp, artist_id, artist_name, track_id, track_name>` tuples
collected from the Last.fm API using the `user.getRecentTracks()` method.
Represents the full listening history (up to May 2009) for ~1,000 users.

## License

Data distributed with permission of Last.fm for **non-commercial research use only**.

## Citation

Celma, O. (2010). *Music Recommendation and Discovery in the Long Tail*. Springer.
