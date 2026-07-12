#!/bin/bash
# Download Last.fm-1K Dataset (Celma, 2010)
# Source: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

mkdir -p data/raw
cd data/raw

echo "Downloading Last.fm 50-user pilot dataset (25MB)..."
curl -L -o lastfm-dataset-50.snappy.parquet \
  "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-50.snappy.parquet"

echo "Downloading user profile data..."
curl -L -o userid-profile.tsv.zip \
  "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/userid-profile.tsv.zip"

echo "Done. Files saved to data/raw/"

# To download the full 992-user dataset (877MB), uncomment below:
# echo "Downloading full 1K dataset (877MB)..."
# curl -L -o lastfm-dataset-1k.snappy.parquet \
#   "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-1k.snappy.parquet"
