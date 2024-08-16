#!/bin/zsh
set -euxo pipefail

for file in data/external/brochure_removed_pages/*.pdf
do
  marker_single "$file" data/processed/brochure/ --langs German
done
