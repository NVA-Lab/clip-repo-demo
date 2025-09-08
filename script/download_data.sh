# Create data directory structure
mkdir -p data/coco2017

# Download and save train images in parallel
wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip && unzip -q train2017.zip -d data/coco2017 &
# Download and save captions annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O captions_annotations_trainval2017.zip && unzip -j captions_annotations_trainval2017.zip -d data/coco2017/annotations &
# Download and save validation images in parallel
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip && unzip -q val2017.zip -d data/coco2017

# Wait for all background jobs to complete
wait

# Remove zip files after successful extraction
rm -f *.zip

echo "COCO 2017 dataset downloaded and extracted to data/coco2017/"