import datasets
from datasets.data_files import DataFilesDict
from datasets.packaged_modules.imagefolder.imagefolder import ImageFolder, ImageFolderConfig

logger = datasets.logging.get_logger(__name__)


class GTSRB(ImageFolder):
    R"""
    GTSRB dataset for image classification.
    """

    BUILDER_CONFIG_CLASS = ImageFolderConfig
    BUILDER_CONFIGS = [
        ImageFolderConfig(name='default', features=("images", "labels"), data_files=DataFilesDict({split: f"data/{split}.zip" for split in ["train", "test"] + ["contrast", "gaussian_noise", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "spatter"]}),)
    ]


    classnames = [
        "red and white circle 20 kph speed limit",
        "red and white circle 30 kph speed limit",
        "red and white circle 50 kph speed limit",
        "red and white circle 60 kph speed limit",
        "red and white circle 70 kph speed limit",
        "red and white circle 80 kph speed limit",
        "end / de-restriction of 80 kph speed limit",
        "red and white circle 100 kph speed limit",
        "red and white circle 120 kph speed limit",
        "red and white circle red car and black car no passing",
        "red and white circle red truck and black car no passing",
        "red and white triangle road intersection warning",
        "white and yellow diamond priority road",
        "red and white upside down triangle yield right-of-way",
        "stop",
        "empty red and white circle",
        "red and white circle no truck entry",
        "red circle with white horizonal stripe no entry",
        "red and white triangle with exclamation mark warning",
        "red and white triangle with black left curve approaching warning",
        "red and white triangle with black right curve approaching warning",
        "red and white triangle with black double curve approaching warning",
        "red and white triangle rough / bumpy road warning",
        "red and white triangle car skidding / slipping warning",
        "red and white triangle with merging / narrow lanes warning",
        "red and white triangle with person digging / construction / road work warning",
        "red and white triangle with traffic light approaching warning",
        "red and white triangle with person walking warning",
        "red and white triangle with child and person walking warning",
        "red and white triangle with bicyle warning",
        "red and white triangle with snowflake / ice warning",
        "red and white triangle with deer warning",
        "white circle with gray strike bar no speed limit",
        "blue circle with white right turn arrow mandatory",
        "blue circle with white left turn arrow mandatory",
        "blue circle with white forward arrow mandatory",
        "blue circle with white forward or right turn arrow mandatory",
        "blue circle with white forward or left turn arrow mandatory",
        "blue circle with white keep right arrow mandatory",
        "blue circle with white keep left arrow mandatory",
        "blue circle with white arrows indicating a traffic circle",
        "white circle with gray strike bar indicating no passing for cars has ended",
        "white circle with gray strike bar indicating no passing for trucks has ended",
    ]

    clip_templates = [
        lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
        lambda c: f'a centered photo of a "{c}" traffic sign.',
        lambda c: f'a close up photo of a "{c}" traffic sign.',
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="GTSRB dataset for image classification.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self.classnames),
                }
            ),
            supervised_keys=("image", "label"),
            task_templates=[datasets.ImageClassification(image_column="image", label_column="label")],
        )

