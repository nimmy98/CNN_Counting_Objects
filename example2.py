# Generator code
import tensorflow_datasets as tfds
import tensorflow as tf

_DESCRIPTION = """
CNN For AI class, Fall 2021
"""

# TODO(example): BibTeX citation
#_CITATION = """
#"""

class Example2(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(58, 58, 3)),
            'label': tfds.features.ClassLabel(num_classes = 6),
        }),
        supervised_keys=('image', 'label'),
    )
  
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download dataset2.zip and place in tensorflow-datasets/downloads/manual
  """

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    archive_path = dl_manager.manual_dir / 'dataset2.zip'
    extracted_path = dl_manager.extract(archive_path)
    return {
        'train': self._generate_examples(extracted_path / 'dataset2/training'),
        'test': self._generate_examples(extracted_path / 'dataset2/testing'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    for f in path.glob('*.png'):
      yield f.name, {
          'image': f,
          'label': (int(f.name[:2])), # Takes first 2 digits of file name as label
      }
