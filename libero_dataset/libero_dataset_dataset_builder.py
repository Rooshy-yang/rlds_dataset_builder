from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class LiberoDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(39,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'success': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the episode is successful.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        path = '/home/v-rusyang/openvla/experiments/logs/COLLECT-libero_90-minivla-2025_02_24-06_41_23--test'
        return {
            'train': self._generate_examples(path=path+ "/*"),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'), # TODO: add val set, currently only train set is used
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        num_steps_wait = 10    # Number of steps to wait for objects to stabilize in sim
        def _parse_example(episode_path, episode, language_instruction):
            success = False
            new_episode = []
            for i, step in enumerate(episode['steps']):
                if i < num_steps_wait:
                    continue
                # TODO: add language embedding
                language_embedding = np.zeros(512, dtype=np.float32)
                if not success:
                    success = step['is_last'] or step['reward'] > 0

                new_episode.append({
                    'observation': {
                        'image': step['observation']['agentview_image'].numpy(),
                        'wrist_image': step['observation']['robot0_eye_in_hand_image'].numpy(),
                        'state': step['observation']['robot0_proprio-state'].numpy(),
                    },
                    'action': step['action'].numpy(),
                    'discount': 1.0,
                    'reward': step['reward'].numpy(),
                    'is_first': step['is_first'].numpy(),
                    'is_last': step['is_last'].numpy(),
                    'is_terminal': step['is_terminal'].numpy(),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': new_episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'success': success.numpy()
                }
            }
            sample_id = episode_path + "_" + str(np.random.randint(0, 1000000))
            # NOTE: if you want to skip an example for whatever reason, simply return None, e.g. only success samples are used 
            return sample_id, sample

        # create list of all examples
        task_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for task_path in task_paths:
            # load raw data --> this should change for your dataset
            datasetbuilder = tfds.builder_from_directory(task_path)     # this is a list of dicts in our case
            language_instruction = datasetbuilder.info.name
            dataset = datasetbuilder.as_dataset(split='train')
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end

            for i, episode in enumerate(dataset):
                episode_path = language_instruction + f"/{i}"
                yield _parse_example(episode_path, episode, language_instruction)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

