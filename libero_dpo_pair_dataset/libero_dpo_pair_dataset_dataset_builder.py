from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_hub as hub
import math
from PIL import Image


def get_libero_image(obs, resize_size, key="agentview_image"):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[key]
    img = np.flipud(img)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)

def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

def preprocess_image(img):
    image = Image.fromarray(img)
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!

    temp_image = np.array(image)  # (H, W, C)
    crop_scale = 0.9
    sqrt_crop_scale = math.sqrt(crop_scale)
    temp_image_cropped = apply_center_crop(
        temp_image,
        t_h=int(sqrt_crop_scale * temp_image.shape[0]),
        t_w=int(sqrt_crop_scale * temp_image.shape[1]),
    )
    temp_image = Image.fromarray(temp_image_cropped)
    temp_image = temp_image.resize(
        image.size, Image.Resampling.BILINEAR
    )  # IMPORTANT: dlimp uses BILINEAR resize
    image = temp_image
    return image

class LiberoDpoPairDataset(tfds.core.GeneratorBasedBuilder):
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
                    'chosen_action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'rejected_action': tfds.features.Tensor(
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
        path = '/home/v-rusyang/shared_data/dataset/libero_minivla_raw_dataset_from_envlogger/COLLECT-libero_90-minivla-2025_02_24-08_19_33--collect libero90 data'
        return {
            'train': self._generate_examples(path=path+ "/*"),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'), # TODO: add val set, currently only train set is used
        }

    def split_success_and_false_trajectory(self, dataset: tf.data.Dataset):
        """Split the dataset into success and false trajectory."""

        def reduction_fn(episode):
            total_reward = 0.0
            for step in episode['steps']:
                total_reward += step['reward']
            episode_success = total_reward > 0
            episode['success'] = episode_success
            return episode

        dataset_with_success_label = dataset.map(reduction_fn)
        success_dataset = dataset_with_success_label.filter(lambda x: x['success'])
        false_dataset = dataset_with_success_label.filter(lambda x: not x['success'])
        def get_dataset_size(dataset):
            size = 0
            for _ in dataset:
                size += 1
            return size
        total_size = get_dataset_size(dataset)
        success_size = get_dataset_size(success_dataset)
        false_size = get_dataset_size(false_dataset)

        assert total_size == success_size + false_size, f"total_size: {total_size} != success_size: {success_size} + false_size: {false_size}"
        return success_dataset, false_dataset, success_size > 0 and false_size > 0

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        num_steps_wait = 10    # Number of steps to wait for objects to stabilize in sim
        def _parse_example(episode_path, true_episode, false_episode, language_instruction):
            success = False
            new_episode = []
            false_action_steps = []
            for step in false_episode['steps']:
                false_action_steps.append(step['action'])
            for i, success_step in enumerate(true_episode['steps']):
                if i < num_steps_wait: # skip the first few steps to wait for objects to stabilize in sim
                    continue
                if i >= len(false_action_steps): # if there are no more false action steps, break
                    print(f"Warning: no more false action steps for {episode_path},"
                          f" which means the false episode is too short.")
                    break

                # TODO: add language embedding
                language_embedding = np.zeros(512, dtype=np.float32)
                if not success:
                    success = success_step['is_last'] or success_step['reward'] > 0
                image = get_libero_image(success_step['observation'], 224, key='agentview_image')
                image = preprocess_image(image)
                wrist_image = get_libero_image(success_step['observation'], 224, key='robot0_eye_in_hand_image')
                wrist_image = preprocess_image(wrist_image)

                new_episode.append({
                    'observation': {
                        'image': np.array(image),
                        'wrist_image': np.array(wrist_image),
                        'state': success_step['observation']['robot0_proprio-state'].numpy(),
                    },
                    'chosen_action': success_step['action'].numpy().astype(np.float32),
                    'rejected_action': false_action_steps[i].numpy().astype(np.float32),
                    'discount': 1.0,
                    'reward': success_step['reward'].numpy(),
                    'is_first': success_step['is_first'].numpy(),
                    'is_last': success_step['is_last'].numpy(),
                    'is_terminal': success_step['is_terminal'].numpy(),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # NOTE: the final state is fake [0,0,0], so we remove it.
            assert np.all(new_episode[-1]['chosen_action'] == np.zeros(7)), f'Assume the last action is zero but got: {new_episode[-1]["chosen_action"]}'
            new_episode.pop(-1)

            # create output data sample
            sample = {
                'steps': new_episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'success': success.numpy()
                }
            }
            sample_id = episode_path + "_" + str(np.random.randint(0, 1000000))
            assert success, f"dpo pair dataset should only contain successful episodes, but {episode_path} is not successful."
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
            success_dataset, false_dataset, legally_split = self.split_success_and_false_trajectory(dataset)
            # ignore the task if it cannot be contructed into DPO pairs.
            if not legally_split:
                continue
            for i, true_episode in enumerate(success_dataset):
                for j, false_episode in enumerate(false_dataset):
                    episode_path = language_instruction + f"/{i}_{j}"
                    sample_id, sample = _parse_example(episode_path, true_episode, false_episode, language_instruction)
                    yield sample_id, sample

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

