import os
import numpy as np
import h5py
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import glob

# ===============================
# Resample
# ===============================
def resample_traj(traj: np.ndarray, n_points: int = 100) -> np.ndarray:
    T = traj.shape[0]
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, n_points)
    f = interp1d(old_t, traj, axis=0, kind="linear")
    return f(new_t)

# ===============================
# Feature extraction
# ===============================
def extract_features(traj: np.ndarray) -> np.ndarray:
    xyz = traj[:, :3]
    rot = traj[:, 3:6]
    grip = traj[:, 6]

    centroid = xyz.mean(axis=0)
    start, end = xyz[0], xyz[-1]
    direction = end - start
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    length = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))

    grip_ratio = grip.mean()
    grip_changes = np.sum(np.abs(np.diff(grip)) > 0.5)

    features = np.hstack([centroid, start, end, direction, length, grip_ratio, grip_changes])
    return features

# ===============================
# Clustering
# ===============================
def cluster_trajectories(trajs, n_clusters=3, n_points=100):
    # resampled = [resample_traj(t, n_points) for t in trajs]
    resampled = trajs
    features = np.vstack([extract_features(t) for t in resampled])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(features)
    return labels, features, resampled

def add_skill_id(path, out_dir):
    states = []
    keys = []
    
    # read all trajs
    with h5py.File(path, "r") as F:
        for key in F["data"]:
            states.append(F["data"][key]["robot_states"][()])
            keys.append(key)
    if len(states) == 0 or len(keys) == 0:
        print("skip", path)
        return
    labels, _, _ = cluster_trajectories(states, n_clusters=3, n_points=max(len(traj) for traj in states))
    assert len(labels) == len(keys)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))

    print("processing", path)
    with h5py.File(path, "r") as F_in, h5py.File(out_path, "w") as F_out:
        # copy all content
        F_in.copy("data", F_out)
    
        for key, label in zip(keys, labels):
            # write skill_id to each demo's root
            if "skill_id" in F_out["data"][key]:
                del F_out["data"][key]["skill_id"]
            F_out["data"][key].create_dataset("skill_id", data=label)


if __name__ == "__main__":
    paths = glob.glob("/mnt/shared_data/dataset/dataset_hdf5/libero_90_no_noops/*.hdf5")
    out_dir = "/mnt/shared_data/dataset/dataset_hdf5/libero_90_no_noops/libero_90_with_skill_id"
    
    for path in paths:
        # 检查输出文件是否已经存在
        out_path = os.path.join(out_dir, os.path.basename(path))
        
        if os.path.exists(out_path):
            print(f"skipping {path} (already processed)")
            continue
        
        add_skill_id(path, out_dir)
    
    # with h5py.File('/mnt/shared_data/dataset/dataset_hdf5/libero_90_no_noops/libero_90_with_skill_id/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5', "r") as F:
    #     for key in F["data"]:
    #         breakpoint()

# python libero_90_w_skill/cluster_traj.py