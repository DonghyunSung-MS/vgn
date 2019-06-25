""""Script to generate a synthetic grasp dataset using physical simulation."""

import argparse

from vgn.data_generator import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/dataset.hdf5',
                        help='Path to which the HDF5 dataset is written to')
    parser.add_argument('--n-scenes', type=int, default=1000,
                        help='Number of generated virtual scenes')
    parser.add_argument('--n-grasps-per-scene', type=int, default=100,
                        help='Number of grasps sampled per scene')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of processes used for the data generation')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualizations')
    args = parser.parse_args()

    generate_dataset(dataset_path=args.dataset_path,
                     n_scenes=args.n_scenes,
                     n_grasps_per_scene=args.n_grasps_per_scene,
                     n_workers=args.n_workers,
                     visualize=not args.no_visualize)


if __name__ == '__main__':
    main()
