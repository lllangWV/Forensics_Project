from data_collection import collect_data
from data_preprocessing import preprocess_images


def data_pipeline():
    collect_data()
    preprocess_images()
    pass

if __name__ == '__main__':
    parent_dir = f"{os.sep}users{os.sep}lllang{os.sep}SCRATCH{os.sep}forensics"
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(f"The project directory is : {project_dir}")

    dataset_dir = f"{project_dir}{os.sep}datasets{os.sep}raw"
    raw_shared_dir=f"{dataset_dir}{os.sep}shared"
    preprocessed_images_dir = f"{dataset_dir}{os.sep}preprocessed_images"
    ncores = 20

