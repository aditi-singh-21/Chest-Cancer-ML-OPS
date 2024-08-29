import setuptools

with open("README.md" , "r" , encoding="utf-8") as f:
    long_desc = f.read()
    
__version__ = "0.0.0"


REPO_NAME = "CHEST-CANCER-ML-OPS"
AUTHOR_USER_NAME = "aditi-singh-21"
SRC_REPO = "chest_cancer_classifier"
AUTHOR_EMAIL = "aditisinghrk906@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version= __version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= "A python package for CNN app",
    long_description= long_desc,
    long_description_content="text/markdown",
    url = f"https://github.com/aditi-singh-21/Chest-Cancer-ML-OPS",
    project_urls = {
        "Bug Tracker" : f"https://github.com/aditi-singh-21/Chest-Cancer-ML-OPS/issues",
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
)