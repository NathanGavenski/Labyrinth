from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='maze',
        version='0.1',
        description='A maze environment for simulations',
        author='Nathan Gavenski',
        author_email='nathangavenski@gmail.com',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        install_requires=[
            "gymnasium",
            "pygame",
            "tqdm",
            "pyglet",
            "numpy"
        ],
        python_requires='>=3.9',
    )

