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
            # List your dependencies here, if any
            # 'numpy>=1.19.2',
        ],
        python_requires='>=3.6',
    )

