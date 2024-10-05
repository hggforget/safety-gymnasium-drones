import pathlib
import setuptools
setuptools.setup(
    name='safety-gymnasium-drones',
    version='0.0.0',
    description='safety-gymnasium-drones',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
)
