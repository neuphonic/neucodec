from setuptools import setup, find_packages


setup(
    name='neucodec',
    version='0.0.1',
    description='A package for neucodec, based on xcodec2.',
    long_description_content_type='text/markdown',
    author='Harry Julian',
    author_email='harry@neuphonic.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.0.2',
        'torch>=2.5.1',
        'torchaudio>=2.5.1',
        'torchao',
        'torchtune>=0.3.1',
        'vector-quantize-pytorch==1.17.8',
        'rotary-embedding-torch>=0.8.4',
        'transformers>=4.44.2',
        'local_attention'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
