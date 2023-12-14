
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('# LPPL Model')
    return text[startpos:]


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


setup(name='lppl',
      version="0.0.2a1",
      description="LPPL Model",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Mathematics",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Intended Audience :: Financial and Insurance Industry"
      ],
      keywords="finance bubble",
      url="https://github.com/stephenhky/lppl",
      author="Kwan Yuet Stephen Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['lppl'],
      package_dir={'lppl': 'lppl'},
      install_requires=install_requirements(),
      test_require=[
          'unittest'
      ],
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
