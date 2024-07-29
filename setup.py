
from setuptools import setup


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


setup(url="https://github.com/stephenhky/lppl",
      install_requires=install_requirements(),
      test_require=['unittest'],
      test_suite="test",
      zip_safe=False)
