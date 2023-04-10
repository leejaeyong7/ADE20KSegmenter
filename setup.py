from pkg_resources import parse_version, parse_requirements
import setuptools
from pathlib import Path
assert parse_version(setuptools.__version__)>=parse_version('36.2')
with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with Path('requirements.txt').open() as requirements_txt:
    required= [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]
print(required)
setuptools.setup(
    name='ADE20KSegmentater',
    license=license,
    url='https://github.com/leejaeyong7/ADE20KSegmenter',
    packages = setuptools.find_packages(),
    install_requires = required,
    python_requires='>=3.6',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    version='0.0.1',
)