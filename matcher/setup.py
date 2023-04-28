import setuptools
with open('README.md', 'r') as f:
    long_description = f.read()
    setuptools.setup(
        name='matcher',
        version='0.0.1',
        author='Minglei Cai',
        author_email='XXXX',
        description='XXXX',
        long_description='',
        long_description_content_type='text/markdown',
        packages=setuptools.find_packages(),
        python_requires='>=3.6',
    )