from setuptools import setup, find_packages
with open('requirements.txt') as f :
    content = f.readlines()
requirements = [x.strip() for x in content]
setup(name='moder_ia',
      description='morder_ia description',
      packages=[])
