from setuptools import setup, find_packages
from typing import List


HYPE_E_DOT = '-e .'
def get_requirements(filepath: str) -> List[str]:
    
    ''' This function reads the requirements file and returns a list of requirements. '''
    
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]
        
        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT)
        
        return requirements
        
    
setup(
    name='mlproject',
    version='0.0.1',
    author='Bikram',
    author_email='iiserkbikram@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
    