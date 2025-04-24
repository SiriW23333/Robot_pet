from setuptools import find_packages, setup

package_name = 'hand_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='231180167@smail.nju.edu.cn',
    description='receive information from hand gesture detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
              'hand_reader = hand_reader.hand_reader_node:main',
        ],
    },
)
