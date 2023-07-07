from setuptools import setup

package_name = 'ground_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roboticsgroup1',
    maintainer_email='aidenneale@sky.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	      'img_publisher = ground_robot.webcam_pub:main',
	      'img_plate = ground_robot.license_plate_sub:main',
	      'img_aruco = ground_robot.aruco_sub:main',
          'img_yolo = ground_robot.yolo_sub:main',
        ],
    },
)
