ImageColorization
=================

A project for CS229.

Installation
------------

1. Make sure you have virtualenv installed
   ```
   $ sudo easy_install virtualenv
   ```

2. Create and activate a virtualenv
   ```
   $ virtualenv env --no-site-packages
   $ source env/bin/activate
   ```

3. Install Python package dependencies
   ```
   pip install -r requirements.txt
   ```

4. Download and install OpenCV 
    - Source: http://opencv.org/
    - Installation instructions here: http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html  
Project Layout
--------------

- ```colorizer/```: The main colorizer package.  Most of the code is in here.
- ```tests/```: Test scripts

