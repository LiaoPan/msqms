

### 编译上传
$ python setup.py sdist bdist_wheel
$ twine upload dist/* --verbose


### 本地测试
$ python setup.py develop

### 本地环境清理
$ python setup.py clean


### 查找版本
https://shields.io/
https://badge.fury.io/