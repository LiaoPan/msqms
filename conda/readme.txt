1. 根据包名，创建meta.yaml和包名对应文件夹
$ conda skeleton pypi <在pypi上已上传的包名>

2. 构建conda包
$ conda build <在pypi上已上传的包名>  # 注意，老版本是使用conda-build这个命令， 
                                   # deprecated and will be removed in 4.0.0. Use `conda build` instead.

3. 查看构建的包路径：
/Users/reallo/opt/miniconda3/conda-bld/
/Users/reallo/opt/miniconda3/conda-bld/osx-64/opmqc-0.1.2-py38_0.tar.bz2

4.使用conda install 来安装本地构建的包; 注意：最好新建一个conda的虚拟环境：conda create -n my-conda-build-environment
$ conda install —use-local <在pypi上包名>


5. 将package转换到all platforms: you can convert it for use on other plarform by using the 2 build files,build.sh and bld.bat.
# conda convert 
- osx-64
- linux-32
- linux-64
- win-32
- win-64
- all
$ conda convert —platform all /Users/reallo/opt/miniconda3/conda-bld/osx-64/opmqc-0.1.2-py38_0.tar.bz2 -o outputdir/

6. Uploading new packages to Anaconda.org
# conda install anaconda-client
$ anaconda login

$ anaconda upload /Users/reallo/opt/miniconda3/conda-bld/osx-64/opmqc-0.1.2-py38_0.tar.bz2

# or
$ anaconda login
$ anaconda upload PACKAGE
NOTE: Replace PACKAGE with the name of the desired package.


# Tip
To save time, you can set conda to always upload a successful build to Anaconda.org with the command: conda config --set anaconda_upload yes.